#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(limma))

args <- commandArgs(trailingOnly=TRUE)

getarg <- function(flag, default=NULL) {
  i <- match(flag, args)
  if (is.na(i)) return(default)
  if (i == length(args)) stop("Missing value after ", flag)
  args[[i + 1]]
}

req <- function(flag) {
  x <- getarg(flag)
  if (is.null(x)) stop("Required: ", flag)
  x
}

matrix_file <- req("--matrix")
manifest_file <- req("--manifest")
outdir <- req("--outdir")
contrast_name <- req("--contrast-name")
group_a <- req("--group-a")
group_b <- req("--group-b")

sample_col <- getarg("--sample-col", "sample_id")

continuous <- strsplit(
  getarg("--continuous", "age_numeric,pmi_numeric"),
  ",",
  fixed=TRUE
)[[1]]

categorical <- strsplit(
  getarg("--categorical", "msex,Batch"),
  ",",
  fixed=TRUE
)[[1]]

continuous <- continuous[nzchar(continuous)]
categorical <- categorical[nzchar(categorical)]

expected_features <- as.integer(
  getarg("--expected-features", "5211")
)

dir.create(
  outdir,
  recursive=TRUE,
  showWarnings=FALSE
)

# ------------------------------------------------------------
# Load TMT matrix and manifest
# ------------------------------------------------------------

mat <- read.csv(
  matrix_file,
  check.names=FALSE,
  stringsAsFactors=FALSE
)

meta <- read.csv(
  manifest_file,
  check.names=FALSE,
  stringsAsFactors=FALSE
)

# The first CSV column contains biological protein IDs but has a blank
# header in this file, so address it by POSITION rather than by name.
protein_ids_all <- as.character(mat[[1]])

# The remaining 400 columns are TMT sample abundance columns.
all_samples <- names(mat)[-1]

Yall <- as.matrix(
  mat[, -1, drop=FALSE]
)
storage.mode(Yall) <- "numeric"

# Freeze the complete-protein universe across the full TMT matrix.
complete <- rowSums(is.na(Yall)) == 0

Y <- Yall[complete, , drop=FALSE]

protein_ids <- protein_ids_all[complete]

if (length(protein_ids) != nrow(Y)) {
  stop(
    "Protein-ID length does not match filtered matrix rows: ",
    length(protein_ids),
    " vs ",
    nrow(Y)
  )
}

if (anyNA(protein_ids) || any(protein_ids == "")) {
  stop("Missing protein IDs in filtered TMT matrix.")
}

if (anyDuplicated(protein_ids)) {
  stop("Duplicate protein IDs in filtered TMT matrix.")
}

# Explicitly retain biological protein identifiers.
rownames(Y) <- protein_ids

if (
  !is.na(expected_features) &&
  nrow(Y) != expected_features
) {
  stop(
    "Expected ",
    expected_features,
    " complete proteins; found ",
    nrow(Y)
  )
}

# ------------------------------------------------------------
# Validate manifest
# ------------------------------------------------------------

need <- c(
  sample_col,
  "analysis_group",
  continuous,
  categorical
)

miss <- setdiff(need, names(meta))

if (length(miss)) {
  stop(
    "Manifest missing: ",
    paste(miss, collapse=", ")
  )
}

if (anyDuplicated(meta[[sample_col]])) {
  stop("Duplicate samples in manifest.")
}

idx <- match(
  meta[[sample_col]],
  all_samples
)

if (anyNA(idx)) {
  stop("Manifest samples missing from TMT matrix.")
}

Ysub <- Y[, idx, drop=FALSE]

stopifnot(
  identical(
    colnames(Ysub),
    as.character(meta[[sample_col]])
  )
)

# Freeze IDs from the actual matrix entering lmFit.
model_protein_ids <- rownames(Ysub)

if (length(model_protein_ids) != nrow(Ysub)) {
  stop("Model protein IDs do not match Ysub rows.")
}

if (anyNA(model_protein_ids)) {
  stop("Missing model protein IDs.")
}

# ------------------------------------------------------------
# Build covariate-adjusted limma design
# ------------------------------------------------------------

meta$analysis_group <- factor(
  meta$analysis_group,
  levels=c(group_a, group_b)
)

terms <- character()

for (v in continuous) {
  x <- suppressWarnings(
    as.numeric(meta[[v]])
  )

  if (anyNA(x)) {
    stop(
      "Invalid continuous covariate: ",
      v
    )
  }

  zn <- paste0(v, "__z")
  meta[[zn]] <- as.numeric(scale(x))
  terms <- c(terms, zn)
}

for (v in categorical) {
  if (anyNA(meta[[v]])) {
    stop(
      "Missing categorical covariate: ",
      v
    )
  }

  meta[[v]] <- factor(meta[[v]])

  if (nlevels(meta[[v]]) < 2) {
    stop(
      "Categorical covariate <2 levels: ",
      v
    )
  }

  terms <- c(terms, v)
}

terms <- c(
  "analysis_group",
  terms
)

form <- as.formula(
  paste(
    "~",
    paste(terms, collapse=" + ")
  )
)

design <- model.matrix(
  form,
  meta
)

if (qr(design)$rank != ncol(design)) {
  stop("Design matrix not full rank.")
}

# ------------------------------------------------------------
# Fit limma
# ------------------------------------------------------------

fit <- lmFit(
  Ysub,
  design
)

fit <- eBayes(
  fit,
  trend=TRUE
)

coef_name <- grep(
  paste0(
    "^analysis_group",
    make.names(group_b),
    "$"
  ),
  colnames(design),
  value=TRUE
)

if (length(coef_name) != 1) {
  coef_name <- grep(
    "^analysis_group",
    colnames(design),
    value=TRUE
  )
}

if (length(coef_name) != 1) {
  stop(
    "Could not uniquely identify group coefficient: ",
    paste(coef_name, collapse=", ")
  )
}

# IMPORTANT:
# sort.by="none" keeps rows in exactly the same order as Ysub.
# We therefore attach biological protein IDs positionally BEFORE
# applying any output sorting.
res <- topTable(
  fit,
  coef=coef_name,
  number=Inf,
  adjust.method="BH",
  sort.by="none"
)

if (nrow(res) != length(model_protein_ids)) {
  stop(
    "limma result row count does not match model protein IDs: ",
    nrow(res),
    " vs ",
    length(model_protein_ids)
  )
}

res$feature_id <- model_protein_ids

if (
  anyNA(res$feature_id) ||
  any(res$feature_id == "")
) {
  stop("Missing protein IDs in final limma results.")
}

if (anyDuplicated(res$feature_id)) {
  stop("Duplicate protein IDs in final limma results.")
}

# Return to conventional P-value ordering after IDs are attached.
res <- res[
  order(res$P.Value, na.last=TRUE),
  ,
  drop=FALSE
]

res <- res[, c(
  "feature_id",
  "logFC",
  "AveExpr",
  "t",
  "P.Value",
  "adj.P.Val",
  "B"
)]

# ------------------------------------------------------------
# Write outputs
# ------------------------------------------------------------

write.csv(
  res,
  file.path(
    outdir,
    paste0(
      contrast_name,
      "_limma_all_results.csv"
    )
  ),
  row.names=FALSE
)

sig <- res[
  !is.na(res$adj.P.Val) &
    res$adj.P.Val < 0.05,
  ,
  drop=FALSE
]

write.csv(
  sig,
  file.path(
    outdir,
    paste0(
      contrast_name,
      "_limma_FDR05.csv"
    )
  ),
  row.names=FALSE
)

write.csv(
  meta,
  file.path(
    outdir,
    paste0(
      contrast_name,
      "_sample_manifest.csv"
    )
  ),
  row.names=FALSE
)

summary <- data.frame(
  contrast=contrast_name,
  group_A=group_a,
  group_B=group_b,
  n=nrow(meta),
  n_A=sum(
    meta$analysis_group == group_a
  ),
  n_B=sum(
    meta$analysis_group == group_b
  ),
  n_features=nrow(Ysub),
  design_rank=qr(design)$rank,
  design_columns=ncol(design),
  condition_number=kappa(
    design,
    exact=TRUE
  ),
  n_FDR05=nrow(sig)
)

write.csv(
  summary,
  file.path(
    outdir,
    "run_summary.csv"
  ),
  row.names=FALSE
)

cat("TMT LIMMA GENERALIZED RUN\n")
cat("Contrast:", contrast_name, "\n")
cat(
  "n =",
  nrow(meta),
  " | ",
  group_a,
  "=",
  sum(meta$analysis_group == group_a),
  " | ",
  group_b,
  "=",
  sum(meta$analysis_group == group_b),
  "\n",
  sep=""
)
cat(
  "Proteins:",
  nrow(Ysub),
  "\n"
)
cat(
  "FDR<0.05:",
  nrow(sig),
  "\n"
)
cat(
  "Outdir:",
  outdir,
  "\n"
)
