#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(DESeq2))

args <- commandArgs(trailingOnly=TRUE)

getarg <- function(flag, default=NULL) {
  i <- match(flag, args)
  if (is.na(i)) return(default)
  if (i == length(args)) stop("Missing value after ", flag)
  args[[i + 1]]
}

req <- function(flag) {
  x <- getarg(flag)
  if (is.null(x)) stop("Required argument missing: ", flag)
  x
}

counts_file <- req("--counts")
manifest_file <- req("--manifest")
outdir <- req("--outdir")
contrast_name <- req("--contrast-name")
group_a <- req("--group-a")
group_b <- req("--group-b")
sample_col <- getarg("--sample-col", "sample_id")
gene_col <- getarg("--gene-col", "Geneid")
column_map_file <- getarg("--column-map", NULL)

continuous <- strsplit(getarg("--continuous", ""), ",", fixed=TRUE)[[1]]
categorical <- strsplit(getarg("--categorical", ""), ",", fixed=TRUE)[[1]]
continuous <- continuous[nzchar(continuous)]
categorical <- categorical[nzchar(categorical)]

dir.create(outdir, recursive=TRUE, showWarnings=FALSE)

read_manifest <- function(path) {
  if (grepl("\\.tsv$", path, ignore.case=TRUE)) {
    read.delim(path, check.names=FALSE, stringsAsFactors=FALSE)
  } else {
    read.csv(path, check.names=FALSE, stringsAsFactors=FALSE)
  }
}

load_counts <- function(path, gene_col="Geneid") {
  if (grepl("\\.rds$", path, ignore.case=TRUE)) {
    obj <- readRDS(path)

    if (inherits(obj, "DESeqDataSet")) {
      mat <- counts(obj, normalized=FALSE)
    } else if (is.matrix(obj)) {
      mat <- obj
    } else if (is.data.frame(obj)) {
      # If the first column is an explicit feature ID column, preserve it.
      if (gene_col %in% names(obj)) {
        feature_ids <- as.character(obj[[gene_col]])
        mat <- as.matrix(obj[, setdiff(names(obj), gene_col), drop=FALSE])
        rownames(mat) <- feature_ids
      } else if (!is.null(rownames(obj))) {
        mat <- as.matrix(obj)
      } else {
        stop("RDS data.frame has neither gene_col nor row names.")
      }
    } else if (inherits(obj, "SummarizedExperiment")) {
      mat <- assay(obj)
    } else {
      stop("Unsupported RDS object class: ", paste(class(obj), collapse=", "))
    }

    if (is.null(rownames(mat))) stop("RDS count matrix has no feature row names.")
    if (is.null(colnames(mat))) stop("RDS count matrix has no sample column names.")
    storage.mode(mat) <- "integer"
    return(mat)
  }

  d <- if (grepl("\\.tsv$", path, ignore.case=TRUE)) {
    read.delim(path, check.names=FALSE, stringsAsFactors=FALSE)
  } else {
    read.csv(path, check.names=FALSE, stringsAsFactors=FALSE)
  }

  if (!gene_col %in% names(d)) {
    # Robust fallback: first column is feature ID when gene_col not present.
    gene_col <- names(d)[1]
  }

  feature_ids <- as.character(d[[gene_col]])
  if (anyDuplicated(feature_ids)) stop("Duplicate feature IDs in count matrix.")

  mat <- as.matrix(d[, setdiff(names(d), gene_col), drop=FALSE])
  rownames(mat) <- feature_ids
  storage.mode(mat) <- "integer"
  mat
}

apply_column_map <- function(mat, map_file, manifest_ids) {
  if (is.null(map_file)) return(mat)
  if (!file.exists(map_file)) stop("Column-map file not found: ", map_file)

  mp <- read.csv(map_file, check.names=FALSE, stringsAsFactors=FALSE)

  # Identify target sample-ID column.
  target_candidates <- c("sample_id", "SampleID", "sampleID", "mapped_sample_id")
  target_col <- target_candidates[target_candidates %in% names(mp)][1]
  if (is.na(target_col)) {
    stop(
      "Could not identify sample ID column in column-map file. Columns: ",
      paste(names(mp), collapse=", ")
    )
  }

  # Identify source column by overlap with actual matrix column names.
  candidate_cols <- setdiff(names(mp), target_col)
  overlaps <- vapply(
    candidate_cols,
    function(x) sum(as.character(mp[[x]]) %in% colnames(mat)),
    integer(1)
  )

  if (length(overlaps) == 0 || max(overlaps) == 0) {
    stop("No column-map column overlaps count-matrix column names.")
  }

  source_col <- names(which.max(overlaps))
  lookup <- setNames(as.character(mp[[target_col]]), as.character(mp[[source_col]]))

  mapped <- unname(lookup[colnames(mat)])
  keep_old <- is.na(mapped) | mapped == ""
  mapped[keep_old] <- colnames(mat)[keep_old]
  colnames(mat) <- mapped

  # Only fail if manifest IDs are still not representable.
  missing <- setdiff(manifest_ids, colnames(mat))
  if (length(missing)) {
    stop(
      "After column mapping, manifest samples still missing from counts: ",
      paste(head(missing, 10), collapse=", ")
    )
  }

  mat
}

meta <- read_manifest(manifest_file)

if (!all(c(sample_col, "analysis_group") %in% names(meta))) {
  stop("Manifest lacks required sample/group columns.")
}

need <- unique(c(sample_col, "analysis_group", continuous, categorical))
miss <- setdiff(need, names(meta))
if (length(miss)) stop("Manifest missing: ", paste(miss, collapse=", "))

if (anyDuplicated(meta[[sample_col]])) stop("Duplicate sample IDs in manifest.")

count_mat <- load_counts(counts_file, gene_col=gene_col)
count_mat <- apply_column_map(
  count_mat,
  column_map_file,
  as.character(meta[[sample_col]])
)

missing_samples <- setdiff(as.character(meta[[sample_col]]), colnames(count_mat))
if (length(missing_samples)) {
  stop(
    "Manifest samples missing from counts: ",
    paste(head(missing_samples, 10), collapse=", ")
  )
}

# Exact manifest sample set/order.
count_mat <- count_mat[, as.character(meta[[sample_col]]), drop=FALSE]
stopifnot(identical(colnames(count_mat), as.character(meta[[sample_col]])))

# Group A is reference; positive LFC = group B higher than group A.
meta$analysis_group <- factor(meta$analysis_group, levels=c(group_a, group_b))
if (anyNA(meta$analysis_group)) {
  stop("Manifest contains analysis groups outside requested group A / group B.")
}

# Standardize continuous covariates within the exact analysis subset.
model_terms <- character()

for (v in continuous) {
  x <- suppressWarnings(as.numeric(meta[[v]]))
  if (anyNA(x)) stop("Missing/non-numeric continuous covariate: ", v)
  if (sd(x) == 0) stop("No variation in continuous covariate: ", v)

  zname <- paste0(v, "__z")
  meta[[zname]] <- as.numeric(scale(x))
  model_terms <- c(model_terms, zname)
}

for (v in categorical) {
  if (anyNA(meta[[v]])) stop("Missing categorical covariate: ", v)
  meta[[v]] <- factor(meta[[v]])
  if (nlevels(meta[[v]]) < 2) stop("Categorical covariate has <2 levels: ", v)
  model_terms <- c(model_terms, v)
}

model_terms <- c(model_terms, "analysis_group")
form <- as.formula(paste("~", paste(model_terms, collapse=" + ")))

X <- model.matrix(form, meta)
if (qr(X)$rank != ncol(X)) stop("Design matrix is not full rank.")

dds <- DESeqDataSetFromMatrix(
  countData=count_mat,
  colData=meta,
  design=form
)
dds <- DESeq(dds)

res <- results(
  dds,
  contrast=c("analysis_group", group_b, group_a),
  alpha=0.05
)

d <- as.data.frame(res)
d$feature_id <- rownames(d)
d$fdr_significant <- !is.na(d$padj) & d$padj < 0.05
d$direction <- ifelse(
  is.na(d$log2FoldChange),
  NA_character_,
  ifelse(
    d$log2FoldChange > 0,
    paste0(group_b, "_higher"),
    paste0(group_b, "_lower")
  )
)

d <- d[
  order(d$pvalue, na.last=TRUE),
  c("feature_id", setdiff(names(d), "feature_id"))
]

write.csv(
  d,
  file.path(outdir, paste0(contrast_name, "_DESeq2_all_results.csv")),
  row.names=FALSE
)

write.csv(
  d[d$fdr_significant, , drop=FALSE],
  file.path(outdir, paste0(contrast_name, "_DESeq2_FDR05.csv")),
  row.names=FALSE
)

write.csv(
  meta,
  file.path(outdir, paste0(contrast_name, "_sample_manifest.csv")),
  row.names=FALSE
)

summary <- data.frame(
  contrast=contrast_name,
  group_A=group_a,
  group_B=group_b,
  counts_file=counts_file,
  column_map=ifelse(is.null(column_map_file), "", column_map_file),
  n=nrow(meta),
  n_A=sum(meta$analysis_group == group_a),
  n_B=sum(meta$analysis_group == group_b),
  n_features=nrow(count_mat),
  design_rank=qr(X)$rank,
  design_columns=ncol(X),
  n_FDR05=sum(d$fdr_significant, na.rm=TRUE),
  stringsAsFactors=FALSE
)

write.csv(
  summary,
  file.path(outdir, "run_summary.csv"),
  row.names=FALSE
)

cat("RNA DESeq2 GENERALIZED RUN\n")
cat("Contrast:", contrast_name, "\n")
cat(
  "n =", nrow(meta),
  " | ", group_a, "=", sum(meta$analysis_group == group_a),
  " | ", group_b, "=", sum(meta$analysis_group == group_b),
  "\n",
  sep=""
)
cat("Features:", nrow(count_mat), "\n")
cat("Counts:", counts_file, "\n")
if (!is.null(column_map_file)) cat("Column map:", column_map_file, "\n")
cat("Design:", deparse(form), "\n")
cat("FDR<0.05:", summary$n_FDR05, "\n")
cat("Outdir:", outdir, "\n")
