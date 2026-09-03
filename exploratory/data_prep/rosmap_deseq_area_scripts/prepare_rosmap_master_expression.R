#!/usr/bin/env Rscript

# ==============================================================================
# prepare_rosmap_master_expression.R
# ==============================================================================
# Build the master ROSMAP bulk RNA-seq expression dataset used by BOTH AREA and
# downstream DESeq2 contrasts.
#
# Core design:
#   raw counts (all RNA-seq samples)
#       -> align to metadata
#       -> DESeq2 size-factor normalization across the full cohort
#       -> global expression filter: mean normalized count > threshold
#       -> save the SAME retained genes as:
#            1) raw counts for DESeq2
#            2) normalized counts for AREA
#
# No phenotype-specific sample filtering occurs here.
# No VST/rlog values are used for AREA.
# ==============================================================================

usage <- function() {
  cat(paste0(
    "Usage:\n",
    "  Rscript prepare_rosmap_master_expression.R \\\n",
    "    --counts /path/to/count_matrix.rds \\\n",
    "    --metadata /path/to/Analysis_Meta_Merged.csv \\\n",
    "    --outdir results/preprocessing [--threshold 1]\n\n",
    "Required:\n",
    "  --counts      RDS file containing raw integer count matrix (genes x samples)\n",
    "  --metadata    CSV containing sample_id and ROSMAP metadata\n",
    "  --outdir      Output directory\n\n",
    "Optional:\n",
    "  --threshold   Mean DESeq2-normalized count threshold; default = 1\n",
    "  --help        Show this message\n"
  ))
}

parse_args <- function(args) {
  if (length(args) == 0 || any(args %in% c("--help", "-h"))) {
    usage()
    quit(status = ifelse(length(args) == 0, 1, 0))
  }

  out <- list()
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) stop("Unexpected argument: ", key)
    if (i == length(args)) stop("Missing value for argument: ", key)
    out[[substring(key, 3)]] <- args[[i + 1]]
    i <- i + 2
  }
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

required <- c("counts", "metadata", "outdir")
missing_required <- required[!required %in% names(args)]
if (length(missing_required) > 0) {
  stop("Missing required argument(s): ", paste(paste0("--", missing_required), collapse = ", "))
}

count_file <- path.expand(args$counts)
metadata_file <- path.expand(args$metadata)
outdir <- args$outdir
threshold <- if (!is.null(args$threshold)) as.numeric(args$threshold) else 1

if (!is.finite(threshold) || threshold < 0) {
  stop("--threshold must be a finite non-negative number.")
}

if (!file.exists(count_file)) stop("Count file not found: ", count_file)
if (!file.exists(metadata_file)) stop("Metadata file not found: ", metadata_file)

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("DESeq2", quietly = TRUE)) {
  stop("DESeq2 is required. Install it with BiocManager::install('DESeq2').")
}

suppressPackageStartupMessages(library(DESeq2))

cat("\n============================================================\n")
cat("ROSMAP MASTER RNA-seq PREPROCESSING\n")
cat("============================================================\n")
cat("Counts:   ", count_file, "\n", sep = "")
cat("Metadata: ", metadata_file, "\n", sep = "")
cat("Outdir:   ", outdir, "\n", sep = "")
cat("Global filter: mean normalized count > ", threshold, "\n\n", sep = "")

# ------------------------------------------------------------------------------
# 1. Load and validate raw counts
# ------------------------------------------------------------------------------
count_matrix <- readRDS(count_file)

if (!is.matrix(count_matrix)) {
  stop("The count RDS must contain a matrix. Found class: ", paste(class(count_matrix), collapse = ", "))
}
if (is.null(rownames(count_matrix)) || is.null(colnames(count_matrix))) {
  stop("Count matrix must have both row names (genes) and column names (samples).")
}
if (anyNA(count_matrix)) stop("Raw count matrix contains NA values.")
if (any(count_matrix < 0)) stop("Raw count matrix contains negative values.")
if (!all(count_matrix == floor(count_matrix))) stop("Raw count matrix contains non-integer values.")
if (anyDuplicated(rownames(count_matrix))) stop("Raw count matrix contains duplicate feature IDs.")

cat("Raw count matrix: ", nrow(count_matrix), " features x ", ncol(count_matrix), " samples\n", sep = "")
cat("Raw-count validation: PASSED\n")

# ------------------------------------------------------------------------------
# 2. Clean count-matrix sample IDs
# ------------------------------------------------------------------------------
original_sample_names <- colnames(count_matrix)
sample_ids <- basename(original_sample_names)
sample_ids <- sub("\\.bam$", "", sample_ids)

if (anyDuplicated(sample_ids)) {
  dupes <- unique(sample_ids[duplicated(sample_ids)])
  stop("Duplicate sample IDs after removing paths/.bam: ", paste(head(dupes, 20), collapse = ", "))
}
colnames(count_matrix) <- sample_ids

# Save provenance mapping from original BAM/path column to cleaned sample ID.
write.csv(
  data.frame(original_count_column = original_sample_names, sample_id = sample_ids),
  file.path(outdir, "ROSMAP_count_column_to_sample_id.csv"),
  row.names = FALSE
)

# ------------------------------------------------------------------------------
# 3. Load metadata and align EXACTLY to count matrix
# ------------------------------------------------------------------------------
meta <- read.csv(metadata_file, stringsAsFactors = FALSE, check.names = FALSE)

required_meta <- c("sample_id", "individual_id", "diagnosis", "braak", "cerad", "sex", "age", "rin", "pmi")
missing_meta <- setdiff(required_meta, colnames(meta))
if (length(missing_meta) > 0) {
  stop("Metadata is missing required column(s): ", paste(missing_meta, collapse = ", "))
}
if (anyDuplicated(meta$sample_id)) {
  dupes <- unique(meta$sample_id[duplicated(meta$sample_id)])
  stop("Metadata contains duplicate sample_id values: ", paste(head(dupes, 20), collapse = ", "))
}

matched <- sample_ids %in% meta$sample_id
cat("RNA-seq samples matching metadata: ", sum(matched), "/", length(sample_ids), "\n", sep = "")

if (!all(matched)) {
  missing_samples <- sample_ids[!matched]
  writeLines(missing_samples, file.path(outdir, "ROSMAP_samples_missing_metadata.txt"))
  stop("Not every count-matrix sample matched metadata. See ROSMAP_samples_missing_metadata.txt")
}

meta_matched <- meta[match(sample_ids, meta$sample_id), , drop = FALSE]
rownames(meta_matched) <- meta_matched$sample_id
stopifnot(identical(colnames(count_matrix), rownames(meta_matched)))

# Add reproducible numeric versions of common covariates without dropping samples.
age_chr <- trimws(as.character(meta_matched$age))
meta_matched$age_numeric <- ifelse(
  age_chr == "90+",
  90,
  suppressWarnings(as.numeric(age_chr))
)
meta_matched$rin_numeric <- suppressWarnings(as.numeric(meta_matched$rin))
meta_matched$pmi_numeric <- suppressWarnings(as.numeric(meta_matched$pmi))
meta_matched$diagnosis <- suppressWarnings(as.integer(meta_matched$diagnosis))

cat("Count matrix and metadata are exactly aligned.\n")
cat("\nDiagnosis distribution among RNA-seq samples:\n")
print(table(meta_matched$diagnosis, useNA = "ifany"))

# ------------------------------------------------------------------------------
# 4. Estimate DESeq2 size factors on ALL aligned RNA-seq samples
# ------------------------------------------------------------------------------
# design = ~1 is intentional: this object is only for global normalization and
# gene-universe definition, not for testing a phenotype.
cat("\nEstimating DESeq2 size factors across the full cohort...\n")

dds_master <- DESeqDataSetFromMatrix(
  countData = count_matrix,
  colData = meta_matched,
  design = ~ 1
)
dds_master <- estimateSizeFactors(dds_master)

size_factors <- sizeFactors(dds_master)
if (anyNA(size_factors) || any(!is.finite(size_factors)) || any(size_factors <= 0)) {
  stop("Invalid DESeq2 size factors were produced.")
}

normalized_counts <- counts(dds_master, normalized = TRUE)
stopifnot(identical(dim(normalized_counts), dim(count_matrix)))
cat("Size-factor normalization: PASSED\n")

# ------------------------------------------------------------------------------
# 5. Gene-level expression QC and GLOBAL gene universe
# ------------------------------------------------------------------------------
cat("Calculating gene-level expression QC...\n")

mean_norm <- rowMeans(normalized_counts)
median_norm <- apply(normalized_counts, 1, median)
raw_ge_1 <- rowSums(count_matrix >= 1)
raw_ge_10 <- rowSums(count_matrix >= 10)
norm_gt_0 <- rowSums(normalized_counts > 0)

keep_global <- mean_norm > threshold

gene_qc <- data.frame(
  ensembl_id_version = rownames(count_matrix),
  ensembl_id = sub("\\..*$", "", rownames(count_matrix)),
  mean_normalized_count = mean_norm,
  median_normalized_count = median_norm,
  samples_raw_count_ge_1 = raw_ge_1,
  samples_raw_count_ge_10 = raw_ge_10,
  samples_normalized_count_gt_0 = norm_gt_0,
  keep_global = keep_global,
  stringsAsFactors = FALSE
)

cat("\n============================================================\n")
cat("GLOBAL EXPRESSION FILTER\n")
cat("============================================================\n")
cat("Criterion: mean normalized count > ", threshold, "\n", sep = "")
cat("Genes before filtering: ", nrow(count_matrix), "\n", sep = "")
cat("Genes retained:         ", sum(keep_global), "\n", sep = "")
cat("Genes removed:          ", sum(!keep_global), "\n", sep = "")
cat("Percent retained:       ", round(100 * mean(keep_global), 2), "%\n", sep = "")

# Alternative filters are reported for transparency/sensitivity; they do NOT
# change the primary global universe.
n_samples <- ncol(count_matrix)
filter_summary <- data.frame(
  filter = c(
    "mean normalized count > 1",
    "mean normalized count > 5",
    "mean normalized count > 10",
    "raw count >=10 in >=3 samples",
    "raw count >=10 in >=10 samples",
    "raw count >=10 in >=5% samples",
    "raw count >=10 in >=10% samples"
  ),
  genes_retained = c(
    sum(mean_norm > 1),
    sum(mean_norm > 5),
    sum(mean_norm > 10),
    sum(raw_ge_10 >= 3),
    sum(raw_ge_10 >= 10),
    sum(raw_ge_10 >= ceiling(0.05 * n_samples)),
    sum(raw_ge_10 >= ceiling(0.10 * n_samples))
  ),
  stringsAsFactors = FALSE
)
filter_summary$percent_of_raw_features <- round(
  100 * filter_summary$genes_retained / nrow(count_matrix), 2
)

cat("\nAlternative filter comparison (informational only):\n")
print(filter_summary, row.names = FALSE)

# ------------------------------------------------------------------------------
# 6. Apply the same global gene universe to raw and normalized matrices
# ------------------------------------------------------------------------------
raw_counts_filtered <- count_matrix[keep_global, , drop = FALSE]
normalized_counts_filtered <- normalized_counts[keep_global, , drop = FALSE]

stopifnot(identical(rownames(raw_counts_filtered), rownames(normalized_counts_filtered)))
stopifnot(identical(colnames(raw_counts_filtered), colnames(normalized_counts_filtered)))

# ------------------------------------------------------------------------------
# 7. Export reusable master objects
# ------------------------------------------------------------------------------
write.csv(
  meta_matched,
  file.path(outdir, "ROSMAP_RNAseq_master_metadata_all_samples.csv"),
  row.names = FALSE
)

write.csv(
  gene_qc,
  file.path(outdir, "ROSMAP_gene_expression_QC_all_features.csv"),
  row.names = FALSE
)

write.csv(
  filter_summary,
  file.path(outdir, "ROSMAP_expression_filter_sensitivity.csv"),
  row.names = FALSE
)

write.table(
  gene_qc[keep_global, c("ensembl_id_version", "ensembl_id", "mean_normalized_count")],
  file.path(outdir, "ROSMAP_global_gene_universe.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE
)

saveRDS(
  raw_counts_filtered,
  file.path(outdir, "ROSMAP_raw_counts_global_gene_universe.rds")
)

saveRDS(
  normalized_counts_filtered,
  file.path(outdir, "ROSMAP_normalized_counts_global_gene_universe.rds")
)

saveRDS(
  dds_master,
  file.path(outdir, "ROSMAP_master_dds_all_samples.rds")
)

write.csv(
  data.frame(sample_id = names(size_factors), size_factor = as.numeric(size_factors)),
  file.path(outdir, "ROSMAP_DESeq2_size_factors_all_samples.csv"),
  row.names = FALSE
)

# AREA rankable-value file: samples x genes, with sample_id first.
# These are DESeq2-normalized counts, NOT VST/rlog values.
area_matrix <- t(normalized_counts_filtered)
area_df <- data.frame(sample_id = rownames(area_matrix), area_matrix, check.names = FALSE)
write.csv(
  area_df,
  file.path(outdir, "ROSMAP_AREA_normalized_counts_all_samples.csv"),
  row.names = FALSE,
  quote = FALSE
)

# ------------------------------------------------------------------------------
# 8. Reproducibility manifest / final validation
# ------------------------------------------------------------------------------
stopifnot(identical(colnames(raw_counts_filtered), meta_matched$sample_id))
stopifnot(identical(rownames(area_matrix), meta_matched$sample_id))

manifest <- data.frame(
  item = c(
    "raw_features", "rna_seq_samples", "matched_samples",
    "global_mean_normalized_threshold", "retained_features"
  ),
  value = c(
    nrow(count_matrix), ncol(count_matrix), nrow(meta_matched),
    threshold, nrow(raw_counts_filtered)
  ),
  stringsAsFactors = FALSE
)
write.csv(manifest, file.path(outdir, "ROSMAP_preprocessing_manifest.csv"), row.names = FALSE)

sink(file.path(outdir, "ROSMAP_preprocessing_sessionInfo.txt"))
cat("ROSMAP master preprocessing\n===========================\n\n")
print(manifest)
cat("\n")
print(sessionInfo())
sink()

cat("\n============================================================\n")
cat("PREPROCESSING COMPLETE\n")
cat("============================================================\n")
cat("Samples retained: ", ncol(raw_counts_filtered), "\n", sep = "")
cat("Genes retained:   ", nrow(raw_counts_filtered), "\n\n", sep = "")
cat("AREA input:\n  ", file.path(outdir, "ROSMAP_AREA_normalized_counts_all_samples.csv"), "\n", sep = "")
cat("DESeq2 raw-count master:\n  ", file.path(outdir, "ROSMAP_raw_counts_global_gene_universe.rds"), "\n", sep = "")
cat("Master metadata:\n  ", file.path(outdir, "ROSMAP_RNAseq_master_metadata_all_samples.csv"), "\n", sep = "")
cat("Gene universe:\n  ", file.path(outdir, "ROSMAP_global_gene_universe.tsv"), "\n", sep = "")
