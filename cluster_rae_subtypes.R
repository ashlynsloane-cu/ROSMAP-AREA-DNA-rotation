#!/usr/bin/env Rscript

# cluster_rae_subtypes.R
# This script performs k-means clustering on the binary Patient-by-Gene RAE risk matrix.
# It automatically selects the optimal number of clusters (k) using the Elbow and Silhouette methods
# if no manual override is provided, plots the diagnostic metrics, and generates a heatmap.
# It uses native R argument parsing (commandArgs) to completely avoid 'argparse' package dependencies.

# Standard library check for cluster silhouette calculations
if (!requireNamespace("cluster", quietly = TRUE)) {
  install.packages("cluster", repos = "https://cloud.r-project.org")
}
library(cluster)

# Native argument parsing (no argparse package needed)
args <- commandArgs(trailingOnly = TRUE)

matrix_path <- NULL
clinical_path <- NULL
out_dir <- NULL
manual_k <- NULL

i <- 1
while (i <= length(args)) {
  if (args[i] %in% c("-m", "--matrix")) {
    matrix_path <- args[i + 1]
    i <- i + 2
  } else if (args[i] %in% c("-c", "--clinical")) {
    clinical_path <- args[i + 1]
    i <- i + 2
  } else if (args[i] %in% c("-o", "--outdir")) {
    out_dir <- args[i + 1]
    i <- i + 2
  } else if (args[i] %in% c("-k", "--clusters")) {
    manual_k <- as.integer(args[i + 1])
    i <- i + 2
  } else {
    i <- i + 1
  }
}

# Display usage helper if missing required args
if (is.null(matrix_path) || is.null(clinical_path) || is.null(out_dir)) {
  cat("\nUsage: Rscript cluster_rae_subtypes.R -m [matrix.csv] -c [clinical.csv] -o [output_dir] [-k clusters]\n\n")
  cat("Options:\n")
  cat("  -m, --matrix    : Path to binary RAE matrix CSV (from prepare_rae_clustering.py)\n")
  cat("  -c, --clinical  : Path to original clinical metadata CSV\n")
  cat("  -o, --outdir    : Output directory for diagnostics, assignments, and heatmaps\n")
  cat("  -k, --clusters  : Optional manual override for cluster count (e.g., -k 5)\n\n")
  stop("Error: Missing required arguments (-m, -c, -o).")
}

# Create output directory
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat("--------------------------------------------------\n")
cat("Starting RAE Patient Clustering Pipeline\n")
cat("--------------------------------------------------\n")

cat("Loading RAE matrix from:", matrix_path, "\n")
rae_mat <- read.csv(matrix_path, row.names = 1, check.names = FALSE)
cat("  -> Dimensions:", nrow(rae_mat), "samples x", ncol(rae_mat), "genes\n")

cat("Loading clinical metadata from:", clinical_path, "\n")
meta_df <- read.csv(clinical_path, row.names = 1, check.names = FALSE)
cat("  -> Dimensions:", nrow(meta_df), "samples x", ncol(meta_df), "variables\n")

# Align samples
common_samples <- intersect(rownames(rae_mat), rownames(meta_df))
cat("  -> Number of aligned samples:", length(common_samples), "\n")

if (length(common_samples) == 0) {
  stop("Error: No overlapping samples found between RAE matrix and clinical metadata!")
}

rae_mat <- rae_mat[common_samples, , drop = FALSE]
meta_df <- meta_df[common_samples, , drop = FALSE]

# Drop any genes that have zero variance (all 0s or all 1s after alignment)
gene_vars <- apply(rae_mat, 2, var, na.rm = TRUE)
genes_to_keep <- names(gene_vars[!is.na(gene_vars) & gene_vars > 0])
rae_mat <- rae_mat[, genes_to_keep, drop = FALSE]
cat("  -> Number of genes with expression variance:", ncol(rae_mat), "\n")

if (ncol(rae_mat) < 2) {
  stop("Error: Less than 2 genes with variance found. K-means clustering cannot be performed!")
}

# Replace NaNs with 0 (no risk) for K-means math
rae_kmeans_input <- rae_mat
rae_kmeans_input[is.na(rae_kmeans_input)] <- 0

# Calculate Elbow (WSS) and Silhouette Diagnostics for k = 2 to 10
k_values <- 2:10
wss <- numeric(length(k_values))
sil_widths <- numeric(length(k_values))

dis <- dist(rae_kmeans_input, method = "euclidean")

for (i in seq_along(k_values)) {
  k <- k_values[i]
  set.seed(42) # For reproducibility
  km <- kmeans(rae_kmeans_input, centers = k, nstart = 25)
  wss[i] <- km$tot.withinss
  
  # Calculate silhouette width
  sil <- silhouette(km$cluster, dis)
  sil_widths[i] <- mean(sil[, "sil_width"])
}

# Determine optimal k based on maximum Average Silhouette Width
optimal_k_idx <- which.max(sil_widths)
auto_k <- k_values[optimal_k_idx]

# Select final k (automatic or manual override)
final_k <- auto_k
if (!is.null(manual_k)) {
  final_k <- manual_k
}

# Save diagnostic plots
diag_plot_path <- file.path(out_dir, "clustering_diagnostics.png")
png(diag_plot_path, width = 1200, height = 600, res = 150)
par(mfrow = c(1, 2), mar = c(5, 4, 4, 2) + 0.1)

# Plot 1: Elbow Method (WSS)
plot(k_values, wss, type = "b", pch = 19, col = "#2c3e50", lwd = 2,
     xlab = "Number of Clusters (k)", ylab = "Total Within-Cluster SS (WSS)",
     main = "Elbow Method")
abline(v = final_k, col = "#e74c3c", lty = 2, lwd = 2)

# Plot 2: Average Silhouette Width
plot(k_values, sil_widths, type = "b", pch = 19, col = "#2c3e50", lwd = 2,
     xlab = "Number of Clusters (k)", ylab = "Average Silhouette Width",
     main = "Silhouette Profile")
abline(v = final_k, col = "#e74c3c", lty = 2, lwd = 2)

dev.off()

# Perform Final K-Means Clustering
cat("Performing final K-means clustering with k =", final_k, "...\n")
set.seed(42)
final_km <- kmeans(rae_kmeans_input, centers = final_k, nstart = 25)

# Add cluster assignments to metadata
meta_df$RAE_Cluster <- as.factor(final_km$cluster)

# Save cluster assignments
cluster_assignments_path <- file.path(out_dir, "patient_rae_clusters.csv")
write.csv(data.frame(Sample_ID = rownames(meta_df), Cluster = final_km$cluster), 
          cluster_assignments_path, row.names = FALSE)
cat("  -> Saved final cluster assignments to:", cluster_assignments_path, "\n")

# Generate Heatmap
if (suppressMessages(require("pheatmap", quietly = TRUE))) {
  cat("Generating heatmap using pheatmap...\n")
  
  ann_df <- data.frame(row.names = common_samples)
  ann_df$Cluster <- meta_df$RAE_Cluster
  
  diag_col <- grep("diag", colnames(meta_df), ignore.case = TRUE, value = TRUE)
  if (length(diag_col) > 0) {
    ann_df$Diagnosis <- as.factor(meta_df[[diag_col[1]]])
  }
  
  braak_col <- grep("braak", colnames(meta_df), ignore.case = TRUE, value = TRUE)
  if (length(braak_col) > 0) {
    ann_df$Braak_Stage <- as.factor(meta_df[[braak_col[1]]])
  }
  
  cerad_col <- grep("cerad", colnames(meta_df), ignore.case = TRUE, value = TRUE)
  if (length(cerad_col) > 0) {
    ann_df$CERAD_Score <- as.factor(meta_df[[cerad_col[1]]])
  }
  
  color_palette <- colorRampPalette(c("#ffffff", "#e74c3c"))(50)
  
  sorted_indices <- order(meta_df$RAE_Cluster)
  sorted_rae_mat <- rae_mat[sorted_indices, ]
  sorted_ann_df <- ann_df[sorted_indices, , drop = FALSE]
  
  heatmap_plot_path <- file.path(out_dir, "rae_subtype_heatmap.png")
  
  png(heatmap_plot_path, width = 1200, height = 1000, res = 150)
  pheatmap(
    t(sorted_rae_mat),
    cluster_cols = FALSE, # Pre-sorted by cluster assignment
    cluster_rows = TRUE,  # Cluster genes hierarchically
    annotation_col = sorted_ann_df,
    color = color_palette,
    show_colnames = FALSE,
    legend = FALSE,
    main = paste0("Molecular Subtypes by Cumulative RAE (k = ", final_k, ")")
  )
  dev.off()
  cat("  -> Heatmap successfully saved to:", heatmap_plot_path, "\n")
  
} else {
  cat("\npheatmap library is not installed. Saving pre-sorted matrix for custom plotting...\n")
  sorted_indices <- order(final_km$cluster)
  clustered_matrix_path <- file.path(out_dir, "clustered_rae_matrix.csv")
  write.csv(rae_mat[sorted_indices, ], clustered_matrix_path, row.names = TRUE)
  cat("  -> Pre-sorted matrix saved to:", clustered_matrix_path, "\n")
}

cat("--------------------------------------------------\n")
cat("RAE Subtyping Pipeline Completed Successfully!\n")
cat("--------------------------------------------------\n")
