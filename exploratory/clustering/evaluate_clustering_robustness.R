#!/usr/bin/env Rscript

# evaluate_clustering_robustness.R
# This script scans your results directories for RAE matrices,
# computes clustering diagnostics (Total Within-SS and Average Silhouette Width) 
# across k = 2 to 10 for every single attribute, and compiles a master robustness table.
# It also saves a diagnostic multi-panel plot comparing all attributes.

if (!requireNamespace("cluster", quietly = TRUE)) {
  install.packages("cluster", repos = "https://cloud.r-project.org")
}
library(cluster)

# Define directories to scan
dirs_to_check <- c(
  "results/rae_analysis_progressive",
  "results/rae_analysis",
  "results/rae_analysis_strict"
)

clinical_path <- "data/ROSMAP_metadata_merged_AREA.csv"

cat("======================================================================\n")
cat("Starting Multi-Attribute Clustering Robustness Evaluation (k = 2 to 10)\n")
cat("======================================================================\n\n")

if (!file.exists(clinical_path)) {
  stop(paste("Error: Clinical metadata file not found at", clinical_path))
}

# Find all RAE matrices
matrix_files <- c()
for (d in dirs_to_check) {
  if (dir.exists(d)) {
    csvs <- list.files(d, pattern = "^rae_matrix_.*\\.csv$", full.names = TRUE)
    matrix_files <- c(matrix_files, csvs)
  }
}

if (length(matrix_files) == 0) {
  stop("Error: No RAE matrix files (rae_matrix_*.csv) found in the results directories.")
}

cat(sprintf("Found %d RAE matrices to evaluate:\n", length(matrix_files)))
for (m in matrix_files) {
  cat(sprintf("  - %s\n", m))
}
cat("\n")

# Load clinical metadata once
meta_df <- read.csv(clinical_path, row.names = 1, check.names = FALSE)

k_values <- 2:10
results_list <- list()

# Loop through each matrix
for (m_path in matrix_files) {
  attr_name <- gsub("^rae_matrix_", "", basename(m_path))
  attr_name <- gsub("\\.csv$", "", attr_name)
  
  cat(sprintf("Processing attribute: '%s'...\n", toupper(attr_name)))
  
  # Load RAE matrix
  rae_mat <- read.csv(m_path, row.names = 1, check.names = FALSE)
  
  # Align samples
  common_samples <- intersect(rownames(rae_mat), rownames(meta_df))
  if (length(common_samples) == 0) {
    cat(sprintf("  -> Warning: No overlapping samples for '%s'. Skipping.\n", attr_name))
    next
  }
  
  rae_mat <- rae_mat[common_samples, , drop = FALSE]
  
  # Filter zero variance genes
  gene_vars <- apply(rae_mat, 2, var, na.rm = TRUE)
  genes_to_keep <- names(gene_vars[!is.na(gene_vars) & gene_vars > 0])
  rae_mat <- rae_mat[, genes_to_keep, drop = FALSE]
  
  if (ncol(rae_mat) < 2) {
    cat(sprintf("  -> Warning: Less than 2 genes with variance for '%s'. Skipping.\n", attr_name))
    next
  }
  
  # Handle NAs
  rae_kmeans_input <- rae_mat
  rae_kmeans_input[is.na(rae_kmeans_input)] <- 0
  
  # Compute distance matrix once per attribute
  dis <- dist(rae_kmeans_input, method = "euclidean")
  
  # Run k = 2 to 10
  for (k in k_values) {
    set.seed(42)
    km <- kmeans(rae_kmeans_input, centers = k, nstart = 25)
    wss_val <- km$tot.withinss
    
    # Calculate silhouette
    sil <- silhouette(km$cluster, dis)
    avg_sil <- mean(sil[, "sil_width"])
    
    # Store results
    results_list[[length(results_list) + 1]] <- data.frame(
      Attribute = attr_name,
      Samples_N = length(common_samples),
      Genes_P = ncol(rae_mat),
      k = k,
      WSS = wss_val,
      Avg_Silhouette = avg_sil,
      stringsAsFactors = FALSE
    )
  }
  cat("  -> Evaluation complete.\n\n")
}

# Combine results
master_df <- do.call(rbind, results_list)

# Write master table
out_csv <- "results/clustering_robustness_metrics.csv"
write.csv(master_df, out_csv, row.names = FALSE)
cat(sprintf("Successfully saved master robustness metrics to: %s\n", out_csv))

# Print clean console summary
cat("\n======================================================================\n")
cat("MASTER CLUSTERING ROBUSTNESS SUMMARY TABLE\n")
cat("======================================================================\n")
cat(sprintf("%-25s | %-3s | %-15s | %-20s\n", "Attribute", "k", "Total WSS", "Avg Silhouette"))
cat("----------------------------------------------------------------------\n")

# Identify local optima for each attribute
unique_attrs <- unique(master_df$Attribute)
for (attr in unique_attrs) {
  attr_df <- master_df[master_df$Attribute == attr, ]
  
  # Find global max silhouette
  max_sil_idx <- which.max(attr_df$Avg_Silhouette)
  opt_k_sil <- attr_df$k[max_sil_idx]
  opt_sil_val <- attr_df$Avg_Silhouette[max_sil_idx]
  
  # Print rows
  for (i in 1:nrow(attr_df)) {
    row <- attr_df[i, ]
    is_opt <- if (row$k == opt_k_sil) " (Global Max)" else ""
    cat(sprintf("%-25s | %-3d | %-15.2f | %-20.4f%s\n", 
                row$Attribute, row$k, row$WSS, row$Avg_Silhouette, is_opt))
  }
  cat("----------------------------------------------------------------------\n")
}

# Generate master plot
pdf_out <- "results/clustering_robustness_curves.pdf"
pdf(pdf_out, width = 10, height = 5 * length(unique_attrs))
par(mfrow = c(length(unique_attrs), 2), mar = c(4, 4, 3, 1))

for (attr in unique_attrs) {
  attr_df <- master_df[master_df$Attribute == attr, ]
  
  # Plot WSS
  plot(attr_df$k, attr_df$WSS, type = "b", pch = 19, col = "#2c3e50", lwd = 2,
       xlab = "Number of Clusters (k)", ylab = "Total Within-SS (WSS)",
       main = sprintf("Elbow Curve: %s", toupper(attr)))
  grid()
  
  # Plot Silhouette
  plot(attr_df$k, attr_df$Avg_Silhouette, type = "b", pch = 19, col = "#e74c3c", lwd = 2,
       xlab = "Number of Clusters (k)", ylab = "Avg Silhouette Width",
       main = sprintf("Silhouette Profile: %s", toupper(attr)),
       ylim = c(0, max(attr_df$Avg_Silhouette) * 1.2))
  grid()
}
dev.off()

cat(sprintf("Saved multi-panel evaluation curves to: %s\n", pdf_out))
cat("======================================================================\n")
