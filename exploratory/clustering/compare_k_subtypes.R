#!/usr/bin/env Rscript

# compare_k_subtypes.R
# This script performs K-means clustering on the binary Patient-by-Gene RAE risk matrix
# for both k = 3 and k = 4. It aligns clusters with clinical metadata, stratifies patients 
# by cognitive status (NCI, MCI, AD), and analyzes how subtyping shifts when increasing k.
# Native argument parsing (no argparse package needed) for zero-dependency execution.

# Load recommended base packages
if (!requireNamespace("cluster", quietly = TRUE)) {
  install.packages("cluster", repos = "https://cloud.r-project.org")
}
library(cluster)

# Native argument parsing
args <- commandArgs(trailingOnly = TRUE)

matrix_path <- NULL
clinical_path <- NULL
out_dir <- NULL
summary_path <- NULL
p_cutoff <- 0.01
nes_cutoff <- 3.0

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
  } else if (args[i] %in% c("-s", "--summary")) {
    summary_path <- args[i + 1]
    i <- i + 2
  } else if (args[i] %in% c("-p", "--p-cutoff")) {
    p_cutoff <- as.numeric(args[i + 1])
    i <- i + 2
  } else if (args[i] %in% c("-n", "--nes-cutoff")) {
    nes_cutoff <- as.numeric(args[i + 1])
    i <- i + 2
  } else {
    i <- i + 1
  }
}

# Display usage helper if missing required args
if (is.null(matrix_path) || is.null(clinical_path) || is.null(out_dir)) {
  cat("\nUsage: Rscript compare_k_subtypes.R -m [matrix.csv] -c [clinical.csv] -o [output_dir] [-s summary.csv] [-p p_cutoff] [-n nes_cutoff]\n\n")
  cat("Options:\n")
  cat("  -m, --matrix    : Path to binary RAE matrix CSV\n")
  cat("  -c, --clinical  : Path to original clinical metadata CSV\n")
  cat("  -o, --outdir    : Output directory for report and heatmaps\n")
  cat("  -s, --summary   : Optional path to rae_thresholds_summary.csv for dynamic filtering\n")
  cat("  -p, --p-cutoff  : BH-adjusted p-value cutoff for filtering (default: 0.01)\n")
  cat("  -n, --nes-cutoff: NES magnitude cutoff for filtering (default: 3.0)\n\n")
  stop("Error: Missing required arguments (-m, -c, -o).")
}

# Create output directory
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat("--------------------------------------------------\n")
cat("Starting RAE Subtype Stratification and Comparison\n")
cat("--------------------------------------------------\n")

# 1. Load data
cat("Loading RAE matrix from:", matrix_path, "\n")
rae_mat <- read.csv(matrix_path, row.names = 1, check.names = FALSE)
cat("  -> Dimensions:", nrow(rae_mat), "samples x", ncol(rae_mat), "genes\n")

cat("Loading clinical metadata from:", clinical_path, "\n")
meta_df <- read.csv(clinical_path, row.names = 1, check.names = FALSE)
cat("  -> Dimensions:", nrow(meta_df), "samples x", ncol(meta_df), "variables\n")

# 2. Align samples
common_samples <- intersect(rownames(rae_mat), rownames(meta_df))
cat("  -> Number of aligned samples:", length(common_samples), "\n")

if (length(common_samples) == 0) {
  stop("Error: No overlapping samples found between RAE matrix and clinical metadata!")
}

rae_mat <- rae_mat[common_samples, , drop = FALSE]
meta_df <- meta_df[common_samples, , drop = FALSE]

# 3. Dynamic Filtering (Significance and Effect Size)
if (!is.null(summary_path) && file.exists(summary_path)) {
  cat("\nApplying statistical and effect-size filtering using summary file...\n")
  summary_df <- read.csv(summary_path, check.names = FALSE)
  
  # Try to extract attribute name from file name
  filename <- basename(matrix_path)
  attr_clean <- gsub("rae_matrix_", "", filename)
  attr_clean <- gsub(".csv", "", attr_clean)
  
  # Match attribute in summary file (case-insensitive)
  matched_attrs <- unique(summary_df$Attribute)
  attr_match <- matched_attrs[tolower(gsub(" ", "_", matched_attrs)) == tolower(attr_clean)]
  
  if (length(attr_match) > 0) {
    cat("  -> Matched attribute in summary file:", attr_match[1], "\n")
    sub_summary <- summary_df[summary_df$Attribute == attr_match[1], ]
    
    # Filter by significance and effect size
    sig_genes <- sub_summary$Gene[sub_summary$p_value_BH < p_cutoff & abs(sub_summary$NES) >= nes_cutoff]
    sig_genes_aligned <- intersect(sig_genes, colnames(rae_mat))
    
    cat(sprintf("  -> Applied p_BH < %s and |NES| >= %.1f filtering:\n", as.character(p_cutoff), nes_cutoff))
    cat(sprintf("     Kept %d of %d original genes in matrix.\n", length(sig_genes_aligned), ncol(rae_mat)))
    
    if (length(sig_genes_aligned) >= 2) {
      rae_mat <- rae_mat[, sig_genes_aligned, drop = FALSE]
    } else {
      cat("  - Warning: Too few genes passed the filter. Proceeding with variance-filtered genes.\")\n")
    }
  } else {
    cat("  - Warning: Could not match attribute", attr_clean, "in summary file. Proceeding with all variance-filtered genes.\n")
  }
}

# Drop any genes that have zero variance
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

# 4. Map Detailed Cognitive Status (including MCI!)
cat("\nStratifying patients by clinical cognitive diagnosis...\n")
diag_col <- grep("diag", colnames(meta_df), ignore.case = TRUE, value = TRUE)

if (length(diag_col) > 0) {
  # ROSMAP diagnosis scheme:
  # 1 = NCI (No Cognitive Impairment)
  # 2, 3 = MCI (Mild Cognitive Impairment)
  # 4, 5 = AD (Alzheimer's Disease)
  # 6 = Other Dementia
  meta_df$Cognitive_Status <- "Unknown"
  raw_diags <- meta_df[[diag_col[1]]]
  
  meta_df$Cognitive_Status[raw_diags == 1] <- "NCI (Healthy)"
  meta_df$Cognitive_Status[raw_diags %in% c(2, 3)] <- "MCI"
  meta_df$Cognitive_Status[raw_diags %in% c(4, 5)] <- "AD"
  meta_df$Cognitive_Status[raw_diags == 6] <- "Other Dementia"
  meta_df$Cognitive_Status[is.na(raw_diags)] <- "Unknown"
  
  cat("  -> Stratification counts:\n")
  print(table(meta_df$Cognitive_Status))
} else {
  cat("  - Warning: 'diagnosis' column not found. Defaulting to general binary mappings.\n")
}

# 5. Perform Clustering for k = 3 and k = 4
set.seed(42) # Reproducibility
cat("\nPerforming K-means clustering for k = 3...\n")
km3 <- kmeans(rae_kmeans_input, centers = 3, nstart = 25)

cat("Performing K-means clustering for k = 4...\n")
km4 <- kmeans(rae_kmeans_input, centers = 4, nstart = 25)

meta_df$Cluster_k3 <- as.factor(km3$cluster)
meta_df$Cluster_k4 <- as.factor(km4$cluster)

# Save cluster assignments
assignments_path <- file.path(out_dir, "patient_k3_k4_assignments.csv")
write.csv(data.frame(
  Sample_ID = rownames(meta_df),
  Cluster_k3 = km3$cluster,
  Cluster_k4 = km4$cluster,
  Cognitive_Status = if("Cognitive_Status" %in% colnames(meta_df)) meta_df$Cognitive_Status else "N/A"
), assignments_path, row.names = FALSE)
cat("  -> Saved cluster assignments to:", assignments_path, "\n")


# 6. Generate the Comparative Subtyping Report
report_path <- file.path(out_dir, "k3_vs_k4_stratification_report.txt")
sink(report_path)

cat("======================================================================\n")
cat("RAE MOLECULAR SUBTYPING COMPARISON: k = 3 VS k = 4\n")
cat("======================================================================\n")
cat("Matrix processed: ", matrix_path, "\n")
cat("Number of patients: ", nrow(rae_mat), "\n")
cat("Number of high-confidence genes used: ", ncol(rae_mat), "\n")
cat("======================================================================\n\n")

if ("Cognitive_Status" %in% colnames(meta_df)) {
  # Cognitive Status vs Clusters for k = 3
  cat("----------------------------------------------------------------------\n")
  cat("COGNITIVE STATUS DISTRIBUTION IN k = 3 SUBTYPES\n")
  cat("----------------------------------------------------------------------\n")
  tbl3 <- table(meta_df$Cluster_k3, meta_df$Cognitive_Status)
  print(tbl3)
  cat("\nProportions by Cluster (Rows):\n")
  print(round(prop.table(tbl3, 1) * 100, 2))
  
  # Chi-Square Test
  chisq3 <- chisq.test(tbl3)
  cat(sprintf("\nChi-Square Test of Independence (Cluster vs Cognitive Status):\n"))
  cat(sprintf("  -> p-value: %s\n\n", format.pval(chisq3$p.value)))
  
  # Cognitive Status vs Clusters for k = 4
  cat("----------------------------------------------------------------------\n")
  cat("COGNITIVE STATUS DISTRIBUTION IN k = 4 SUBTYPES\n")
  cat("----------------------------------------------------------------------\n")
  tbl4 <- table(meta_df$Cluster_k4, meta_df$Cognitive_Status)
  print(tbl4)
  cat("\nProportions by Cluster (Rows):\n")
  print(round(prop.table(tbl4, 1) * 100, 2))
  
  # Chi-Square Test
  chisq4 <- chisq.test(tbl4)
  cat(sprintf("\nChi-Square Test of Independence (Cluster vs Cognitive Status):\n"))
  cat(sprintf("  -> p-value: %s\n\n", format.pval(chisq4$p.value)))
}

# Cluster Transition / Split Analysis
cat("----------------------------------------------------------------------\n")
cat("TRANSITION MATRIX: How k = 3 Clusters Split into k = 4\n")
cat("----------------------------------------------------------------------\n")
transition_tbl <- table(meta_df$Cluster_k3, meta_df$Cluster_k4)
rownames(transition_tbl) <- paste0("k3_Cluster_", rownames(transition_tbl))
colnames(transition_tbl) <- paste0("k4_Cluster_", colnames(transition_tbl))
print(transition_tbl)

cat("\nTransition Flow (Percentage of k3 Cluster that flows to k4 Cluster):\n")
print(round(prop.table(table(meta_df$Cluster_k3, meta_df$Cluster_k4), 1) * 100, 2))

cat("\n======================================================================\n")
cat("BIOLOGICAL ANALYSIS GUIDANCE:\n")
cat("  1. Look for 'Transitional' Clusters:\n")
cat("     In k = 3, see if MCI patients enrich in a single 'intermediate' cluster,\n")
cat("     marking a molecular bridge between Healthy/NCI and Severe AD.\n")
cat("  2. Observe the Split in k = 4:\n")
cat("     Analyze the Transition Matrix above. When moving from k=3 to k=4, does\n")
cat("     the high-burden AD cluster split into two distinct molecular pathways,\n")
cat("     or does the intermediate/MCI cluster subdivide?\n")
cat("======================================================================\n")

sink()
cat("  -> Comparative report saved to:", report_path, "\n")


# 7. Generate heatmaps if pheatmap is installed
if (suppressMessages(require("pheatmap", quietly = TRUE))) {
  cat("\nGenerating side-by-side heatmaps using pheatmap...\n")
  
  ann_df <- data.frame(row.names = common_samples)
  if ("Cognitive_Status" %in% colnames(meta_df)) {
    ann_df$Cognitive_Status <- as.factor(meta_df$Cognitive_Status)
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
  
  # Heatmap for k = 3
  ann_df_k3 <- ann_df
  ann_df_k3$Cluster_k3 <- meta_df$Cluster_k3
  sorted_indices_k3 <- order(meta_df$Cluster_k3)
  
  plot_k3_path <- file.path(out_dir, "rae_heatmap_k3.png")
  png(plot_k3_path, width = 1200, height = 1000, res = 150)
  pheatmap(
    t(rae_mat[sorted_indices_k3, ]),
    cluster_cols = FALSE,
    cluster_rows = TRUE,
    annotation_col = ann_df_k3[sorted_indices_k3, , drop = FALSE],
    color = color_palette,
    show_colnames = FALSE,
    legend = FALSE,
    main = paste0("Molecular Subtypes (k = 3) showing Clinical Stratification")
  )
  dev.off()
  
  # Heatmap for k = 4
  ann_df_k4 <- ann_df
  ann_df_k4$Cluster_k4 <- meta_df$Cluster_k4
  sorted_indices_k4 <- order(meta_df$Cluster_k4)
  
  plot_k4_path <- file.path(out_dir, "rae_heatmap_k4.png")
  png(plot_k4_path, width = 1200, height = 1000, res = 150)
  pheatmap(
    t(rae_mat[sorted_indices_k4, ]),
    cluster_cols = FALSE,
    cluster_rows = TRUE,
    annotation_col = ann_df_k4[sorted_indices_k4, , drop = FALSE],
    color = color_palette,
    show_colnames = FALSE,
    legend = FALSE,
    main = paste0("Molecular Subtypes (k = 4) showing Clinical Stratification")
  )
  dev.off()
  
  cat("  -> Heatmaps successfully saved to output directory.\n")
} else {
  cat("\npheatmap library not detected. Cluster comparisons saved in assignments CSV.\n")
}

cat("--------------------------------------------------\n")
cat("K-Comparison Pipeline Completed Successfully!\n")
cat("--------------------------------------------------\n")
