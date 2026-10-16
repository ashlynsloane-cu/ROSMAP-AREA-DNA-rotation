# Stage 11 Conventional Phenotype-Only Batch Runner

Replays 6 DLPFC RNA DESeq2 and 6 TMT limma phenotype-only comparisons. Reads original research scripts and frozen reference output files; all new outputs remain in Downloads (or another explicitly specified external output root).

Requires Rscript, DESeq2, limma, Python, pandas, NumPy. The historical DESeq2 counts file path is taken from each `run_summary.csv`; a matching file must exist locally, and any original column map must be available. The preflight will stop rather than substitute another count matrix without provenance. The limma matrix comes from the Stage 11 task manifest. Phenotype-only models pass empty continuous/categorical lists explicitly.

First run preflight with no `--execute`, then pilot `--max-jobs 2 --execute --stop-on-failure`. The order is NCI-versus-AD DLPFC then TMT. Results may differ across versions of R/Bioconductor, which is surfaced as a validation failure for further analysis, not silently accepted.

Use a new output root for each independent experiment; `--resume` revalidates completed files and refuses to overwrite a failed existing job.
