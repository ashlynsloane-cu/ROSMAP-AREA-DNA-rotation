# Monocyte input validation for authorized ROSMAP analysis

The two transferred TSVs are **historical model inputs** (DESeq2 raw gene counts and normalized AREA counts), not original sequencing FASTQs. Do not commit them or donor-level metadata to the repository.

Use `python3 workflows/validate_monocyte_inputs.py --raw-counts ... --normalized-counts ... --metadata ... --output ...` to stream-check the full matrices. The report contains file hashes, sizes, dimensions, and aggregate validation outcomes but not sample identifiers.

A passing matrix-integrity check establishes feature ordering, 554 matching sample columns, unique Geneid rows, finite nonnegative normalized data, and nonnegative integer raw counts. The metadata mapping gate requires exact alignment with a metadata specimen field; if ROSMAP sample IDs require a mapping table, the tool **does not guess the mapping** and exits nonzero with `NEEDS_MAPPING_VERIFICATION`. The package does not yet regenerate counts from original sequencing files or connect all eight figures. Recovered model-input matrices must never be described as raw-to-figure validation.
