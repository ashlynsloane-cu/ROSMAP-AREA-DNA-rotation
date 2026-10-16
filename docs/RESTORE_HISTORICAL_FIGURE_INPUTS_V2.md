# Restore historical figure inputs

The Figure 6 formal comparison and Figure 8 protein evidence are also
listed in `finalization_audit/repository_freeze/additional_figure_inputs.json`.
The recovery tool finds these in the original research or frozen workspace,
checks SHA-256 against the manifest, and links only into Git-ignored destinations.
Nothing is staged or copied to Git. It will fail if the source is absent,
its hash differs, or the destination is already occupied incorrectly.

This enables **historical figure replay**, not newly regenerated Stage 7-12.
Run with `--research-root`, `--frozen-root`, `--execute`.
