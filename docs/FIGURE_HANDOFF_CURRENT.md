# Figure-ready handoff: provenance-preserving bridge

The script `workflows/prepare_figure_handoff.py` inventories **fresh model result directories** and optionally invokes the repository's existing, hash-checked **historical** Figure 1–8 replay. The two evidence classes are kept separate. It does not assert that Figures 1–8 were regenerated from the 30 new models.

Run `python3 workflows/prepare_figure_handoff.py --repo-root "$PWD" --pipeline-output "$HOME/Developer/ROSMAP-authorized-data/pipeline-output" --handoff-root "$HOME/Downloads/rosmap-final-handoff"` to generate a provenance summary. After reviewing that summary, optionally include `--frozen-root "$HOME/Developer/stage11_offline_2026-09-28" --replay-figures` to request all eight archived-input figures, provided the historical figure prerequisites and runner are installed.

The script deliberately checks only *existence* of model CSVs, not numerical concordance or statistical correctness; it also cannot prove that model CSVs map to the required Stage 7–12 contracts. Inspect `HANDOFF_PROVENANCE.json` and the independent pipeline report. Do not publish donor-level data or the reference GMTs.

Remaining genuine raw-to-figure work: adapt Stage 7/9/10/11/12 producers to the fresh output directory, verify the figure-input contracts, and perform a clean-input end-to-end run. Starting at normalized RNA/monocyte counts and processed TMT abundance is not FASTQ or raw MS preprocessing.
