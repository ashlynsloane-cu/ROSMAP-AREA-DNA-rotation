# ROSMAP–AREA lab handoff — verified scope and release criteria

**Verified to date from the operator's reported successful execution:**
- 12 fully adjusted brain AREA models plus 276 intercept-only/subset models: 288/288 verified.
- Phenotype-only brain conventional DESeq2/limma models: 12/12 verified.
- Stage 12 replay: 63 TSV files compared (57 byte-identical, 6 numerically equivalent).
- Stage 11 Shapley outputs: 9/9 output tables verified against archived model states.
- Monocyte model input matrix integrity and sample metadata mapping: 14,756 genes × 554 samples, PASS.
- Monocyte full AREA model execution: 6/6 finished, **strict comparison passed only 1/6**. Five models have discrepancies involving 4 distinct genes; no historical reference was changed.
- Python unit test suite: 34 passed on the operator's Mac.

**Not verified:** authorized upstream raw reads to feature counts, RNA normalization from raw counts, TMT source preprocessing, monocyte source preprocessing, manifest reconstruction independently of historical copies, fully adjusted DESeq2/limma family, all Stage 7/9/10 producers, independent Stage 12 and figures 1–8 generated from newly created upstream inputs, clean-clone execution on another machine.

**Release data policy:** Never commit or upload donor-level ROSMAP matrices, metadata, manifests containing IDs, reference model outputs, private reference datasets, Baffin-specific access credentials, or local paths into the public repository. Review any committed samples and logs for indirect identifiers. Use authorization-controlled external inputs and a documented access process.

**Definitions:** the `run_main_figures.py` runner verifies frozen hash-locked inputs. A frozen replay is not an end-to-end reproduction. The `run_raw_to_figures.py` skeleton currently contains unconfigured producers and properly reports `RAW_TO_FIGURE_COMPLETE: False`.

## Last-day operator checklist

1. `python3 -m unittest discover -s tests -p 'test_*.py' -q` — full tests pass.
2. `python3 workflows/check_handoff_release.py --release-root "$PWD" --output "$HOME/Downloads/rosmap-release-safety.json"` — inspect all suspicious tracked/staged/untracked paths; fix before staging.
3. `git diff --check` — no whitespace/errors in tracked modifications.
4. `git status --short` and `git diff --cached --name-status` — inspect exactly what is in the Git index.
5. Stage **explicitly reviewed source and documentation paths** (never `git add .` until every path is audited).
6. Re-run the checker **after staging** and again before pushing. An `OK` test suite alone cannot validate unrestricted publication.
7. Record Git commit SHA, environment versions, data access location (without credentials), test logs, successful model reports, and remaining blockers in the internal lab handoff record.
8. Do not represent the repository as raw-to-figure complete until a clean clone with authorized upstream inputs passes all producer and eight-figure checks.

## High priority engineering follow-up

Port genuine RNA/TMT/monocyte preprocessing producers and clinical manifest construction to accept external data roots; remove hardcoded `/home/assl5508/...` dependency; make existing runners write/consume fresh model outputs in an isolated workspace; replace frozen-reference gating in final figure path with provenance-checked fresh outputs; perform clean-clone acceptance tests.
