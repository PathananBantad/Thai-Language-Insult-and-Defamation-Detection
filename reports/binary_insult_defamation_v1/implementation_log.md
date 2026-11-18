# Binary Experiment v1 — Implementation Log

## Phase A — Repository and data audit

- Branch: `main`.
- Pre-existing work preserved: `requirements.txt` remains modified; the repository also contained pre-existing untracked Phase 3–6.2 outputs and Cascade files.
- Specified interpreter `D:\venvs\thai-mod-venv\Scripts\python.exe` is available (Python 3.13.15). No packages were installed.
- Source split counts matched the requested values. Phase 4.1 and Phase 5.1 queues each had 200 unique IDs; source-row identity and Train data were revalidated by the dataset preparation script.
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6.0 GB. Transformer runs are sequential.

## Phase B — Binary dataset

- Created `datasets/processed/binary_insult_defamation_v1/` with train/validation/test CSVs, summary, exclusion log, and README.
- Actual counts: Train 7,358 (6,990 Insult, 368 Defamation), Validation 1,557 (1,541/16), Test 1,556 (1,541/15).
- Exactly 200 unique Train samples excluded; Validation/Test unchanged.
- Train Defamation provenance: Synthetic 296, Real/Other 72.
- Original source hashes are recorded in `dataset_summary.json`; the preparation script compares hashes before and after generation.

## Phase C — Preprocessing and tests

- Added `binary_preprocess_v1`, with deterministic evasion rules and fuzzy correction disabled.
- Added rule-based binary target/allegation context and separate binary inference service; legacy API behavior was not changed.
- Focused tests: 11 passed, 1 skipped. The API compatibility test skipped because FastAPI is absent in the specified interpreter. Running pytest without `--noconftest` fails at the existing `tests/conftest.py` import of FastAPI; no installation was performed.
- Loaded the completed ComplementNB artifact and exercised a mention/allegation example plus out-of-scope and empty inputs; the service predicted for the scoped example and abstained for the latter inputs.

## Phases D–E — Models and evaluation

- ComplementNB: completed; selected alpha and Validation result are stored with the model artifact.
- WangchanBERTa: an initial attempt stopped before the first epoch because Transformers 4.57.6 rejected a redundant `num_labels` constructor argument. The call was fixed and the corrected run is active; epoch 1 Validation macro F1 was 0.6477. Seven of 460 optimizer updates were skipped by the FP16 scaler; 453 updates ran. No Test data was opened.
- PhayaThaiBERT: pending WangchanBERTa completion.
- Final Test evaluation has not yet been run. Test remains closed to training/model selection.

## Phases F–H

- Target and allegation rule baseline implemented, with no fabricated performance metrics.
- Binary inference API route implemented as opt-in; runtime integration remains unverified because FastAPI is unavailable.
- Final full-suite status and experiment metrics will be added after model runs.
