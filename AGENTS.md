# AGENTS.md

This file provides architectural guidance and behavioral instructions when working with code in this repository.

## 1. Project Overview & Primary Task

THAI-MOD is an NLP decision-support screening system for Thai cyberbullying content.
The primary operational task of this project is **Binary Classification: Insult (ดูหมิ่น) vs Defamation (หมิ่นประมาท)** following the Proposed Workflow (Slide 5), complete with Evasion Normalization, Target Identification, Allegation Analysis, and Rule-based Evidence Highlighting.

> **Legal Disclaimer:** This system provides preliminary risk screening for content moderators only. It does NOT constitute legal advice or formal judicial determinations.

---

## 2. Dataset & Preprocessing Contracts

- **Primary Dataset:** `datasets/processed/binary_insult_defamation_v2/`
  - `train.csv` (7,008 samples)
  - `validation.csv` (1,474 samples)
  - `test.csv` (1,470 samples)
  - Summary: `datasets/processed/binary_insult_defamation_v2/dataset_summary.json`
- **Label Mapping:**
  - `0`: **Insult** (ดูหมิ่น)
  - `1`: **Defamation** (เข้าข่ายหมิ่นประมาท)
  - Neutral / general non-toxic comments are excluded from this binary task split.
- **Preprocessing Contract:** `PREPROCESSING_VERSION = "binary_preprocess_v1"` (`src/thai_mod_api/binary_preprocessing.py`)
  - **`USE_FUZZY = False`**: Fuzzy keyword correction is deliberately turned off to prevent false positives and rewriting of legitimate vocabulary.
  - **RULE:** Do not alter the preprocessing contract, function names, normalization sequence, or dataset splits without explicit authorization.

---

## 3. Core Architecture & API Scopes

### A. Binary API (Primary Production Path)
- **Endpoint:** `POST /api/binary/insult-defamation/predict`
- **Metadata Endpoint:** `GET /api/binary/insult-defamation/info`
- **Default Model:** `models/experiments/binary_insult_defamation_v2/wangchanberta_v2`
- **Fallback Model:** `models/experiments/binary_insult_defamation_v2/complement_nb`
- **Override:** Configurable via `THAI_MOD_BINARY_MODEL_DIR` environment variable.
- **Scope Gating & Conservative Screening:**
  - **Conservative Target Identification:** Only specific targets (`mention`, `named`, `organization`, or specific deictic `reference`) qualify as `Identified`. Broad pronouns (`มึง`, `มัน`, `เขา`) or generic slur addresses without an identifiable name or entity do **NOT** establish a target and must result in `target.status = "Unknown"`.
  - **Abstention Policy:** If an input lacks an identifiable target, lacks an explicit insult or factual allegation cue, or if prediction confidence falls below `MIN_CONFIDENCE = 0.55`, the service must return `status = "abstained"` with a human-review recommendation.
- **Evidence Spans & Highlighting:**
  - The returned `evidence_spans` (target and allegation spans) are **Rule-based evidence cues** designed for moderator inspection. They are **NOT** model rationales or neural network attention weights.

### B. Legacy Toxicity API (Preserved Backward Compatibility)
- **Endpoints:**
  - `GET /api/health`
  - `GET /api/model-info`
  - `POST /api/predict` (Toxic vs Non-toxic classification)
  - `POST /api/batch-predict`
- **Service:** `src/thai_mod_api/model_service.py`
- **RULE:** Preserve all legacy endpoints and behavior. Never break existing routes or tests in `tests/test_api.py`.

---

## 4. Development Environment & Conventions

- **Python Version:** 3.11+ / 3.13 (conda env: `cedt` or virtual environment)
- **Dependencies:** `pip install -r requirements.txt` (Hugging Face Transformers, PyTorch, PyThaiNLP, Scikit-learn, FastAPI, Pydantic, Emoji)
- **Model Artifacts:** Located under `models/` (gitignored). Do not alter or overwrite completed training artifacts.
- **PhayaThaiBERT Status:** PhayaThaiBERT was paused and is not evaluated on Dataset V2; do not include it in current binary comparisons or train third-model variants without instruction.
- **Dataset Constraints:** The test set contains only **14 defamation samples**; class-level metrics for Defamation have high variance and must be reported with explicit acknowledgement of this limitation.
- **Web Frontend:** Located at `src/thai_mod_api/static/` (`index.html`, `app.js`, `styles.css`), directly integrated with `POST /api/binary/insult-defamation/predict`.

---

## 5. Verification Commands

Run unit and integration tests:
```bash
pytest tests/test_target_allegation.py tests/test_binary_preprocessing.py tests/test_binary_inference.py tests/test_binary_metrics.py tests/test_evasion_normalizer.py
```
Start development server:
```bash
uvicorn src.thai_mod_api.main:app --reload
```
