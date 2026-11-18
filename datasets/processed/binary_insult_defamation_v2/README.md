# Binary Insult vs Defamation Dataset binary_insult_defamation_v2

This derived dataset maps original label 1 (Insult) to binary label 0 and original label 2 (Defamation) to binary label 1. Original Normal (0) rows are not included. The original split membership is preserved.

The 200 unique Phase 4.1/5.1 pending-review Insult rows are excluded from Train only. They are not relabeled; the original datasets and review queues remain unchanged. See `exclusion_log.csv` for every stable sample ID, source row, reason, review IDs and queue references.

All rows whose source is HateThaiSent are excluded from every split. Their lineage is recorded in `exclusion_log.csv`.

`text_raw` preserves the original message. `source_text_norm` preserves the pre-existing normalization. `text_norm` is regenerated with `binary_preprocess_v1` and must be used by training and inference for this experiment. The `source`, original labels and synthetic flag retain provenance.

Validation and Test are not excluded or resampled. Train Defamation includes synthetic examples from the pre-existing processed split; no synthetic examples were created here. This is a preliminary risk-screening dataset, not a legal determination.
