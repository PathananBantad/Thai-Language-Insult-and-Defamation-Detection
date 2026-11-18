#!/usr/bin/env python
"""Fit a ComplementNB binary Insult/Defamation candidate using Validation selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thai_mod_api.binary_metrics import calculate_binary_metrics
from src.thai_mod_api.binary_preprocessing import PREPROCESSING_VERSION
from src.thai_mod_api.text_processing import tokenize
from src.thai_mod_api.binary_training import _load_binary_split

DATA_DIR = ROOT / "datasets" / "processed" / "binary_insult_defamation_v1"
OUTPUT_DIR = ROOT / "models" / "experiments" / "binary_insult_defamation_v1" / "complement_nb"
ALPHAS = (0.1, 0.3, 0.5, 1.0)


def make_model(alpha: float) -> Pipeline:
    return Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), min_df=2, max_features=50000, lowercase=False)),
        ("tfidf", TfidfTransformer()),
        ("clf", ComplementNB(alpha=alpha)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite model output: {args.output_dir}")
    train = _load_binary_split(args.data_dir / "train.csv", "train")
    validation = _load_binary_split(args.data_dir / "validation.csv", "validation")
    if set(train.label.unique()) != {0, 1} or set(validation.label.unique()) != {0, 1}:
        raise ValueError("Train and Validation must each contain both binary classes")
    sample_weights = compute_sample_weight(class_weight="balanced", y=train.label.to_numpy())
    trials = []
    best = None
    for alpha in ALPHAS:
        model = make_model(alpha)
        model.fit(train.text_norm.tolist(), train.label.tolist(), clf__sample_weight=sample_weights)
        pred = model.predict(validation.text_norm.tolist()).tolist()
        probs = model.predict_proba(validation.text_norm.tolist()).tolist()
        metrics = calculate_binary_metrics(validation.label.astype(int).tolist(), pred, probs, validation.source.astype(str).tolist())
        trials.append({"alpha": alpha, "validation": metrics})
        if best is None or metrics["macro_f1"] > best["metrics"]["macro_f1"]:
            best = {"alpha": alpha, "model": model, "metrics": metrics}
    assert best is not None
    args.output_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(best["model"], args.output_dir / "model.joblib")
    metadata = {
        "model_type": "complement_nb",
        "model_name": "PyThaiNLP CountVectorizer + TF-IDF + ComplementNB",
        "model_version": "scikit-learn ComplementNB binary experiment v1",
        "label_mapping": {"0": "Insult", "1": "Defamation"},
        "preprocessing_version": PREPROCESSING_VERSION,
        "input_column": "text_norm",
        "tokenizer": "PyThaiNLP word_tokenize(engine='newmm', keep_whitespace=False)",
        "alpha_grid": list(ALPHAS),
        "selected_alpha": best["alpha"],
        "selection_metric": "validation macro_f1",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_label_counts": {str(int(k)): int(v) for k, v in train.label.value_counts().items()},
        "validation_label_counts": {str(int(k)): int(v) for k, v in validation.label.value_counts().items()},
        "train_class_weighting": "balanced sample weights",
        "synthetic_defamation_train": int(((train.label == 1) & (train.source.str.casefold() == "synthetic")).sum()),
        "real_or_other_defamation_train": int(((train.label == 1) & (train.source.str.casefold() != "synthetic")).sum()),
        "test_used_for_training_or_selection": False,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "validation_selection.json").write_text(json.dumps({"selected_alpha": best["alpha"], "selected_validation": best["metrics"], "trials": trials}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "training.log").write_text("\n".join(f"alpha={row['alpha']} validation_macro_f1={row['validation']['macro_f1']:.6f}" for row in trials) + "\n", encoding="utf-8")
    print(json.dumps({"model": metadata["model_name"], "selected_alpha": best["alpha"], "validation_macro_f1": best["metrics"]["macro_f1"], "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
