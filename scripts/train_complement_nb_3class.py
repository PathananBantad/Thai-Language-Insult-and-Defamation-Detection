from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATA_DIR = ROOT / "datasets" / "processed"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "candidates" / "complement_nb_3class"

LABEL_NAMES = {0: "normal", 1: "insult", 2: "defamation"}
LABELS = tuple(LABEL_NAMES)
REQUIRED_COLUMNS = {"text_raw", "text_norm", "label", "label_name", "source"}
ALPHAS = (0.1, 0.3, 0.5, 1.0)

from src.thai_mod_api.text_processing import tokenize  # noqa: E402


def validate_split(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{split_name} is missing required columns: {sorted(missing_columns)}"
        )
    if frame.empty:
        raise ValueError(f"{split_name} must contain at least one row.")

    for column in ("text_norm", "text_raw", "label_name", "source"):
        values = frame[column]
        missing = values.isna() | values.astype("string").str.strip().eq("").fillna(True)
        if missing.any():
            raise ValueError(
                f"{split_name} contains missing or empty values in '{column}'."
            )

    for column in ("text_norm", "text_raw", "source"):
        if not frame[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{split_name} column '{column}' must contain text values.")

    if frame["label"].isna().any():
        raise ValueError(f"{split_name} contains missing values in 'label'.")

    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any():
        raise ValueError(f"{split_name} labels must be numeric values in {LABELS}.")
    if not labels.mod(1).eq(0).all():
        raise ValueError(f"{split_name} labels must be integers in {LABELS}.")
    labels = labels.astype(int)
    invalid_labels = set(labels.unique()) - set(LABELS)
    if invalid_labels:
        raise ValueError(f"{split_name} contains unsupported labels: {sorted(invalid_labels)}")

    expected_names = labels.map(LABEL_NAMES)
    actual_names = frame["label_name"].astype(str).str.strip().str.lower()
    if not actual_names.eq(expected_names).all():
        raise ValueError(f"{split_name} label_name values do not match label mapping.")

    texts = frame["text_norm"]
    if texts.isna().any() or not texts.map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError(f"{split_name} text_norm must contain non-empty strings.")

    validated = frame.copy()
    validated["label"] = labels
    return validated


def load_split(data_dir: Path, split_name: str) -> pd.DataFrame:
    if split_name not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split_name}")
    path = data_dir / f"{split_name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    return validate_split(
        pd.read_csv(
            path,
            dtype={
                "text_raw": "string",
                "text_norm": "string",
                "label_name": "string",
                "source": "string",
            },
        ),
        split_name,
    )


def make_model(alpha: float) -> Pipeline:
    return Pipeline(
        [
            (
                "vect",
                CountVectorizer(
                    tokenizer=tokenize,
                    token_pattern=None,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50_000,
                    lowercase=False,
                ),
            ),
            ("tfidf", TfidfTransformer()),
            ("clf", ComplementNB(alpha=alpha)),
        ]
    )


def fit_model(
    train: pd.DataFrame,
    alpha: float,
    model_factory: Callable[[float], Any] | None = None,
) -> Any:
    observed_labels = set(train["label"].astype(int).unique())
    if observed_labels != set(LABELS):
        raise ValueError(
            f"train must contain all labels {LABELS}; found {sorted(observed_labels)}."
        )

    factory = model_factory or make_model
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=train["label"].to_numpy(),
    )
    model = factory(alpha)
    model.fit(
        train["text_norm"].tolist(),
        train["label"].tolist(),
        clf__sample_weight=sample_weights,
    )
    return model


def calculate_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(LABELS),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "per_class": {
            str(label): {
                "name": LABEL_NAMES[label],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(LABELS),
        ).tolist(),
        "labels": list(LABELS),
        "total_support": int(len(y_true)),
    }


def evaluate_model(model: Any, split: pd.DataFrame) -> dict[str, Any]:
    predictions = model.predict(split["text_norm"].tolist())
    return calculate_metrics(split["label"].tolist(), predictions.tolist())


def select_best_configuration(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    alphas: tuple[float, ...] = ALPHAS,
    model_factory: Callable[[float], Any] | None = None,
) -> dict[str, Any]:
    if not alphas:
        raise ValueError("At least one alpha value is required.")
    if any(not math.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("Alpha values must be finite and greater than zero.")

    best: dict[str, Any] | None = None
    candidate_metrics = []
    for alpha in alphas:
        model = fit_model(train, alpha, model_factory=model_factory)
        validation_metrics = evaluate_model(model, validation)
        candidate_metrics.append(
            {"alpha": float(alpha), "validation_metrics": validation_metrics}
        )

        if best is None or validation_metrics["macro_f1"] > best["validation_metrics"]["macro_f1"]:
            best = {
                "alpha": float(alpha),
                "model": model,
                "validation_metrics": validation_metrics,
            }

    assert best is not None
    best["candidate_metrics"] = candidate_metrics
    return best


def _label_counts(split: pd.DataFrame) -> dict[str, int]:
    counts = split["label"].value_counts()
    return {str(label): int(counts.get(label, 0)) for label in LABELS}


def save_artifacts(
    output_dir: Path,
    model: Any,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing candidate artifacts: {output_dir}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        joblib.dump(model, staging_dir / "model.joblib")
        (staging_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging_dir.rename(output_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a three-class ComplementNB candidate."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing candidate artifacts: {args.output_dir}"
        )

    train = load_split(args.data_dir, "train")
    validation = load_split(args.data_dir, "validation")
    selection = select_best_configuration(train, validation)

    # Test is loaded and evaluated only after validation has selected alpha.
    test = load_split(args.data_dir, "test")
    test_metrics = evaluate_model(selection["model"], test)
    metrics = {
        "selection_metric": "validation macro_f1",
        "selected_alpha": selection["alpha"],
        "validation_candidates": selection["candidate_metrics"],
        "validation": selection["validation_metrics"],
        "test": test_metrics,
    }
    metadata = {
        "model_name": "CountVectorizer + TfidfTransformer + ComplementNB",
        "label_mapping": {str(label): name for label, name in LABEL_NAMES.items()},
        "input_column": "text_norm",
        "text_norm_source": (
            "datasets/processed/{split}.csv:text_norm; generated by "
            "scripts/prepare_dataset.py via "
            "src/thai_mod_api/text_processing.py::preprocess "
            "(the --no-fuzzy option controls fuzzy normalization)."
        ),
        "tokenizer": "PyThaiNLP word_tokenize(engine='newmm', keep_whitespace=False)",
        "alpha_grid": list(ALPHAS),
        "selected_alpha": selection["alpha"],
        "selection_metric": "validation macro_f1",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_label_counts": _label_counts(train),
        "validation_label_counts": _label_counts(validation),
        "test_label_counts": _label_counts(test),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    save_artifacts(args.output_dir, selection["model"], metadata, metrics)
    print(json.dumps({"artifact_dir": str(args.output_dir), **metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
