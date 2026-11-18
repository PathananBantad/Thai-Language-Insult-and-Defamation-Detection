"""Shared binary metrics for training selection and final evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

LABEL_NAMES = {0: "Insult", 1: "Defamation"}
LABELS = (0, 1)


def calculate_binary_metrics(
    y_true: list[int], y_pred: list[int], probabilities: list[list[float]] | None = None,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=list(LABELS), target_names=[LABEL_NAMES[x] for x in LABELS],
        output_dict=True, zero_division=0,
    )
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(LABELS), average="weighted", zero_division=0)),
        "per_class": {
            str(label): {
                "name": LABEL_NAMES[label], "precision": float(precision[index]),
                "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist(),
        "classification_report": report,
        "labels": list(LABELS),
        "label_names": [LABEL_NAMES[x] for x in LABELS],
        "total_support": int(len(y_true)),
    }
    if probabilities is not None:
        result["mean_confidence"] = float(np.asarray(probabilities, dtype=float).max(axis=1).mean())
    if sources is not None:
        real_indices = [i for i, source in enumerate(sources) if str(source).strip().casefold() != "synthetic"]
        real_defamation = [i for i in real_indices if int(y_true[i]) == 1]
        correct_real_defamation = sum(int(y_pred[i]) == 1 for i in real_defamation)
        result["real_only"] = {
            "rows": len(real_indices),
            "defamation_support": len(real_defamation),
            "defamation_recall": float(correct_real_defamation / len(real_defamation)) if real_defamation else None,
            "synthetic_rows_excluded": len(sources) - len(real_indices),
        }
    return result
