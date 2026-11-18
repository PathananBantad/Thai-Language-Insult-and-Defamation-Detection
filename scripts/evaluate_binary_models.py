#!/usr/bin/env python
"""Evaluate completed binary model artifacts on Validation and Test."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thai_mod_api.binary_metrics import calculate_binary_metrics
from scripts.train_binary_transformer import predict_transformer_frame

DATA_DIR = ROOT / "datasets" / "processed" / "binary_insult_defamation_v1"
MODEL_ROOT = ROOT / "models" / "experiments" / "binary_insult_defamation_v1"
REPORT_ROOT = ROOT / "reports" / "binary_insult_defamation_v1" / "evaluations"


def evaluate_one(model_dir: Path, output_dir: Path, data_dir: Path) -> dict[str, Any]:
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("label_mapping") != {"0": "Insult", "1": "Defamation"}:
        raise ValueError(f"Unexpected label mapping in {model_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation outputs: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    results: dict[str, Any] = {}
    reports: list[str] = []
    confusion_rows: list[dict[str, Any]] = []
    test_defamation_support = 0
    for split in ("validation", "test"):
        frame = pd.read_csv(data_dir / f"{split}.csv", dtype={"text_norm": "string", "source": "string"})
        labels = frame["label"].astype(int).tolist()
        if split == "test":
            test_defamation_support = sum(label == 1 for label in labels)
        if metadata["model_type"] == "complement_nb":
            model = joblib.load(model_dir / "model.joblib")
            predictions = model.predict(frame["text_norm"].astype(str).tolist()).astype(int).tolist()
            probabilities = model.predict_proba(frame["text_norm"].astype(str).tolist()).tolist()
        elif metadata["model_type"] == "transformer":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(model_dir / "model", local_files_only=True, use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir / "model", local_files_only=True)
            model.to(device)
            predictions, probabilities = predict_transformer_frame(
                model, tokenizer, frame, max_length=int(metadata.get("max_length", 128)), device=device, batch_size=4
            )
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise ValueError(f"Unsupported model type {metadata.get('model_type')}")
        metrics = calculate_binary_metrics(
            labels, predictions, probabilities, frame["source"].astype(str).tolist()
        )
        results[split] = metrics
        reports.append(f"## {split.title()} (n={len(frame)})\n\n" + pd.DataFrame(metrics["classification_report"]).to_string() + "\n")
        matrix = metrics["confusion_matrix"]
        for i, true_name in enumerate(("Insult", "Defamation")):
            for j, predicted_name in enumerate(("Insult", "Defamation")):
                confusion_rows.append({"split": split, "true_label": true_name, "predicted_label": predicted_name, "count": matrix[i][j]})
        (output_dir / f"{split}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps({"model_name": metadata["model_name"], "model_id": metadata.get("model_id"), "validation": results["validation"], "test": results["test"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "classification_report.txt").write_text("\n".join(reports), encoding="utf-8")
    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("split", "true_label", "predicted_label", "count"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(confusion_rows)
    summary = (
        f"# Evaluation: {metadata['model_name']}\n\n"
        "This is a direct binary Insult/Defamation classifier. Test was opened only after the checkpoint was selected using Validation macro F1. No test threshold tuning was performed.\n\n"
        f"- Validation macro F1: {results['validation']['macro_f1']:.4f}; support Insult={results['validation']['per_class']['0']['support']}, Defamation={results['validation']['per_class']['1']['support']}.\n"
        f"- Test macro F1: {results['test']['macro_f1']:.4f}; support Insult={results['test']['per_class']['0']['support']}, Defamation={results['test']['per_class']['1']['support']}.\n"
        f"- Real-only Test Defamation recall: {results['test']['real_only']['defamation_recall']} on {results['test']['real_only']['defamation_support']} real/other Defamation samples.\n"
        + (
            f"- Test Defamation support is {test_defamation_support}; one missed Defamation changes recall by "
            f"{100 / test_defamation_support:.2f} percentage points.\n"
            if test_defamation_support
            else "- Test Defamation support is 0; recall and single-miss impact are undefined.\n"
        )
        + "The result is preliminary and does not establish generalization.\n"
    )
    (output_dir / "evaluation_summary.md").write_text(summary, encoding="utf-8")
    return {"model_name": metadata["model_name"], "directory": str(output_dir), "validation": results["validation"], "test": results["test"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()
    completed = []
    for name in ("complement_nb", "wangchanberta_v2", "phayathaibert_v1"):
        model_dir = args.model_root / name
        if not (model_dir / "metadata.json").is_file():
            print(f"SKIP {name}: completed model metadata not found")
            continue
        completed.append(evaluate_one(model_dir, args.output_dir / name, args.data_dir))
    if not completed:
        raise RuntimeError("No completed model artifacts were found; no evaluation results were written")
    comparison = pd.DataFrame([
        {"model": item["model_name"], "validation_macro_f1": item["validation"]["macro_f1"],
         "test_macro_f1": item["test"]["macro_f1"], "test_defamation_precision": item["test"]["per_class"]["1"]["precision"],
         "test_defamation_recall": item["test"]["per_class"]["1"]["recall"], "test_defamation_support": item["test"]["per_class"]["1"]["support"]}
        for item in completed
    ])
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    (args.output_dir / "completed_models.json").write_text(json.dumps(completed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
