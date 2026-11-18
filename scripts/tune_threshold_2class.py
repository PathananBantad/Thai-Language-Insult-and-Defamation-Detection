"""Pick the defamation decision threshold for the 2-class WangchanBERTa on VALIDATION.

With 99:1 class imbalance the default argmax (threshold 0.5) almost never
predicts defamation. We choose the threshold that maximises defamation F1 on
the validation split, then apply it once to test.

Usage:
    python scripts/tune_threshold_2class.py   # -> docs/reports/model_metrics/wangchanberta_2class_threshold.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.thai_mod_api.text_processing import preprocess  # noqa: E402

MODEL_DIR = ROOT / "models" / "candidates" / "wangchanberta_2class"
DATA = ROOT / "datasets" / "processed_2class"
OUT = ROOT / "docs" / "reports" / "model_metrics" / "wangchanberta_2class_threshold.json"


def probs(model, tok, texts, device, bs=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            out.append(torch.softmax(model(**enc).logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def metrics(y, pred):
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
            "insult": {"p": p[0], "r": r[0], "f1": f[0], "n": int(n[0])}, "defamation": {"p": p[1], "r": r[1], "f1": f[1], "n": int(n[1])},
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval().to(device)
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")
    pv = probs(model, tok, list(val["text_norm"]), device)
    grid = np.round(np.arange(0.02, 0.51, 0.01), 2)
    scores = [(t, f1_score(val["label"], (pv >= t).astype(int), zero_division=0)) for t in grid]
    best_f1 = max(s for _, s in scores)
    thr = float(max(t for t, s in scores if s == best_f1))  # highest threshold among ties = fewest false alarms
    pt = probs(model, tok, list(test["text_norm"]), device)

    eva = pd.read_csv(ROOT / "datasets" / "annotation" / "evasion_testset.csv")
    eva = eva[eva["label"] != 0].reset_index(drop=True)
    eva["label"] -= 1
    pe = probs(model, tok, [preprocess(t) for t in eva["text"]], device)
    dis = (eva["variant"] != "plain").to_numpy()
    pred_e = (pe >= thr).astype(int)

    res = {
        "threshold": thr,
        "selected_on": "validation defamation F1",
        "validation": metrics(val["label"], (pv >= thr).astype(int)),
        "test": metrics(test["label"], (pt >= thr).astype(int)),
        "test_default_0.5": metrics(test["label"], (pt >= 0.5).astype(int)),
        "evasion_disguised_accuracy": float((pred_e[dis] == eva["label"].to_numpy()[dis]).mean()),
        "evasion_n_disguised": int(dis.sum()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    t = res["test"]
    print(f"threshold={thr}  val defF1={res['validation']['defamation']['f1']:.3f}  "
          f"test macroF1={t['macro_f1']:.3f} def P/R/F1={t['defamation']['p']:.2f}/{t['defamation']['r']:.2f}/{t['defamation']['f1']:.2f} "
          f"cm={t['confusion_matrix']}  evasion={res['evasion_disguised_accuracy']:.3f}")


if __name__ == "__main__":
    main()
