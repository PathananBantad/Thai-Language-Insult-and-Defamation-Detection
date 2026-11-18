"""Score the fine-tuned WangchanBERTa on the evasion robustness set.

Usage:
    python scripts/evaluate_bert_evasion.py               # 3-class model -> docs/reports/bert_evasion.json
    python scripts/evaluate_bert_evasion.py --two-class   # 2-class model -> docs/reports/bert_evasion_2class.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.thai_mod_api.text_processing import preprocess  # noqa: E402

TWO = "--two-class" in sys.argv
MODEL_DIR = ROOT / "models" / "candidates" / ("wangchanberta_2class" if TWO else "wangchanberta_3class")
EVASION = ROOT / "datasets" / "annotation" / "evasion_testset.csv"
OUT = ROOT / "docs" / "reports" / ("bert_evasion_2class.json" if TWO else "bert_evasion.json")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    df = pd.read_csv(EVASION)
    if TWO:  # insult (1) / defamation (2) rows only, relabelled to 0 / 1
        df = df[df["label"] != 0].reset_index(drop=True)
        df["label"] = df["label"] - 1
    texts = [preprocess(t) for t in df["text"]]
    with torch.no_grad():
        enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        pred = model(**enc).logits.argmax(-1).cpu().numpy()
    df["pred"] = pred
    dis = df["variant"] != "plain"
    res = {
        "n": len(df),
        "n_disguised": int(dis.sum()),
        "accuracy_all": float((df["pred"] == df["label"]).mean()),
        "disguised_accuracy": float((df.loc[dis, "pred"] == df.loc[dis, "label"]).mean()),
        "errors": df.loc[df["pred"] != df["label"], ["text", "label", "pred"]].to_dict("records"),
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "errors"}, indent=2))


if __name__ == "__main__":
    main()
