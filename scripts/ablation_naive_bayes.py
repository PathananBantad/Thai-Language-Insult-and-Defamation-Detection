"""Ablation study with the Naive Bayes baseline.

Questions answered:
  1. Does evasion normalization help?    clean_text only  vs  preprocess()
  2. Do synthetic defamation rows help?  train with  vs  without  source == "Synthetic"

All variants are scored on the VALIDATION split (the test split stays untouched
for the final model comparison) and on the evasion robustness set.

Usage:
    python scripts/ablation_naive_bayes.py      # -> docs/reports/ablation_naive_bayes.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.thai_mod_api.text_processing import clean_text, preprocess, tokenize  # noqa: E402

PROCESSED = ROOT / "datasets" / "processed"
EVASION = ROOT / "datasets" / "annotation" / "evasion_testset.csv"
OUT = ROOT / "docs" / "reports" / "ablation_naive_bayes.md"
LABELS = [0, 1, 2]
NAMES = ["normal", "insult", "defamation"]


def make_model() -> Pipeline:
    # ComplementNB is the NB variant designed for imbalanced text classes.
    return Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                 ngram_range=(1, 2), min_df=2, max_features=50000)),
        ("tfidf", TfidfTransformer()),
        ("clf", ComplementNB(alpha=0.3)),
    ])


def evaluate(model: Pipeline, texts: list[str], y: pd.Series) -> dict:
    pred = model.predict(texts)
    rep = classification_report(y, pred, labels=LABELS, target_names=NAMES, output_dict=True, zero_division=0)
    return {
        "macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0),
        **{f"{n}_f1": rep[n]["f1-score"] for n in NAMES},
        **{f"{n}_recall": rep[n]["recall"] for n in NAMES},
        "pred": pred,
    }


def main() -> None:
    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "validation.csv")
    eva = pd.read_csv(EVASION)

    # Input variants: "clean" = no evasion normalization, "full" = preprocess()
    for df in (train, val, eva):
        raw = df["text_raw"] if "text_raw" in df else df["text"]
        df["clean"] = [clean_text(t) for t in raw]
        df["full"] = df["text_norm"] if "text_norm" in df else [preprocess(t) for t in raw]

    rows = []
    for use_norm in (False, True):
        for use_syn in (False, True):
            col = "full" if use_norm else "clean"
            tr = train if use_syn else train[train["source"] != "Synthetic"]
            model = make_model()
            model.fit(tr[col], tr["label"], clf__sample_weight=compute_sample_weight("balanced", tr["label"]))
            v = evaluate(model, list(val[col]), val["label"])
            e = evaluate(model, list(eva[col]), eva["label"])
            dis = eva["variant"] != "plain"
            rows.append({
                "normalization": "yes" if use_norm else "no",
                "synthetic": "yes" if use_syn else "no",
                "val_macro_f1": v["macro_f1"],
                "val_normal_f1": v["normal_f1"],
                "val_insult_f1": v["insult_f1"],
                "val_defamation_f1": v["defamation_f1"],
                "val_defamation_recall": v["defamation_recall"],
                "evasion_acc_disguised": (e["pred"][dis.values] == eva.loc[dis, "label"].values).mean(),
                "evasion_acc_all": (e["pred"] == eva["label"].values).mean(),
            })
            print(rows[-1], flush=True)

    res = pd.DataFrame(rows)
    n_def = int((val["label"] == 2).sum())
    lines = [
        "# Naive Bayes ablation",
        "",
        "Model: CountVectorizer(newmm, 1-2 gram) + TF-IDF + ComplementNB, balanced sample weights.",
        f"Scored on the **validation** split (n = {len(val):,}, defamation n = {n_def}) and on the "
        f"evasion robustness set (`{EVASION.name}`, {len(eva)} rows, {int((eva['variant'] != 'plain').sum())} disguised).",
        "",
        res.to_markdown(index=False, floatfmt=".3f"),
        "",
        "- **normalization** = evasion normalization on (`preprocess`) vs off (`clean_text` only)",
        "- **synthetic** = synthetic defamation rows kept in train vs removed",
        "- **evasion_acc_disguised** = accuracy on the disguised sentences only",
        "",
        f"Note: validation has only {n_def} defamation examples, so defamation F1 moves a lot with a single "
        "prediction. Treat differences there as indicative, not conclusive.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
