"""How well does the rule-based target + allegation analyzer flag defamation?

Runs ``allegation_analyzer.analyze`` on the real validation + test rows and
compares its ``defamation_risk`` flag with the gold label (2 = defamation).

Usage:
    python scripts/evaluate_rule_analyzer.py   # -> docs/reports/rule_analyzer.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.thai_mod_api.allegation_analyzer import analyze  # noqa: E402

OUT = ROOT / "docs" / "reports" / "rule_analyzer.md"


def main() -> None:
    df = pd.concat([pd.read_csv(ROOT / "datasets" / "processed" / f"{s}.csv") for s in ("validation", "test")])
    df["risk"] = [analyze(t)["defamation_risk"] for t in df["text_raw"]]
    gold = df["label"] == 2
    tp = int((df["risk"] & gold).sum())
    fp = int((df["risk"] & ~gold).sum())
    fn = int((~df["risk"] & gold).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    by_label = df.groupby("label_name")["risk"].mean().rename("flagged_rate")

    lines = [
        "# Rule-based defamation flag (target + allegation)",
        "",
        f"Data: real validation + test rows (n = {len(df):,}, defamation n = {int(gold.sum())}).",
        "",
        "| metric | value |", "|---|---|",
        f"| precision | {precision:.3f} |", f"| recall | {recall:.3f} |", f"| F1 | {f1:.3f} |",
        f"| TP / FP / FN | {tp} / {fp} / {fn} |",
        "",
        "Share of each gold class flagged as defamation risk:",
        "",
        by_label.to_frame().to_markdown(floatfmt=".3f"),
        "",
        "High recall + low precision is expected: the rules find *candidates* (target + accusation words) and the "
        "classifier decides. The flag and its evidence spans are meant for explanation/highlighting, not as the "
        "final label.",
        "",
        "## Missed defamation examples (FN)",
        "",
    ]
    for t in df.loc[~df["risk"] & gold, "text_raw"].head(10):
        lines.append(f"- {t[:120]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
