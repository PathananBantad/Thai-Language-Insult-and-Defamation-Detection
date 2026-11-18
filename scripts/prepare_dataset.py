"""Build the Thai cyberbullying dataset (Normal / Insult / Defamation).

Usage (from the repo root):
    python scripts/prepare_dataset.py

Outputs in datasets/processed/:
    train.csv, validation.csv, test.csv   columns: text_raw, text_norm, label, label_name, source
    data_report.md                        counts for every cleaning step + class distribution

Label scheme (see datasets/annotation_guidelines.md):
    0 normal      ไม่ได้โจมตีบุคคล
    1 insult      ดูหมิ่น: ด่า/เหยียด/ดูถูก เป็นความเห็นหรือคำด่า
    2 defamation  เข้าข่ายหมิ่นประมาท: ระบุตัวบุคคลได้ + กล่าวหาข้อเท็จจริงที่ทำให้เสียชื่อเสียง
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.thai_mod_api.text_processing import preprocess  # noqa: E402

LABEL_NAMES = {0: "normal", 1: "insult", 2: "defamation"}
TWO_CLASS_NAMES = {0: "insult", 1: "defamation"}

# Source datasets and how their original labels map to ours.
# None = drop rows with that original label.
SOURCES: dict[str, dict] = {
    # Thai Toxicity Tweet Corpus: tweets collected with toxic keywords
    "dataset2": {"name": "ThaiToxicityTweet", "map": {"neg": 1, "neu": 0}},
    # HateThaiSent: hate speech toward migrant workers + sentiment
    "dataset3": {"name": "HateThaiSent", "map": {"neg": 1, "neu": 0, "pos": 0}},
    # Thai Cyberbullying LGBT
    "dataset5": {"name": "ThaiCyberbullyingLGBT", "map": {"neg": 1, "neu": 0}},
    # Wisesight Sentiment: 'neg' means negative sentiment, NOT insult -> dropped.
    # neu/pos are used (capped) as extra normal examples.
    "dataset1": {"name": "WisesightSentiment", "map": {"neg": None, "neu": 0, "pos": 0}, "cap": 4000},
}
# dataset4 (product-review sentiment) and dataset6-8 (English) are out of scope.

# Adult-content spam found in dataset5 (porn ads, not cyberbullying).
SPAM_PATTERN = r"หนังโป๊|คลิปโป๊|หนังเอวี|ขุดทอง|ดูหนังออนไลน์|แหล่งคลิป|คลิปหลุด|เย็ดหี|ไซด์ไลน์"

MIN_NORM_LEN = 3
SEED = 42


def load_sources(data_dir: Path, report: list[str]) -> pd.DataFrame:
    frames = []
    for key, cfg in SOURCES.items():
        df = pd.read_csv(data_dir / f"{key}.csv")
        df = df.rename(columns={"texts": "text_raw", "category": "orig_label"})[["text_raw", "orig_label"]]
        df["label"] = df["orig_label"].map(cfg["map"])
        before = len(df)
        df = df.dropna(subset=["label"])
        if "cap" in cfg and len(df) > cfg["cap"]:
            df = df.sample(cfg["cap"], random_state=SEED)
        df["source"] = cfg["name"]
        report.append(f"| {cfg['name']} ({key}) | {before:,} | {len(df):,} |")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["label"] = out["label"].astype(int)
    return out


SOURCE_NAMES = {k: v["name"] for k, v in SOURCES.items()}


def apply_manual_labels(df: pd.DataFrame, manual_path: Path) -> tuple[pd.DataFrame, int, int]:
    """Hand-checked labels win over the mapped source labels.

    Manually labelled texts that were not loaded (e.g. Wisesight 'neg' rows,
    which are otherwise dropped, or rows outside the Wisesight sample) are
    appended so no manual label is lost.
    """
    manual = pd.read_csv(manual_path)
    lookup = dict(zip(manual["text"], manual["label"]))
    mask = df["text_raw"].isin(lookup)
    df.loc[mask, "label"] = df.loc[mask, "text_raw"].map(lookup).astype(int)
    df["manual"] = mask
    missing = manual[~manual["text"].isin(df["text_raw"])]
    extra = pd.DataFrame({
        "text_raw": missing["text"],
        "orig_label": "manual",
        "label": missing["label"].astype(int),
        "source": missing["source"].map(SOURCE_NAMES).fillna(missing["source"]),
        "manual": True,
    })
    return pd.concat([df, extra], ignore_index=True), int(mask.sum()), len(extra)


def resolve_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Deduplicate on normalized text.

    Same text + same label -> keep one. Conflicting labels -> keep the manual
    label if there is one, otherwise drop the text entirely (label noise).
    """
    n_dupes = 0
    n_conflict = 0
    keep = []
    for _, g in df.groupby("text_norm", sort=False):
        if len(g) == 1:
            keep.append(g.index[0])
            continue
        n_dupes += len(g) - 1
        manual = g[g["manual"]]
        if len(manual):
            keep.append(manual.index[0])
        elif g["label"].nunique() == 1:
            keep.append(g.index[0])
        else:
            n_conflict += 1
    return df.loc[keep], n_dupes, n_conflict


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=SEED)
    return train, val, test


def distribution_table(parts: dict[str, pd.DataFrame], names: dict[int, str] = LABEL_NAMES) -> list[str]:
    keys = sorted(names)
    lines = ["| split | " + " | ".join(f"{names[k]} ({k})" for k in keys) + " | total |", "|---" * (len(keys) + 2) + "|"]
    for name, part in parts.items():
        c = part["label"].value_counts()
        cells = [f"{c.get(k, 0):,} ({c.get(k, 0) / len(part):.1%})" for k in keys]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {len(part):,} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    ap.add_argument("--manual", type=Path, default=ROOT / "datasets" / "annotation" / "manual_labels.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "datasets" / "processed")
    ap.add_argument("--synthetic", type=Path, default=ROOT / "datasets" / "annotation" / "synthetic_defamation.csv",
                    help="synthetic defamation rows, added to TRAIN only")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--two-class", action="store_true",
                    help="insult (0) vs defamation (1) only, as in the proposal; normal texts are dropped")
    ap.add_argument("--no-fuzzy", action="store_true", help="skip edit-distance correction (faster)")
    args = ap.parse_args()

    report = ["# Data preparation report", "", "## 1. Sources", "",
              "| source | rows in file | rows used |", "|---|---|---|"]
    df = load_sources(args.data_dir, report)
    steps = [("loaded (after label mapping)", len(df))]

    df, n_manual, n_manual_added = apply_manual_labels(df, args.manual)
    steps.append(("+ manually labelled rows not in the loaded sources", len(df)))

    df = df.dropna(subset=["text_raw"])
    df["text_raw"] = df["text_raw"].astype(str)
    df = df[df["text_raw"].str.strip() != ""]
    steps.append(("drop empty text", len(df)))

    spam = df["text_raw"].str.contains(SPAM_PATTERN, regex=True) & ~df["manual"]
    df = df[~spam]
    steps.append(("drop adult-content spam", len(df)))

    print(f"normalizing {len(df):,} texts ...", flush=True)
    df["text_norm"] = [preprocess(t, use_fuzzy=not args.no_fuzzy) for t in df["text_raw"]]
    content = df["text_norm"].str.replace(r"<USER>|:[a-z_]+:", "", regex=True).str.strip()
    df = df[content.str.len() >= MIN_NORM_LEN]
    steps.append((f"drop texts shorter than {MIN_NORM_LEN} chars after cleaning", len(df)))

    df, n_dupes, n_conflict = resolve_duplicates(df)
    steps.append((f"deduplicate on normalized text ({n_dupes:,} duplicates, {n_conflict:,} conflicting texts dropped)", len(df)))

    names = LABEL_NAMES
    if args.two_class:
        # Proposal scope: classify harmful texts into insult vs defamation.
        before = len(df)
        df = df[df["label"] != 0].copy()
        df["label"] = df["label"] - 1
        names = TWO_CLASS_NAMES
        steps.append((f"two-class mode: drop normal texts ({before - len(df):,} rows)", len(df)))
    df["label_name"] = df["label"].map(names)
    train, val, test = split(df)

    n_synth = 0
    if not args.no_synthetic and args.synthetic.exists():
        syn = pd.read_csv(args.synthetic)
        syn = pd.DataFrame({"text_raw": syn["text"], "label": syn["label"].astype(int), "source": "Synthetic"})
        syn["text_norm"] = [preprocess(t, use_fuzzy=not args.no_fuzzy) for t in syn["text_raw"]]
        syn = syn[~syn["text_norm"].isin(df["text_norm"])].drop_duplicates("text_norm")
        if args.two_class:
            syn["label"] = syn["label"] - 1
        syn["label_name"] = syn["label"].map(names)
        n_synth = len(syn)
        train = pd.concat([train, syn], ignore_index=True).sample(frac=1, random_state=SEED)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["text_raw", "text_norm", "label", "label_name", "source"]
    for name, part in (("train", train), ("validation", val), ("test", test)):
        part[cols].to_csv(args.out_dir / f"{name}.csv", index=False, encoding="utf-8")

    report += ["", "## 2. Cleaning steps", "", "| step | rows |", "|---|---|"]
    report += [f"| {s} | {n:,} |" for s, n in steps]
    report += ["", f"Manual labels: {n_manual:,} rows relabelled in place, {n_manual_added:,} rows added "
               f"from `{args.manual.name}`.",
               f"Synthetic defamation rows added to **train only**: {n_synth:,} "
               "(validation and test contain real data only).", "",
               "## 3. Class distribution (stratified 70/15/15, seed 42)", ""]
    report += distribution_table({"train": train, "validation": val, "test": test, "all (real only)": df}, names)
    report += ["", "## 4. Source x label", "", df.pivot_table(index="source", columns="label_name",
               values="text_norm", aggfunc="count", fill_value=0).to_markdown()]
    (args.out_dir / "data_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
