#!/usr/bin/env python
"""Create a reproducible binary Insult-vs-Defamation dataset without touching sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUT = ROOT / "datasets" / "processed" / "binary_insult_defamation_v1"
SPLITS = ("train", "validation", "test")
EXPECTED_ORIGINAL = {
    "train": {0: 11926, 1: 7190, 2: 368},
    "validation": {0: 2555, 1: 1541, 2: 16},
    "test": {0: 2556, 1: 1541, 2: 15},
}
SOURCE_COLUMNS = ("text_raw", "text_norm", "label", "label_name", "source")
OUTPUT_COLUMNS = (
    "sample_id", "split", "source_row", "text_raw", "source_text_norm", "text_norm",
    "original_label", "original_label_name", "label", "label_name", "source", "is_synthetic",
)
LABEL_NAMES = {0: "Insult", 1: "Defamation"}
EXCLUSION_COLUMNS = (
    "sample_id", "source_row", "original_split", "original_label", "source",
    "exclusion_reason", "review_ids", "source_review_queues",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def map_binary_label(original_label: int) -> int | None:
    if original_label == 0:
        return None
    if original_label == 1:
        return 0
    if original_label == 2:
        return 1
    raise ValueError(f"Unsupported original label: {original_label}")


def stable_sample_id(split: str, source_row: int) -> str:
    if split not in SPLITS or source_row < 2:
        raise ValueError(f"Invalid source identity: {split}/{source_row}")
    return f"{split}-{source_row:06d}"


def _validate_source(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    missing = set(SOURCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{split} missing source columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{split} source split is empty")
    data = frame.copy()
    if data[list(SOURCE_COLUMNS)].isna().any().any():
        raise ValueError(f"{split} has missing values in required source columns")
    labels = pd.to_numeric(data["label"], errors="coerce")
    if labels.isna().any() or not labels.map(lambda v: float(v).is_integer()).all():
        raise ValueError(f"{split} has non-integer labels")
    data["label"] = labels.astype(int)
    if not set(data["label"].unique()).issubset({0, 1, 2}):
        raise ValueError(f"{split} has unsupported labels")
    expected_names = data["label"].map({0: "normal", 1: "insult", 2: "defamation"})
    if not data["label_name"].astype(str).str.strip().str.lower().eq(expected_names).all():
        raise ValueError(f"{split} label_name does not agree with original label")
    return data


def validate_review_queues(
    train: pd.DataFrame, phase4_path: Path, phase5_path: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Return the validated exclusion set and verify both queues against Train rows."""
    q4 = pd.read_csv(phase4_path, dtype=str, keep_default_na=False)
    q5 = pd.read_csv(phase5_path, dtype=str, keep_default_na=False)
    required = {"review_id", "sample_id", "source_row", "split", "source", "text_raw", "text_norm", "current_label", "current_label_name"}
    if not required.issubset(q4.columns) or not required.issubset(q5.columns):
        raise ValueError("Review queue schema is missing required lineage columns")
    if len(q4) != 200 or len(q5) != 200:
        raise ValueError(f"Expected 200 rows in each review queue; got {len(q4)} and {len(q5)}")
    if q4["sample_id"].duplicated().any() or q5["sample_id"].duplicated().any():
        raise ValueError("A review queue contains duplicate sample_id values")
    if set(q4["sample_id"]) != set(q5["sample_id"]):
        raise ValueError("Phase 4.1 and Phase 5.1 queues have different sample_id sets")
    if q4["review_id"].duplicated().any() or q5["review_id"].duplicated().any():
        raise ValueError("A review queue contains duplicate review_id values")
    q5_by_id = q5.set_index("sample_id", drop=False)
    exclusions: dict[str, dict[str, Any]] = {}
    queue_refs: dict[str, str] = {}
    for row in q4.to_dict(orient="records"):
        sample_id = row["sample_id"]
        try:
            source_row = int(row["source_row"])
        except ValueError as exc:
            raise ValueError(f"Invalid source_row in review row {row['review_id']}") from exc
        expected_queue_id = f"train-row-{source_row:06d}"
        source_index = source_row - 2
        if row["sample_id"] != expected_queue_id or row["split"] != "train" or row["current_label"] != "1":
            raise ValueError(f"Review row has unexpected identity/split/label: {row['review_id']}")
        if not 0 <= source_index < len(train):
            raise ValueError(f"Review source_row is outside Train: {row['review_id']}")
        original = train.iloc[source_index]
        if (
            str(original["text_raw"]) != row["text_raw"]
            or str(original["text_norm"]) != row["text_norm"]
            or str(original["source"]) != row["source"]
            or int(original["label"]) != 1
            or str(original["label_name"]).lower() != "insult"
        ):
            raise ValueError(f"Review row does not exact-match original Train: {row['review_id']}")
        q5_row = q5_by_id.loc[sample_id]
        for field in ("review_id", "source_row", "split", "source", "text_raw", "text_norm", "current_label", "current_label_name"):
            if str(q5_row[field]) != str(row[field]):
                raise ValueError(f"Phase 5.1 differs from Phase 4.1 for {sample_id}/{field}")
        cascade_id = f"train-{source_row:06d}"
        exclusions[cascade_id] = {
            "sample_id": cascade_id,
            "source_row": source_row,
            "original_split": "train",
            "original_label": 1,
            "source": str(original["source"]),
            "review_ids": f"{row['review_id']}",
            "source_review_queues": "phase4_1_insult_review_queue.csv;phase5_1_priority_review_queue.csv",
        }
        queue_refs[cascade_id] = row["review_id"]
    if len(exclusions) != 200:
        raise ValueError(f"Expected 200 unique exclusions, got {len(exclusions)}")
    return exclusions, queue_refs


def derive_split(
    split: str,
    frame: pd.DataFrame,
    excluded: dict[str, dict[str, Any]],
    excluded_sources: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    from src.thai_mod_api.binary_preprocessing import normalize_binary_text

    data = _validate_source(frame, split)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(data.to_dict(orient="records")):
        source_row = index + 2
        sample_id = stable_sample_id(split, source_row)
        if sample_id in seen:
            raise ValueError(f"Duplicate generated sample_id: {sample_id}")
        seen.add(sample_id)
        if map_binary_label(int(source["label"])) is None:
            continue
        if split == "train" and sample_id in excluded:
            continue
        if str(source["source"]) in excluded_sources:
            continue
        normalized = normalize_binary_text(source["text_raw"])
        if not normalized:
            raise ValueError(f"Preprocessing produced empty text at {sample_id}")
        original_label = int(source["label"])
        rows.append({
            "sample_id": sample_id,
            "split": split,
            "source_row": source_row,
            "text_raw": source["text_raw"],
            "source_text_norm": source["text_norm"],
            "text_norm": normalized,
            "original_label": original_label,
            "original_label_name": str(source["label_name"]),
            "label": map_binary_label(original_label),
            "label_name": LABEL_NAMES[map_binary_label(original_label)],
            "source": str(source["source"]),
            "is_synthetic": str(str(source["source"]).strip().casefold() == "synthetic").lower(),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(row["label"]) for row in rows)
    return {str(label): counts.get(label, 0) for label in LABEL_NAMES}


def validate_unique_sample_ids(rows: list[dict[str, Any]], context: str) -> None:
    identifiers = [str(row["sample_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate sample_id values in {context}")


def prepare(
    data_dir: Path,
    phase4_path: Path,
    phase5_path: Path,
    output_dir: Path,
    excluded_sources: tuple[str, ...] = (),
) -> dict[str, Any]:
    from src.thai_mod_api.binary_preprocessing import preprocessing_metadata

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    input_paths = {split: data_dir / f"{split}.csv" for split in SPLITS}
    input_paths.update({"phase4_queue": phase4_path, "phase5_queue": phase5_path})
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")
    before_hashes = {key: sha256(path) for key, path in input_paths.items()}
    originals = {split: _validate_source(pd.read_csv(input_paths[split]), split) for split in SPLITS}
    for split, expected in EXPECTED_ORIGINAL.items():
        actual = originals[split]["label"].value_counts().to_dict()
        if actual != expected:
            raise ValueError(f"Unexpected {split} source counts: {actual}; expected {expected}")
    reviewed_exclusions, _ = validate_review_queues(originals["train"], phase4_path, phase5_path)
    exclusions = {
        sample_id: {**row, "exclusion_reason": "temporary_human_review_pending"}
        for sample_id, row in reviewed_exclusions.items()
    }
    source_exclusion_counts: dict[str, int] = {}
    for split, frame in originals.items():
        excluded_count = 0
        for index, source in enumerate(frame.to_dict(orient="records")):
            source_name = str(source["source"])
            if source_name not in excluded_sources:
                continue
            excluded_count += 1
            sample_id = stable_sample_id(split, index + 2)
            existing = exclusions.get(sample_id, {})
            reasons = [f"source_excluded:{source_name}"]
            if sample_id in reviewed_exclusions:
                reasons.append("temporary_human_review_pending")
            exclusions[sample_id] = {
                "sample_id": sample_id,
                "source_row": index + 2,
                "original_split": split,
                "original_label": int(source["label"]),
                "source": source_name,
                "exclusion_reason": ";".join(reasons),
                "review_ids": existing.get("review_ids", ""),
                "source_review_queues": existing.get("source_review_queues", ""),
            }
        source_exclusion_counts[split] = excluded_count

    results = {
        split: derive_split(split, originals[split], reviewed_exclusions, excluded_sources)
        for split in SPLITS
    }
    all_ids: set[str] = set()
    for split, rows in results.items():
        validate_unique_sample_ids(rows, split)
        duplicate_ids = all_ids.intersection(str(row["sample_id"]) for row in rows)
        if duplicate_ids:
            raise ValueError(f"Sample IDs overlap splits: {sorted(duplicate_ids)[:5]}")
        all_ids.update(str(row["sample_id"]) for row in rows)

    for split, rows in results.items():
        actual = {int(k): v for k, v in _counts(rows).items()}
        expected_frame = originals[split]
        expected_frame = expected_frame[~expected_frame["source"].isin(excluded_sources)]
        if split == "train":
            expected_ids = expected_frame.index.map(lambda index: stable_sample_id(split, int(index) + 2))
            expected_frame = expected_frame[~expected_ids.isin(reviewed_exclusions)]
        expected_original_counts = expected_frame["label"].value_counts().to_dict()
        expected = {
            0: int(expected_original_counts.get(1, 0)),
            1: int(expected_original_counts.get(2, 0)),
        }
        if actual != expected:
            raise ValueError(f"Unexpected {split} binary counts: {actual}; expected {expected}")
        if any(row["source"] in excluded_sources for row in rows):
            raise ValueError(f"{split} contains an excluded source")
    after_hashes = {key: sha256(path) for key, path in input_paths.items()}
    if before_hashes != after_hashes:
        raise RuntimeError("An input file changed while preparing the binary dataset")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}_", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        for split, rows in results.items():
            write_csv(stage / f"{split}.csv", rows, OUTPUT_COLUMNS)
        exclusion_rows = sorted(
            exclusions.values(),
            key=lambda item: (item["original_split"], item["source_row"]),
        )
        write_csv(stage / "exclusion_log.csv", exclusion_rows, EXCLUSION_COLUMNS)
        source_counts = {
            split: {
                "label_counts": _counts(rows),
                "source_by_label": {
                    LABEL_NAMES[label]: dict(Counter(row["source"] for row in rows if row["label"] == label))
                    for label in LABEL_NAMES
                },
                "synthetic_defamation": sum(row["label"] == 1 and row["is_synthetic"] == "true" for row in rows),
                "real_or_other_defamation": sum(row["label"] == 1 and row["is_synthetic"] != "true" for row in rows),
                "rows": len(rows),
            }
            for split, rows in results.items()
        }
        summary = {
            "dataset_name": "THAI-MOD Binary Insult vs Defamation",
            "version": output_dir.name,
            "task": "direct_binary_classification",
            "label_mapping": {"original_1": {"binary": 0, "name": "Insult"}, "original_2": {"binary": 1, "name": "Defamation"}, "original_0": "excluded"},
            "splits": source_counts,
            "original_split_counts": {
                split: {str(int(label)): int(count) for label, count in originals[split]["label"].value_counts().items()}
                for split in SPLITS
            },
            "sample_ids_unique_across_splits": True,
            "excluded_unique_train_samples": len(reviewed_exclusions),
            "excluded_reviewed_train_samples": len(reviewed_exclusions),
            "excluded_reviewed_sample_ids": sorted(reviewed_exclusions),
            "exclusion_reason": "temporary_human_review_pending",
            "exclusion_sources": [phase4_path.name, phase5_path.name],
            "excluded_sources": list(excluded_sources),
            "source_exclusion_counts_by_split": source_exclusion_counts,
            "preprocessing": preprocessing_metadata(),
            "input_sha256": before_hashes,
            "input_sha256_after": after_hashes,
            "synthetic_defamation_added": False,
        }
        (stage / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        source_exclusion_note = (
            f"All rows whose source is {', '.join(excluded_sources)} are excluded from every split. "
            "Their lineage is recorded in `exclusion_log.csv`."
            if excluded_sources
            else ""
        )
        readme = f"""# Binary Insult vs Defamation Dataset {output_dir.name}

This derived dataset maps original label 1 (Insult) to binary label 0 and original label 2 (Defamation) to binary label 1. Original Normal (0) rows are not included. The original split membership is preserved.

The 200 unique Phase 4.1/5.1 pending-review Insult rows are excluded from Train only. They are not relabeled; the original datasets and review queues remain unchanged. See `exclusion_log.csv` for every stable sample ID, source row, reason, review IDs and queue references.

{source_exclusion_note}

`text_raw` preserves the original message. `source_text_norm` preserves the pre-existing normalization. `text_norm` is regenerated with `binary_preprocess_v1` and must be used by training and inference for this experiment. The `source`, original labels and synthetic flag retain provenance.

Validation and Test are not excluded or resampled. Train Defamation includes synthetic examples from the pre-existing processed split; no synthetic examples were created here. This is a preliminary risk-screening dataset, not a legal determination.
"""
        (stage / "README.md").write_text(readme, encoding="utf-8")
        after_hashes = {key: sha256(path) for key, path in input_paths.items()}
        if before_hashes != after_hashes:
            raise RuntimeError("An input file changed while writing the binary dataset")
        summary["input_sha256_after"] = after_hashes
        (stage / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
        stage.rename(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "processed")
    parser.add_argument("--phase4-queue", type=Path, default=ROOT / "datasets" / "exploration" / "phase4_1_insult_review_queue.csv")
    parser.add_argument("--phase5-queue", type=Path, default=ROOT / "datasets" / "exploration" / "phase5_1_priority_review_queue.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--exclude-source", action="append", default=[], help="Exclude this exact source value from every split; may be repeated")
    args = parser.parse_args()
    summary = prepare(
        args.data_dir,
        args.phase4_queue,
        args.phase5_queue,
        args.output_dir,
        tuple(args.exclude_source),
    )
    print(json.dumps({"output_dir": str(args.output_dir), "splits": summary["splits"], "excluded_unique_train_samples": summary["excluded_unique_train_samples"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
