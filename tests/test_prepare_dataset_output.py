import sys
from pathlib import Path

import pandas as pd

from scripts import prepare_dataset

ROOT = prepare_dataset.ROOT
resolve_output_dir = prepare_dataset.resolve_output_dir


def test_two_class_defaults_to_separate_directory():
    assert resolve_output_dir(None, two_class=True) == ROOT / "datasets" / "processed_2class"


def test_explicit_output_directory_takes_precedence():
    explicit_dir = Path("custom-output")

    assert resolve_output_dir(explicit_dir, two_class=True) == explicit_dir


def test_three_class_default_remains_processed():
    assert resolve_output_dir(None, two_class=False) == ROOT / "datasets" / "processed"


def test_two_class_cli_maps_labels_and_keeps_synthetic_rows_in_train(tmp_path, monkeypatch):
    data_dir = tmp_path / "sources"
    data_dir.mkdir()
    for source in ("dataset1", "dataset2", "dataset5"):
        if source == "dataset1":
            rows = [{"texts": f"normal sample {i}", "category": "neu"} for i in range(10)]
        else:
            rows = [{"texts": f"{source} insult {i}", "category": "neg"} for i in range(20)]
        pd.DataFrame(rows).to_csv(data_dir / f"{source}.csv", index=False)

    manual_path = tmp_path / "manual.csv"
    pd.DataFrame(
        [
            {"text": f"defamation sample {i}", "label": 2, "source": "manual_test"}
            for i in range(20)
        ]
    ).to_csv(manual_path, index=False)
    synthetic_path = tmp_path / "synthetic.csv"
    pd.DataFrame(
        [{"text": f"synthetic defamation {i}", "label": 2} for i in range(2)]
    ).to_csv(synthetic_path, index=False)

    output_dir = tmp_path / "processed_2class"
    monkeypatch.setattr(prepare_dataset, "preprocess", lambda text, use_fuzzy=True: text.lower())
    monkeypatch.setattr(pd.DataFrame, "to_markdown", lambda self: "| test report |")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_dataset.py",
            "--data-dir",
            str(data_dir),
            "--manual",
            str(manual_path),
            "--synthetic",
            str(synthetic_path),
            "--out-dir",
            str(output_dir),
            "--two-class",
            "--no-fuzzy",
        ],
    )

    prepare_dataset.main()

    splits = {
        name: pd.read_csv(output_dir / f"{name}.csv")
        for name in ("train", "validation", "test")
    }
    for split in splits.values():
        assert set(split["label"]) == {0, 1}
        assert set(zip(split["label"], split["label_name"])) <= {
            (0, "insult"),
            (1, "defamation"),
        }
        assert "normal" not in set(split["label_name"])
    assert (splits["train"]["source"] == "Synthetic").sum() == 2
    assert (splits["train"].loc[splits["train"]["source"] == "Synthetic", "label"] == 1).all()
    assert not (splits["validation"]["source"] == "Synthetic").any()
    assert not (splits["test"]["source"] == "Synthetic").any()
    assert (output_dir / "data_report.md").is_file()
