"""Checks on the processed dataset written by scripts/prepare_dataset.py.

Skipped when datasets/processed/ has not been generated yet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path(__file__).resolve().parents[1] / "datasets" / "processed"
SPLITS = ("train", "validation", "test")

pytestmark = pytest.mark.skipif(
    not all((PROCESSED / f"{s}.csv").exists() for s in SPLITS),
    reason="run scripts/prepare_dataset.py first",
)


@pytest.fixture(scope="module")
def splits() -> dict[str, pd.DataFrame]:
    return {s: pd.read_csv(PROCESSED / f"{s}.csv") for s in SPLITS}


def test_columns(splits):
    for df in splits.values():
        assert list(df.columns) == ["text_raw", "text_norm", "label", "label_name", "source"]


def test_labels_valid(splits):
    names = {0: "normal", 1: "insult", 2: "defamation"}
    for df in splits.values():
        assert set(df["label"]) <= {0, 1, 2}
        assert (df["label"].map(names) == df["label_name"]).all()


def test_every_split_has_every_class(splits):
    for name, df in splits.items():
        assert set(df["label"]) == {0, 1, 2}, name


def test_no_leakage_between_splits(splits):
    train = set(splits["train"]["text_norm"])
    val = set(splits["validation"]["text_norm"])
    test = set(splits["test"]["text_norm"])
    assert not train & val
    assert not train & test
    assert not val & test


def test_no_duplicates_within_split(splits):
    for df in splits.values():
        assert not df["text_norm"].duplicated().any()


def test_synthetic_only_in_train(splits):
    assert (splits["train"]["source"] == "Synthetic").any()
    assert not (splits["validation"]["source"] == "Synthetic").any()
    assert not (splits["test"]["source"] == "Synthetic").any()


def test_no_adult_spam(splits):
    for df in splits.values():
        assert not df["text_raw"].str.contains("หนังโป๊|คลิปโป๊|หนังเอวี").any()
