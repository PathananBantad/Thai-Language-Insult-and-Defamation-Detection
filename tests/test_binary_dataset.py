import pandas as pd

import pytest

from scripts.prepare_binary_dataset import (
    derive_split,
    map_binary_label,
    stable_sample_id,
    validate_unique_sample_ids,
)


def _mini_split():
    return pd.DataFrame(
        [
            {"text_raw": "ปกติ", "text_norm": "ปกติ", "label": 0, "label_name": "normal", "source": "source-a"},
            {"text_raw": "@bob ค-ว-ย", "text_norm": "legacy norm", "label": 1, "label_name": "insult", "source": "source-b"},
            {"text_raw": "@alice โกงเงิน", "text_norm": "legacy norm", "label": 2, "label_name": "defamation", "source": "Synthetic"},
        ]
    )


def test_binary_label_mapping_and_normal_exclusion():
    assert map_binary_label(0) is None
    assert map_binary_label(1) == 0
    assert map_binary_label(2) == 1


def test_stable_sample_id_uses_physical_csv_row():
    assert stable_sample_id("train", 2) == "train-000002"


def test_derive_split_excludes_normal_and_only_train_exclusion_ids():
    frame = _mini_split()
    excluded = {"train-000003": {}}
    train = derive_split("train", frame, excluded)
    validation = derive_split("validation", frame, excluded)
    assert [row["sample_id"] for row in train] == ["train-000004"]
    assert train[0]["label"] == 1
    assert train[0]["original_label"] == 2
    assert train[0]["is_synthetic"] == "true"
    assert len(validation) == 2
    assert {row["label"] for row in validation} == {0, 1}


def test_derive_split_excludes_requested_sources():
    rows = derive_split("validation", _mini_split(), {}, ("source-b",))
    assert len(rows) == 1
    assert rows[0]["source"] == "Synthetic"
    assert rows[0]["label"] == 1


def test_duplicate_sample_id_detection():
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        validate_unique_sample_ids([{"sample_id": "train-000002"}, {"sample_id": "train-000002"}], "test")
