from __future__ import annotations

import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import ComplementNB

from scripts import train_complement_nb_3class as trainer


class ArtifactProbe:
    def predict(self, texts):
        return [1] * len(texts)


@pytest.fixture
def train_frame() -> pd.DataFrame:
    labels = [0, 0, 1, 1, 2, 2]
    return pd.DataFrame(
        {
            "text_raw": [f"raw-{index}" for index in range(len(labels))],
            "text_norm": [f"normalized-{index}" for index in range(len(labels))],
            "label": labels,
            "label_name": [trainer.LABEL_NAMES[label] for label in labels],
            "source": ["fixture"] * len(labels),
        }
    )


@pytest.fixture
def validation_frame() -> pd.DataFrame:
    labels = [0, 1, 2]
    return pd.DataFrame(
        {
            "text_raw": ["raw-0", "raw-1", "raw-2"],
            "text_norm": ["validation-0", "validation-1", "validation-2"],
            "label": labels,
            "label_name": [trainer.LABEL_NAMES[label] for label in labels],
            "source": ["fixture"] * len(labels),
        }
    )


def test_label_mapping_and_split_validation(train_frame):
    assert trainer.LABEL_NAMES == {0: "normal", 1: "insult", 2: "defamation"}
    validated = trainer.validate_split(train_frame, "train")
    assert validated["label"].tolist() == train_frame["label"].tolist()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("text_norm", None),
        ("text_norm", "  "),
        ("text_raw", None),
        ("label", None),
        ("label_name", None),
        ("source", "  "),
    ],
)
def test_split_validation_rejects_missing_or_empty_values(train_frame, column, value):
    invalid = train_frame.copy()
    invalid.loc[0, column] = value

    with pytest.raises(ValueError, match=column):
        trainer.validate_split(invalid, "train")


@pytest.mark.parametrize("value", [3, 1.5, "not-a-label"])
def test_split_validation_rejects_invalid_labels(train_frame, value):
    invalid = train_frame.copy()
    invalid["label"] = invalid["label"].astype(object)
    invalid.loc[0, "label"] = value

    with pytest.raises(ValueError, match="label"):
        trainer.validate_split(invalid, "train")


def test_split_validation_rejects_missing_required_column(train_frame):
    invalid = train_frame.drop(columns=["source"])

    with pytest.raises(ValueError, match="source"):
        trainer.validate_split(invalid, "train")


def test_pipeline_uses_complement_nb_and_expected_vectorizer():
    model = trainer.make_model(alpha=0.3)

    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["clf"], ComplementNB)
    vectorizer = model.named_steps["vect"]
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.min_df == 2
    assert vectorizer.max_features == 50_000
    assert vectorizer.lowercase is False


def test_fit_model_uses_only_train_and_balanced_train_weights(monkeypatch, train_frame):
    seen = {}

    class FitSpy:
        def fit(self, texts, labels, **fit_params):
            seen["texts"] = texts
            seen["labels"] = labels
            seen["weights"] = fit_params["clf__sample_weight"]
            return self

    spy = FitSpy()
    monkeypatch.setattr(trainer, "make_model", lambda alpha: spy)

    fitted = trainer.fit_model(train_frame, alpha=0.3)

    assert fitted is spy
    assert seen["texts"] == train_frame["text_norm"].tolist()
    assert seen["labels"] == train_frame["label"].tolist()
    expected = trainer.compute_sample_weight(
        class_weight="balanced",
        y=train_frame["label"].to_numpy(),
    )
    np.testing.assert_allclose(seen["weights"], expected)


def test_select_best_configuration_uses_validation_macro_f1(train_frame, validation_frame):
    class PredictByAlpha:
        def __init__(self, alpha):
            self.alpha = alpha

        def fit(self, texts, labels, **fit_params):
            return self

        def predict(self, texts):
            predictions = {
                0.1: [0, 0, 0],
                0.3: [0, 1, 2],
                0.5: [0, 1, 1],
            }
            return np.array(predictions[self.alpha][: len(texts)])

    selection = trainer.select_best_configuration(
        train_frame,
        validation_frame,
        alphas=(0.1, 0.3, 0.5),
        model_factory=PredictByAlpha,
    )

    assert selection["alpha"] == 0.3
    assert selection["validation_metrics"]["macro_f1"] == 1.0
    assert [candidate["alpha"] for candidate in selection["candidate_metrics"]] == [
        0.1,
        0.3,
        0.5,
    ]


def test_metrics_include_all_classes_even_when_classes_are_never_predicted():
    metrics = trainer.calculate_metrics(
        y_true=[0, 1, 2],
        y_pred=[0, 0, 0],
    )

    assert metrics["labels"] == [0, 1, 2]
    assert metrics["per_class"]["1"]["support"] == 1
    assert metrics["per_class"]["2"]["support"] == 1
    assert metrics["per_class"]["1"]["f1"] == 0.0
    assert metrics["per_class"]["2"]["recall"] == 0.0
    assert len(metrics["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in metrics["confusion_matrix"])


def test_save_artifacts_refuses_existing_output_directory(tmp_path):
    output_dir = tmp_path / "candidate"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        trainer.save_artifacts(output_dir, ArtifactProbe(), {}, {})

    assert sentinel.read_text(encoding="utf-8") == "existing"


def test_saved_artifact_can_be_loaded_and_used(tmp_path):
    output_dir = tmp_path / "candidate"
    metadata = {"label_mapping": {"0": "normal", "1": "insult", "2": "defamation"}}
    metrics = {"validation": {"macro_f1": 0.5}}

    trainer.save_artifacts(output_dir, ArtifactProbe(), metadata, metrics)

    loaded_model = joblib.load(output_dir / "model.joblib")
    assert loaded_model.predict(["one", "two"]) == [1, 1]
    assert json.loads((output_dir / "metadata.json").read_text(encoding="utf-8")) == metadata
    assert json.loads((output_dir / "metrics.json").read_text(encoding="utf-8")) == metrics


def test_cli_help_runs_without_starting_training():
    script = trainer.ROOT / "scripts" / "train_complement_nb_3class.py"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=trainer.ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
