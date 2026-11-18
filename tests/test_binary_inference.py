import json

import pytest

from src.thai_mod_api.binary_inference import BinaryPredictionService, analyze_binary_context


def test_broad_collective_pronoun_is_not_specific_target():
    result = analyze_binary_context("พวกมึงโกงเงิน")
    assert result["target"]["status"] == "Unknown"
    assert result["in_scope"] is False


def test_explicit_mention_and_allegation_are_rule_evidence():
    result = analyze_binary_context("@alice โกงเงินบริษัท")
    assert result["target"]["status"] == "Identified"
    assert result["allegation_analysis"]["result"] == "Factual allegation cue"
    assert any(item["kind"] == "target_rule" for item in result["evidence_spans"])
    assert any(item["kind"] == "allegation_rule" for item in result["evidence_spans"])


def test_normal_or_ambiguous_input_is_out_of_scope():
    result = analyze_binary_context("วันนี้อากาศดี")
    assert result["in_scope"] is False
    assert result["allegation_analysis"]["result"] == "Unclear"


class _FixedModel:
    classes_ = [0, 1]

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, _texts):
        return [self.probabilities]


def test_binary_output_mapping_and_abstention(tmp_path):
    metadata = {
        "model_type": "complement_nb",
        "preprocessing_version": "binary_preprocess_v1",
        "label_mapping": {"0": "Insult", "1": "Defamation"},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    service = BinaryPredictionService(tmp_path, model=_FixedModel([0.1, 0.9]))
    prediction = service.predict("@alice โกงเงิน")
    assert prediction["status"] == "predicted"
    assert prediction["predicted_label"] == "Defamation"
    assert prediction["confidence"] == 0.9
    assert "not legal advice" in prediction["disclaimer"].lower()
    out_of_scope = service.predict("วันนี้อากาศดี")
    assert out_of_scope["status"] == "abstained"
    assert out_of_scope["predicted_label"] is None
    assert service.predict("")["reason"] == "empty_input"


def test_model_loading_failure_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):
        BinaryPredictionService.load(tmp_path)
