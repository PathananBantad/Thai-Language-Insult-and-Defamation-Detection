"""Tests for target identification and allegation analysis."""

from __future__ import annotations

import pytest

from src.thai_mod_api.allegation_analyzer import analyze, analyze_allegation
from src.thai_mod_api.target_extractor import extract_targets


def _types(text):
    return [(t["text"], t["type"]) for t in extract_targets(text)]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("@somchai โกงเงิน", ("@somchai", "mention")),
        ("ครูวิชัยแอบมีเมียน้อย", ("ครูวิชัย", "named")),
        ("ได้ยินว่าพี่ตั้มติดยา", ("พี่ตั้ม", "named")),
        ("ร้านนี้ขายของปลอม", ("ร้านนี้", "reference")),
        ("มึงขโมยของกู", ("มึง", "addressee")),
        ("อิตุ๊ดนี่โกงกู", ("อิตุ๊ด", "slur")),
    ],
)
def test_extract_targets(text, expected):
    assert expected in _types(text)


def test_slur_after_prefix_is_not_a_name():
    assert not any(t["type"] == "named" for t in extract_targets("อีควาย ไปตายซะ"))


def test_target_spans_match_text():
    text = "บอกไว้ก่อน @user_nok โกงเงิน"
    for t in extract_targets(text):
        assert text[t["start"]: t["end"]] == t["text"]


@pytest.mark.parametrize(
    "text, kind",
    [
        ("@x โกงเงินบริษัท", "factual_claim"),
        ("ครูคนนี้มีเมียน้อย", "factual_claim"),
        ("มึงมันขี้โกง", "insult_only"),  # character judgement, not an act
        ("อีตุ๊ด ตอแหล", "insult_only"),
        ("วันนี้ร้อนมาก", None),
    ],
)
def test_allegation_type(text, kind):
    assert analyze_allegation(text)["type"] == kind


def test_negated_allegation_is_ignored():
    assert not analyze_allegation("กูไม่ได้โกง")["detected"]


def test_hedged_claim_still_detected():
    r = analyze_allegation("ได้ยินมาว่า X ติดยาจริงป่ะ")
    assert r["detected"] and r["hedged"]


def test_disguised_allegation_detected():
    assert analyze_allegation("@x โ-ก-ง เงิน")["detected"]


def test_analyze_defamation_risk():
    assert analyze("@somchai โกงเงินบริษัทไปสองล้าน")["defamation_risk"]
    assert not analyze("มึงมันควาย")["defamation_risk"]


def test_analyze_explanation_is_thai_text():
    r = analyze("ครูวิชัยแอบมีเมียน้อย")
    assert "ครูวิชัย" in r["explanation"] and "เมียน้อย" in r["explanation"]
