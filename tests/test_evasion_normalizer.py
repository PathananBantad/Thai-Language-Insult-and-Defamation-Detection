"""Tests for evasion normalization and the cyberbullying preprocess() pipeline."""

from __future__ import annotations

import math

import pytest

from src.thai_mod_api.evasion_normalizer import (
    collapse_repeats,
    find_evasions,
    join_spaced_letters,
    map_karaoke,
    normalize_evasion,
)
from src.thai_mod_api.text_processing import clean_text, preprocess, tokenize


# ---------------------------------------------------------------------------
# Evasion rules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("ค-ว-ย", "ควย"),
        ("ค.ว.ย", "ควย"),
        ("ค ว ย", "ควย"),
        ("ค*ว*ย", "ควย"),
        ("มึงมันค-ว-ย", "มึงมันควย"),
        ("k-u-y", "kuy"),
    ],
)
def test_join_spaced_letters(raw, expected):
    assert join_spaced_letters(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("ควายยยย", "ควาย"),  # final char belongs to the word -> keep one
        ("โง่ววววว", "โง่"),  # final char is pure elongation -> drop
        ("ดีมากกกก", "ดีมาก"),
        ("5555555", "555"),
        ("!!!!!", "!"),
    ],
)
def test_collapse_repeats(raw, expected):
    assert collapse_repeats(raw) == expected


@pytest.mark.parametrize("raw, expected", [("kuy", "ควย"), ("KUY", "ควย"), ("sus", "สัส"), ("hia", "เหี้ย")])
def test_map_karaoke(raw, expected):
    assert map_karaoke(raw) == expected


@pytest.mark.parametrize("text", ["focus", "AI model", "Kuyper"])
def test_karaoke_leaves_english_alone(text):
    assert map_karaoke(text) == text


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("kuy ไรเนี่ย", "ควย ไรเนี่ย"),
        ("เชี่ยมาก", "เหี้ยมาก"),
        ("อิสัส", "อีสัส"),
        ("อีควัย", "อีควาย"),
        ("มรึงอะ", "มึงอะ"),
        ("แมร่งเอ้ย", "แม่งเอ้ย"),
        ("เหี่ยยยย", "เหี้ย"),
        ("กุไม่รู้", "กูไม่รู้"),
    ],
)
def test_normalize_evasion(raw, expected):
    assert normalize_evasion(raw) == expected


@pytest.mark.parametrize(
    "text",
    ["ความรู้สึก", "เชี่ยวชาญมาก", "อิทธิพล", "แม่มด", "สาดน้ำ", "กุ้งเผา", "ปี 2-5-6-7", "สวัสดีครับ"],
)
def test_normal_words_are_not_changed(text):
    """Over-normalization would turn innocent words into slurs."""
    assert normalize_evasion(text) == text


def test_normalize_handles_empty_and_non_string():
    assert normalize_evasion("") == ""
    assert normalize_evasion(None) == ""  # type: ignore[arg-type]


def test_find_evasions_reports_changes():
    assert find_evasions("มึงมันค-ว-ย kuy สวัสดี") == [("มึงมันค-ว-ย", "มึงมันควย"), ("kuy", "ควย")]


# ---------------------------------------------------------------------------
# preprocess() pipeline
# ---------------------------------------------------------------------------
def test_clean_text_mentions_urls_hashtags():
    out = clean_text("@somchai_99 ดูนี่ https://x.com/a #ซีเกมส์2017 pic.twitter.com/abc")
    assert out == "<USER> ดูนี่ ซีเกมส์2017"


def test_preprocess_full_pipeline():
    assert preprocess("@a มึงมันค-ว-ย KUY") == "<USER> มึงมันควย ควย"


@pytest.mark.parametrize("value", [None, float("nan"), ""])
def test_preprocess_missing(value):
    assert preprocess(value) == ""


def test_tokenize_no_whitespace_tokens():
    toks = tokenize("<USER> มึง มันควาย")
    assert toks and all(t.strip() for t in toks)
    assert "ควาย" in toks
