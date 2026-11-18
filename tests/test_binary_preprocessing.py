from src.thai_mod_api.binary_preprocessing import (
    PREPROCESSING_VERSION,
    normalize_binary_text,
)


def test_versioned_preprocessing_handles_representative_evasion():
    assert PREPROCESSING_VERSION == "binary_preprocess_v1"
    assert normalize_binary_text("ค-ว-ย") == "ควย"
    assert normalize_binary_text("kuy") == "ควย"


def test_versioned_preprocessing_preserves_ordinary_text_and_ascii_words():
    assert normalize_binary_text("สวัสดีครับ") == "สวัสดีครับ"
    assert normalize_binary_text("focus") == "focus"


def test_versioned_preprocessing_handles_null_empty_and_urls():
    assert normalize_binary_text(None) == ""
    assert normalize_binary_text("") == ""
    assert "example.com" not in normalize_binary_text("ดูข้อมูล https://example.com")
