"""Versioned preprocessing contract for the Insult/Defamation experiment.

The raw message is always retained. This v1 path deliberately disables fuzzy
keyword correction; only the existing deterministic cleaning and evasion rules
are applied so normal words are less likely to be rewritten.
"""

from __future__ import annotations

from typing import Any

from .evasion_normalizer import normalize_evasion
from .text_processing import clean_text

PREPROCESSING_VERSION = "binary_preprocess_v1"
USE_FUZZY = False


def normalize_binary_text(text: Any) -> str:
    """Normalize one raw message using the deterministic binary-experiment rules."""
    cleaned = clean_text(text)
    return normalize_evasion(cleaned, use_fuzzy=USE_FUZZY).strip()


def preprocessing_metadata() -> dict[str, object]:
    return {
        "version": PREPROCESSING_VERSION,
        "function": "src.thai_mod_api.binary_preprocessing.normalize_binary_text",
        "steps": [
            "remove Twitter links and URLs",
            "demojize emoji, lowercase ASCII, replace @mentions with <USER>",
            "strip hashtag marker and collapse whitespace",
            "join separated evasion spellings",
            "collapse repeated characters",
            "map configured romanized slurs and exact Thai euphemism tokens",
            "fuzzy correction disabled",
        ],
        "preserves_raw_text": True,
        "fuzzy_matching": False,
        "limitations": [
            "rule dictionaries do not cover every evasion spelling",
            "deterministic rules may still alter emphasis or ambiguous tokens",
            "normalization does not establish whether a message is harmful",
            "there is no raw-to-normalized character offset alignment",
        ],
    }
