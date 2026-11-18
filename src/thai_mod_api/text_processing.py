from __future__ import annotations

import re
from typing import Any

import emoji


def _is_na(val: Any) -> bool:
    if val is None:
        return True
    try:
        import pandas as pd

        return bool(pd.isna(val))
    except ImportError:
        return isinstance(val, float) and val != val


def preprocess_text(text: str) -> str:
    if _is_na(text):
        return ""

    cleaned = str(text)
    cleaned = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        cleaned,
    )
    cleaned = emoji.demojize(cleaned, language="en").strip()

    lowered = []
    for char in cleaned:
        lowered.append(char.lower() if ord(char) < 128 else char)
    return "".join(lowered)


# ---------------------------------------------------------------------------
# Cyberbullying (Insult / Defamation) preprocessing
# ---------------------------------------------------------------------------
_MENTION_RE = re.compile(r"@[\w\.]+")
_HASHTAG_RE = re.compile(r"#(\S+)")
_PIC_RE = re.compile(r"pic\.twitter\.com/\S+|twitter\.com/\S+")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Basic cleaning shared by every model.

    URL / pic links removed, @mentions -> <USER>, '#' stripped from hashtags,
    emoji -> :name:, ASCII lowercased, whitespace collapsed.
    """
    if text is None or (not isinstance(text, str) and _is_na(text)):
        return ""
    cleaned = _PIC_RE.sub(" ", str(text))
    cleaned = preprocess_text(cleaned)  # URLs, emoji, ASCII lowercase
    cleaned = _MENTION_RE.sub(" <USER> ", cleaned)
    cleaned = _HASHTAG_RE.sub(r"\1", cleaned)
    return _WS_RE.sub(" ", cleaned).strip()


def preprocess(text: str, use_fuzzy: bool = True) -> str:
    """Full pipeline used for training and inference: clean + evasion normalize.

    Feed the result to WangchanBERTa / PhayaThaiBERT as-is, or to ``tokenize``
    for Naive Bayes.
    """
    from .evasion_normalizer import normalize_evasion

    return _WS_RE.sub(" ", normalize_evasion(clean_text(text), use_fuzzy=use_fuzzy)).strip()


def tokenize(text: str) -> list[str]:
    """PyThaiNLP newmm tokens without whitespace tokens (for BoW / TF-IDF)."""
    from pythainlp.tokenize import word_tokenize

    return [t for t in word_tokenize(text, engine="newmm", keep_whitespace=False) if t.strip()]
