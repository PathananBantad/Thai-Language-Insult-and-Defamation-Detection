"""Evasion normalization for Thai cyberbullying text.

Turns deliberately disguised spellings back into their canonical form so that
the classifiers see the same token no matter how the word was typed.

Rules, applied in this order:
  1. Remove separators between single letters      ค-ว-ย / ค.ว.ย / ค ว ย -> ควย
  2. Collapse characters repeated 3+ times         ควายยยย -> ควาย, 5555555 -> 555
  3. Map romanised (karaoke) slurs to Thai          kuy -> ควย, sus -> สัส
  4. Map known Thai euphemism tokens               เชี่ย -> เหี้ย, มรึง -> มึง
  5. Fuzzy match (edit distance 1) against the toxic keyword list, only for
     tokens that are NOT real Thai words           เหี่ย -> เหี้ย, but ความ stays ความ

Use ``normalize_evasion(text)`` for the normalized string and
``find_evasions(text)`` to see which spans were changed (useful for the report
and for explainability/highlighting).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:
    class Levenshtein:  # type: ignore
        @staticmethod
        def distance(s1: str, s2: str, score_cutoff: int | None = None) -> int:
            if s1 == s2:
                return 0
            if len(s1) < len(s2):
                s1, s2 = s2, s1
            if not s2:
                return len(s1)
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                if score_cutoff is not None and min(current_row) > score_cutoff:
                    return score_cutoff + 1
                previous_row = current_row
            return previous_row[-1]

# ---------------------------------------------------------------------------
# Character classes
# ---------------------------------------------------------------------------
_TH_CONS = "ก-ฮ"  # ก-ฮ
_TH_MARK = "ะ-ฺเ-๎"  # vowels, tone marks, ๆ etc.
_SEP = r"[\-\._\*\s/|+~,]+"

# A "letter unit" is one Thai consonant plus optional marks, or one Latin letter.
# Any single Thai letter (consonant, vowel or mark) plus trailing marks, so
# spelled-out vowels also join: ค-ว-า-ย, ข-โ-ม-ย, ส ั ส.
_TH_UNIT = rf"[{_TH_CONS}{_TH_MARK}][{_TH_MARK}]*"
_LAT_UNIT = r"[a-zA-Z]"

# 3+ single letter units separated by separators: ค-ว-ย, ค ว ย, k.u.y
# Punctuation separators (ค-ว-ย) are unambiguous and may sit inside a word
# (มึงมันค-ว-ย). Space separators need word boundaries on both sides.
_PUNCT_SEP = r"[\-\._\*/|+~,]+"
_SPACED_TH_PUNCT = re.compile(rf"(?:{_TH_UNIT}{_PUNCT_SEP}){{2,}}{_TH_UNIT}(?![{_TH_MARK}])")
_SPACED_TH = re.compile(rf"(?<![{_TH_CONS}{_TH_MARK}])(?:{_TH_UNIT}{_SEP}){{2,}}{_TH_UNIT}(?![{_TH_CONS}{_TH_MARK}])")
_SPACED_LAT = re.compile(rf"(?<![a-zA-Z])(?:{_LAT_UNIT}{_SEP}){{2,}}{_LAT_UNIT}(?![a-zA-Z])")
_SEP_RE = re.compile(_SEP)

_REPEAT_DIGIT = re.compile(r"(\d)\1{3,}")  # 55555 -> 555
_REPEAT_CHAR = re.compile(r"([^\d\s])\1{2,}")  # ยยย -> ย, !!! -> !

# ---------------------------------------------------------------------------
# Dictionaries
# ---------------------------------------------------------------------------
# Romanised / karaoke spellings (matched as whole Latin words, case-insensitive).
# Short ambiguous English words (ai, ee, ku, sat, dok) are deliberately left out.
KARAOKE_MAP: dict[str, str] = {
    "kuy": "ควย", "kuay": "ควย", "kvy": "ควย", "kwy": "ควย", "khuy": "ควย",
    "sus": "สัส", "sas": "สัส",
    "hee": "หี", "hia": "เหี้ย", "heer": "เหี้ย",
    "kwai": "ควาย", "kway": "ควาย",
    "mueng": "มึง", "meung": "มึง", "maeng": "แม่ง",
    "torlae": "ตอแหล",
}

# Thai euphemism spellings -> canonical keyword. Matched against whole newmm
# tokens, so real words that merely contain them (เชี่ยวชาญ, แม่มด, สาดน้ำ)
# are not touched.
EUPHEMISM_MAP: dict[str, str] = {
    "เชี่ย": "เหี้ย", "เชี้ย": "เหี้ย", "เหี่ย": "เหี้ย", "เฮี้ย": "เหี้ย", "เหี้ยะ": "เหี้ย",
    "ควัย": "ควาย", "ฟาย": "ควาย",
    "สัด": "สัส", "สาส": "สัส",
    "มรึง": "มึง", "เมิง": "มึง", "มริง": "มึง",
    "กุ": "กู", "ตรู": "กู",
    "แม่ม": "แม่ง", "แมร่ง": "แม่ง", "เเม่ง": "แม่ง", "แม่งง": "แม่ง",
    "อิ": "อี",
    "ตุ้ด": "ตุ๊ด",
    "กระหรี่": "กะหรี่",
    "ส้นตรีน": "ส้นตีน",
}

_SAFE_SUBSTRINGS = ("ควัย", "มรึง", "แมร่ง", "เชี้ย", "เหี่ย", "เฮี้ย", "ส้นตรีน", "เเม่ง", "เหี้ยะ", "กระหรี่")
_SAFE_SUBSTR_RE = re.compile(
    "|".join(sorted(map(re.escape, _SAFE_SUBSTRINGS), key=len, reverse=True)) + r"|แม่ม(?!ด|[ะ-ฺ็-๎])"
)
# อิ used as a derogatory prefix (อิดอก, อิสัด) -> อี, only before a slur so that
# words like อิทธิพล / อินเทอร์เน็ต are untouched.
_I_PREFIX_TARGETS = (
    "ดอก", "สัส", "สัด", "หอก", "ห่า", "เหี้ย", "เชี่ย", "ควาย", "ควัย", "บ้า", "หมา", "เวร", "ตุ๊ด", "เกย์",
    "ผี", "เปรต", "ชั่ว", "ร่าน", "แรด", "กะหรี่", "ปลวก", "ส้นตีน", "ตอแหล", "หอย", "เลว", "โง่",
)
_I_PREFIX_RE = re.compile("อิ(?=" + "|".join(map(re.escape, _I_PREFIX_TARGETS)) + ")")

_KARAOKE_RE = re.compile(
    r"(?<![a-zA-Z])(" + "|".join(sorted(map(re.escape, KARAOKE_MAP), key=len, reverse=True)) + r")(?![a-zA-Z])",
    re.IGNORECASE,
)

_DEFAULT_KEYWORDS = Path(__file__).resolve().parents[2] / "datasets" / "toxic_keywords.csv"


@lru_cache(maxsize=1)
def _keywords() -> tuple[str, ...]:
    words: set[str] = set(EUPHEMISM_MAP.values()) | set(KARAOKE_MAP.values())
    if _DEFAULT_KEYWORDS.exists():
        import csv

        with _DEFAULT_KEYWORDS.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                kw = (row.get("keyword") or "").strip()
                if kw:
                    words.add(kw)
    # Fuzzy matching is only reliable on longer words.
    return tuple(sorted(w for w in words if len(w) >= 4))


@lru_cache(maxsize=1)
def _thai_dictionary() -> frozenset[str]:
    from pythainlp.corpus import thai_words

    return frozenset(thai_words())


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------
def join_spaced_letters(text: str) -> str:
    """ค-ว-ย / ค ว ย / k.u.y -> ควย / kuy"""
    text = _SPACED_TH_PUNCT.sub(lambda m: _SEP_RE.sub("", m.group(0)), text)
    text = _SPACED_TH.sub(lambda m: _SEP_RE.sub("", m.group(0)), text)
    return _SPACED_LAT.sub(lambda m: _SEP_RE.sub("", m.group(0)), text)


_TH_WORD_BEFORE = re.compile(rf"[{_TH_CONS}{_TH_MARK}]{{1,8}}$")


def _drop_or_keep_one(text: str, m: re.Match) -> str:
    """For a run like ววววว decide between keeping one char or none.

    ควายยยย -> ควาย (ย belongs to the word), โง่ววววว -> โง่ (ว is pure
    elongation). Pick whichever leaves a dictionary word right before the run.
    """
    ch = m.group(1)
    before = _TH_WORD_BEFORE.search(text[: m.start()])
    if ch == "ๆ" or not before or not ("ก" <= ch <= "๎"):
        return ch
    stem = before.group(0)
    dictionary = _thai_dictionary()
    for i in range(len(stem)):
        tail = stem[i:]
        if tail + ch in dictionary:
            return ch
        if tail in dictionary and len(tail) >= 2:
            return ""
    return ch


def collapse_repeats(text: str) -> str:
    """ควายยยย -> ควาย, โง่ววววว -> โง่, 5555555 -> 555, !!!!! -> !"""
    text = _REPEAT_DIGIT.sub(lambda m: m.group(1) * 3, text)
    return _REPEAT_CHAR.sub(lambda m: _drop_or_keep_one(text, m), text)


def map_karaoke(text: str) -> str:
    """kuy -> ควย (whole Latin words only, so 'focus' or 'kuyper' are untouched)."""
    return _KARAOKE_RE.sub(lambda m: KARAOKE_MAP[m.group(1).lower()], text)


def _fuzzy_match(tok: str, max_distance: int) -> str | None:
    for kw in _keywords():
        if abs(len(kw) - len(tok)) > max_distance:
            continue
        if Levenshtein.distance(tok, kw, score_cutoff=max_distance) <= max_distance:
            return kw
    return None


def map_thai_tokens(text: str, use_fuzzy: bool = True, max_distance: int = 1) -> str:
    """Token-level Thai rules: euphemism dictionary, then fuzzy keyword match.

    * เชี่ย -> เหี้ย, มรึง -> มึง, อิ(สัส) -> อี(สัส)   (exact token lookup)
    * เหี่ยย -> เหี้ย  (edit distance <= 1 to a toxic keyword; only for tokens of
      4+ characters that are NOT valid Thai words, so ความ/ควาญ stay as they are)
    """
    from pythainlp.tokenize import word_tokenize

    # Spellings that never occur inside a real word can be replaced as
    # substrings, which also catches them in unspaced text (อีควัย, มรึงอะ).
    text = _SAFE_SUBSTR_RE.sub(lambda m: EUPHEMISM_MAP[m.group(0)], text)
    text = _I_PREFIX_RE.sub("อี", text)

    dictionary = _thai_dictionary()
    out: list[str] = []
    for tok in word_tokenize(text, engine="newmm", keep_whitespace=True):
        if tok in EUPHEMISM_MAP:
            out.append(EUPHEMISM_MAP[tok])
        elif use_fuzzy and len(tok) >= 4 and not tok.isascii() and tok not in dictionary:
            out.append(_fuzzy_match(tok, max_distance) or tok)
        else:
            out.append(tok)
    return "".join(out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def normalize_evasion(text: str, use_fuzzy: bool = True) -> str:
    """Apply all evasion rules. Safe on empty / non-string input."""
    if not isinstance(text, str) or not text:
        return ""
    text = join_spaced_letters(text)
    text = collapse_repeats(text)
    text = map_karaoke(text)
    return map_thai_tokens(text, use_fuzzy=use_fuzzy)


def find_evasions(text: str) -> list[tuple[str, str]]:
    """Return (original_token, normalized_token) pairs that changed.

    Works on whitespace-separated chunks, which is enough for reporting which
    parts of a message were disguised.
    """
    if not isinstance(text, str):
        return []
    changes = []
    for chunk in text.split():
        norm = normalize_evasion(chunk)
        if norm != chunk:
            changes.append((chunk, norm))
    return changes


def find_evasion_spans(text: str) -> list[dict[str, Any]]:
    """Find character spans in raw text where evasion normalization applies."""
    if not isinstance(text, str) or not text:
        return []
    spans: list[dict[str, Any]] = []

    # 1. Spaced letters (Thai and Latin)
    for pattern in (_SPACED_TH_PUNCT, _SPACED_TH, _SPACED_LAT):
        for m in pattern.finditer(text):
            raw = m.group(0)
            norm = _SEP_RE.sub("", raw)
            if norm != raw:
                spans.append({"raw": raw, "normalized": norm, "start": m.start(), "end": m.end(), "type": "spaced_letters"})

    # 2. Repeated characters
    for m in _REPEAT_CHAR.finditer(text):
        raw = m.group(0)
        norm = _drop_or_keep_one(text, m)
        if norm != raw:
            spans.append({"raw": raw, "normalized": norm, "start": m.start(), "end": m.end(), "type": "character_elongation"})

    # 3. Karaoke slurs
    for m in _KARAOKE_RE.finditer(text):
        raw = m.group(0)
        norm = KARAOKE_MAP.get(raw.lower(), raw)
        if norm != raw:
            spans.append({"raw": raw, "normalized": norm, "start": m.start(), "end": m.end(), "type": "karaoke_slur"})

    # 4. Safe substring euphemisms
    for m in _SAFE_SUBSTR_RE.finditer(text):
        raw = m.group(0)
        norm = EUPHEMISM_MAP.get(raw, raw)
        if norm != raw:
            spans.append({"raw": raw, "normalized": norm, "start": m.start(), "end": m.end(), "type": "euphemism"})

    # Sort by start position and deduplicate overlapping
    spans.sort(key=lambda s: s["start"])
    deduped: list[dict[str, Any]] = []
    for s in spans:
        if not any(not (s["end"] <= d["start"] or s["start"] >= d["end"]) for d in deduped):
            deduped.append(s)
    return deduped
