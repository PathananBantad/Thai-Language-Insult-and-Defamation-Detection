"""Target identification: who is the message about?

Rule-based, so it runs without downloading an NER model. Finds:
  * mention     @username
  * named       honorific/prefix + name: นายสมชาย, ครูวิชัย, อีเจน, ไอ้โจ้, พี่ตั้ม
  * reference   pointing phrases: คนนี้, ร้านนี้, บริษัทนี้, เพจนี้, หมอคนนี้
  * addressee   second-person pronouns: มึง, แก, เธอ, คุณ, เมิง, ท่าน
  * slur        derogatory address used as a nickname for one person: อิตุ๊ด, ไอ้เด็กเหี้ย
  * third       third-person pronouns pointing at someone: มัน, แม่ง, เขา, เค้า

Spans refer to the text passed in, so call it on the RAW text when you want to
highlight the original message.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_TH = "ก-๎"

# Prefixes that are followed by a person's (nick)name
_NAME_PREFIXES = (
    "นางสาว", "นาย", "นาง", "คุณ", "ครู", "อาจารย์", "อจ.", "หมอ", "ผอ.", "ผู้ใหญ่", "กำนัน", "ลุง", "ป้า",
    "น้า", "พี่", "น้อง", "ไอ้", "อี", "อิ", "เจ๊", "เฮีย", "ส.ส.", "นายก",
)
# Words that are NOT names even when they follow a prefix (slurs, identity
# terms, generic nouns). "อีตุ๊ด" / "ไอ้ควาย" are insults, not identified people.
_NOT_NAMES = {
    "ตุ๊ด", "เกย์", "กะเทย", "กระเทย", "ควาย", "ควัย", "สัส", "สัด", "ดอก", "หอก", "ห่า", "เหี้ย", "เชี่ย", "บ้า",
    "หมา", "เวร", "ผี", "เปรต", "ชั่ว", "ร่าน", "แรด", "กะหรี่", "ปลวก", "ส้นตีน", "ตอแหล", "หอย", "เลว", "โง่",
    "สัตว์", "ฟาย", "เด็ก", "แก่", "อ้วน", "คน", "พวก", "นี่", "นั่น", "นั้น", "นี้", "ที่", "ไร", "อะไร", "ก็",
    "จะ", "ไม่", "ได้", "มัน", "มา", "ไป", "ห่าน", "ขี้", "หน้า", "ตัว", "ผัว", "เมีย",
}
_MENTION_RE = re.compile(r"@[\w\.]+")
_NAMED_RE = re.compile(
    "(" + "|".join(sorted(map(re.escape, _NAME_PREFIXES), key=len, reverse=True)) + rf")\s?([{_TH}a-zA-Z]{{2,15}})"
)
_REFERENCE_RE = re.compile(
    rf"(?:[{_TH}]{{0,12}}?)(คนนี้|คนนั้น|ร้านนี้|ร้านนั้น|บริษัทนี้|เพจนี้|แม่ค้าคนนี้|หมอคนนี้|ครูคนนี้|ตำรวจคนนี้|โรงพักนี้|รพ\.?[{_TH}A-Za-z]+)"
)
_ADDRESSEE_RE = re.compile(rf"(?<![{_TH}])(มึง|เมิง|มรึง|แก|เธอ|คุณ|หล่อน|ตัวเอง|ท่าน)(?![{_TH}])|มึง|เมิง")
_SLUR_RE = re.compile(r"(อิ|อี|ไอ้|ไอ)(ตุ๊ด|เกย์|กะเทย|เด็ก|แก่|อ้วน|ผี|เปรต|ห่า|ดอก|สัส|หอก|ควาย|เหี้ย|ชั่ว)")
_THIRD_RE = re.compile(r"มัน|แม่ง|เขา|เค้า")


@dataclass
class Target:
    text: str
    type: str
    start: int
    end: int


def _split_name(candidate: str) -> str:
    """Cut a run of Thai characters down to the first word (the name)."""
    from pythainlp.tokenize import word_tokenize

    toks = word_tokenize(candidate, engine="newmm")
    return toks[0] if toks else candidate


def extract_targets(text: str) -> list[dict]:
    if not isinstance(text, str) or not text:
        return []
    found: list[Target] = []
    taken: list[tuple[int, int]] = []

    def add(t: Target) -> None:
        if any(not (t.end <= s or t.start >= e) for s, e in taken):
            return
        taken.append((t.start, t.end))
        found.append(t)

    for m in _MENTION_RE.finditer(text):
        add(Target(m.group(0), "mention", m.start(), m.end()))

    for m in _NAMED_RE.finditer(text):
        prefix, rest = m.group(1), m.group(2)
        name = rest if rest.isascii() else _split_name(rest)
        if name in _NOT_NAMES or len(name) < 2:
            continue
        start = m.start()
        end = m.start(2) + len(name)
        add(Target(text[start:end], "named", start, end))

    for m in _REFERENCE_RE.finditer(text):
        add(Target(m.group(1), "reference", m.start(1), m.end(1)))

    for m in _ADDRESSEE_RE.finditer(text):
        add(Target(m.group(0), "addressee", m.start(), m.end()))

    for m in _SLUR_RE.finditer(text):
        add(Target(m.group(0), "slur", m.start(), m.end()))

    for m in _THIRD_RE.finditer(text):
        add(Target(m.group(0), "third", m.start(), m.end()))

    return [asdict(t) for t in sorted(found, key=lambda t: t.start)]


def has_identifiable_target(text: str) -> bool:
    """True when the message points at a specific person/organisation.

    Pronouns and slur nicknames count too: on social media "มึง", "มัน" or
    "อิตุ๊ด" in a post almost always point at one specific person.
    """
    return bool(extract_targets(text))
