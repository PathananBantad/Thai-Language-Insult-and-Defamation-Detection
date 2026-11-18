"""Allegation analysis: is the message accusing someone of a concrete act?

Separates
  * factual_claim   an act that could be proven true/false and harms reputation
                    ("โกงเงินบริษัท", "ติดยา", "มีเมียน้อย", "รับส่วย")
  * insult_only     name-calling / character judgement with no concrete act
                    ("ขี้โกง", "ตอแหล", "ชาติชู้", "เลว")
and flags hedged claims ("ได้ยินมาว่า", "ข่าวว่า") which still count as
allegations under the annotation guideline.

``analyze(text)`` combines this with target identification and returns the
evidence spans for highlighting. Run it on the raw text; matching is also done
on the evasion-normalized text so disguised words (โ-ก-ง) are caught.
"""

from __future__ import annotations

import re

from .evasion_normalizer import normalize_evasion
from .target_extractor import extract_targets

# Concrete misconduct (verb phrases). Grouped only for the report/explanation.
ALLEGATIONS: dict[str, tuple[str, ...]] = {
    "fraud_theft": ("โกงเงิน", "โกง", "ยักยอก", "ขโมย", "ฉ้อโกง", "หลอกเงิน", "หลอกให้โอน", "หลอกลวง", "ต้มตุ๋น",
                    "อมเงิน", "เชิดเงิน", "ชักดาบ", "เบี้ยว", "ยืมเงิน", "ไม่คืน", "ไม่จ่าย", "แอบอ้าง", "ปลอมแปลง",
                    "ขายของปลอม", "แชร์ลูกโซ่", "หนีหนี้", "ติดหนี้", "หลอก", "ปลอม"),
    "corruption": ("รับส่วย", "ส่วย", "รับเงินใต้โต๊ะ", "เงินใต้โต๊ะ", "สินบน", "รับเงิน", "ทุจริต", "คอรัปชั่น",
                   "คอร์รัปชั่น", "ซื้อเสียง", "โกงเลือกตั้ง", "โกงกิน", "ตบทรัพย์", "รีดไถ"),
    "drugs_crime": ("ติดยา", "เสพยา", "ค้ายา", "ยาบ้า", "ติดคุก", "หนีคดี", "หนีทหาร", "ข่มขืน", "ล่วงละเมิด",
                    "ทำร้ายร่างกาย", "ฆ่า", "แอบถ่าย", "ปลอมเวชระเบียน", "ปลอมเอกสาร"),
    "sexual_affair": ("เมียน้อย", "ผัวน้อย", "มีชู้", "เป็นชู้", "นอกใจ", "มีกิ๊ก", "กิ๊กเมีย", "แย่งผัว", "แย่งแฟน",
                      "ขายตัว", "เด็กเสี่ย", "ไซด์ไลน์", "เมียเก็บ"),
    "academic": ("ลอกงาน", "ขโมยงาน", "ซื้อวุฒิ", "ก็อปงาน", "ลอกข้อสอบ"),
}
# Character judgements that look like accusations but are insults
INSULT_ONLY = ("ขี้โกง", "ตอแหล", "ชาติชู้", "เจ้าชู้", "ขี้ขโมย", "ขี้ข้า", "หน้าไหว้หลังหลอก", "หลอกหลอน", "หลอกตัวเอง", "ไม่หลอก", "หลอกดาว", "เลว", "ชั่ว")
HEDGES = ("ได้ยินมาว่า", "ได้ข่าวว่า", "ข่าวว่า", "เขาว่า", "เค้าว่า", "ลือกันว่า", "มีคนบอกว่า", "จริงป่ะ", "จริงไหม",
          "จริงมั้ย", "รู้ยัง", "แฉ", "เตือนภัย")
# Negations right before an allegation ("ไม่ได้โกง") cancel it
_NEG_RE = re.compile(r"(ไม่ได้|ไม่เคย|ไม่ได้ไป|ไม่คิดจะ|มิได้|หาได้|มิเคย|ไม่)\s?(คิดจะ|ตั้งใจ|ยอมรับว่า)?\s?$")

_ALL = sorted(((w, cat) for cat, ws in ALLEGATIONS.items() for w in ws), key=lambda x: len(x[0]), reverse=True)
_ALLEG_RE = re.compile("|".join(re.escape(w) for w, _ in _ALL))
_CAT = {w: c for w, c in _ALL}
_INSULT_RE = re.compile("|".join(map(re.escape, sorted(INSULT_ONLY, key=len, reverse=True))))
_HEDGE_RE = re.compile("|".join(map(re.escape, HEDGES)))


def _find(text: str) -> list[dict]:
    out = []
    insult_spans = [(m.start(), m.end()) for m in _INSULT_RE.finditer(text)]
    for m in _ALLEG_RE.finditer(text):
        if any(s <= m.start() < e for s, e in insult_spans):  # "ขี้โกง" is an insult, not "โกง"
            continue
        if _NEG_RE.search(text[max(0, m.start() - 15): m.start()]):
            continue
        out.append({"text": m.group(0), "category": _CAT[m.group(0)], "start": m.start(), "end": m.end()})
    return out


def analyze_allegation(text: str, *, use_fuzzy: bool = True) -> dict:
    if not isinstance(text, str) or not text:
        return {"type": None, "detected": False, "hedged": False, "evidence": []}
    evidence = _find(text)
    if not evidence:
        # Try finding in normalized evasion text
        norm_text = normalize_evasion(text, use_fuzzy=use_fuzzy)
        norm_evidence = _find(norm_text)
        if norm_evidence:
            from .evasion_normalizer import find_evasion_spans

            ev_spans = find_evasion_spans(text)
            evidence = []
            for e in norm_evidence:
                matched_span = next((s for s in ev_spans if e["text"] in s["normalized"]), None)
                if matched_span:
                    evidence.append({
                        "text": matched_span["raw"],
                        "category": e["category"],
                        "start": matched_span["start"],
                        "end": matched_span["end"],
                    })
                else:
                    evidence.append({
                        "text": e["text"],
                        "category": e["category"],
                        "start": None,
                        "end": None,
                    })
    insult = bool(_INSULT_RE.search(text))
    if evidence:
        kind = "factual_claim"
    elif insult:
        kind = "insult_only"
    else:
        kind = None
    return {
        "type": kind,
        "detected": bool(evidence),
        "hedged": bool(_HEDGE_RE.search(text)),
        "evidence": evidence,
    }


def analyze(text: str) -> dict:
    """Targets + allegation + a rule-based defamation-risk flag with explanation."""
    targets = extract_targets(text)
    allegation = analyze_allegation(text)
    identified = [t for t in targets if t["type"] in {"mention", "named", "organization", "reference"}]
    addressed = [t for t in targets if t["type"] in {"addressee", "slur", "third"}]
    risk = allegation["detected"] and bool(identified or addressed)

    if risk:
        who = (identified or addressed)[0]["text"]
        what = ", ".join(sorted({e["text"] for e in allegation["evidence"]}))
        explanation = f"พบการกล่าวหา ({what}) ที่ระบุตัวบุคคล ({who})"
        if allegation["hedged"]:
            explanation += " แม้จะเป็นการเล่าต่อหรือตั้งคำถาม"
    elif allegation["detected"]:
        explanation = "พบคำกล่าวหา แต่ไม่พบเป้าหมายที่ระบุตัวได้"
    elif allegation["type"] == "insult_only":
        explanation = "เป็นคำด่า/ตัดสินนิสัย ไม่ได้กล่าวหาการกระทำที่ชัดเจน"
    else:
        explanation = "ไม่พบการกล่าวหา"
    return {"targets": targets, "allegation": allegation, "defamation_risk": risk, "explanation": explanation}
