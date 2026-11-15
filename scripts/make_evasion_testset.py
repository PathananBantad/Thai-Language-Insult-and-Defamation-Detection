"""Build a small evasion robustness test set.

Every toxic base sentence appears once in plain form and several times in a
disguised form (spaced letters, repeated chars, karaoke, euphemism spelling).
Normal sentences with look-alike words are included as negatives so that a
model/normalizer that over-corrects is penalised too.

Usage:
    python scripts/make_evasion_testset.py   # -> datasets/annotation/evasion_testset.csv

Columns: text, label (0 normal / 1 insult / 2 defamation), variant, base_id
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "annotation" / "evasion_testset.csv"

# (base_id, label, plain, [(variant_name, disguised), ...])
ITEMS = [
    (1, 1, "มึงมันควาย", [("spaced", "มึงมันค-ว-า-ย"), ("euphemism", "มึงมันควัย"), ("repeat", "มึงมันควายยยยย")]),
    (2, 1, "ไอ้เหี้ย ไปตายซะ", [("euphemism", "ไอ้เชี่ย ไปตายซะ"), ("karaoke", "ai hia ไปตายซะ"), ("misspell", "ไอ้เหี่ย ไปตายซะ")]),
    (3, 1, "อีสัส หน้าด้าน", [("karaoke", "อี sus หน้าด้าน"), ("euphemism", "อิสัด หน้าด้าน"), ("spaced", "อี ส ั ส หน้าด้าน")]),
    (4, 1, "ควยไรมึงเนี่ย", [("karaoke", "kuy ไรมึงเนี่ย"), ("spaced", "ค-ว-ย ไรมึงเนี่ย"), ("spaced", "ค.ว.ย ไรมึงเนี่ย")]),
    (5, 1, "แม่งโง่ฉิบหาย", [("euphemism", "แมร่งโง่ฉิบหาย"), ("repeat", "แม่งโง่ววววววฉิบหาย"), ("euphemism", "แม่มโง่ฉิบหาย")]),
    (6, 1, "มึงนี่ปัญญาอ่อนจริงๆ", [("euphemism", "มรึงนี่ปัญญาอ่อนจริงๆ"), ("euphemism", "เมิงนี่ปัญญาอ่อนจริงๆ")]),
    (7, 1, "อีดอก ตอแหลเก่งนัก", [("euphemism", "อิดอก ตอแหลเก่งนัก"), ("karaoke", "อีดอก torlae เก่งนัก"), ("repeat", "อีดอกกกกก ตอแหลเก่งนัก")]),
    (8, 1, "ไอ้สัตว์ ส้นตีน", [("misspell", "ไอ้สัตว์ ส้นตรีน"), ("repeat", "ไอ้สัตว์ ส้นตีนนนนน")]),
    (9, 1, "กูเกลียดมึงไอ้ควาย", [("euphemism", "กุเกลียดมึงไอ้ควัย"), ("karaoke", "กูเกลียดมึงไอ้ kwai")]),
    (10, 1, "อีกะหรี่ ร่านไปทั่ว", [("euphemism", "อีกระหรี่ ร่านไปทั่ว"), ("repeat", "อีกะหรี่ ร่านนนนไปทั่ว")]),
    (11, 2, "@somchai โกงเงินบริษัทไปสองล้าน", [("spaced", "@somchai โ-ก-ง เงินบริษัทไปสองล้าน"), ("repeat", "@somchai โกงงงงงเงินบริษัทไปสองล้าน")]),
    (12, 2, "ครูวิชัยแอบมีเมียน้อย", [("repeat", "ครูวิชัยแอบมีเมียน้อยยยยย")]),
    (13, 2, "อีเจนขโมยของในหอ", [("euphemism", "อิเจนขโมยของในหอ"), ("spaced", "อีเจน ข-โ-ม-ย ของในหอ")]),
    (14, 2, "ไอ้โจ้ติดยาบ้า ระวังมึงด้วย", [("euphemism", "ไอ้โจ้ติดยาบ้า ระวังมรึงด้วย"), ("repeat", "ไอ้โจ้ติดยาบ้าาาาา ระวังมึงด้วย")]),
    # Normal sentences with look-alike words (must stay normal)
    (15, 0, "ผมเชี่ยวชาญด้านนี้มาก", []),
    (16, 0, "ความรู้สึกวันนี้ดีมาก", []),
    (17, 0, "แม่มดในนิทานน่ากลัว", []),
    (18, 0, "สงกรานต์ปีนี้สาดน้ำสนุกมาก", []),
    (19, 0, "กุ้งเผาร้านนี้อร่อย", []),
    (20, 0, "อิทธิพลของสื่อต่อเด็ก", []),
    (21, 0, "ควายเป็นสัตว์ที่ช่วยชาวนา", []),
    (22, 0, "ดีมากกกกก ชอบสุดๆ", [("repeat", "ดีมากกกกกกกกก ชอบสุดๆๆๆ")]),
    (23, 0, "focus on the AI model", []),
    (24, 0, "รถติดมาก ร้อนด้วย", [("repeat", "รถติดมากกกก ร้อนด้วยยยย")]),
    (25, 0, "ขอบคุณนะครับ ช่วยได้เยอะเลย", []),
]


def main() -> None:
    rows = []
    for base_id, label, plain, variants in ITEMS:
        rows.append({"text": plain, "label": label, "variant": "plain", "base_id": base_id})
        for name, text in variants:
            rows.append({"text": text, "label": label, "variant": name, "base_id": base_id})
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {len(df)} rows ({(df.variant != 'plain').sum()} disguised) -> {OUT}")


if __name__ == "__main__":
    main()
