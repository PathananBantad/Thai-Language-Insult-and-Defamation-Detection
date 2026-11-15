"""Generate synthetic Thai defamation examples for TRAINING ONLY.

Real defamation tweets are rare in the public datasets (~100 hand-labelled), so
we augment the training split with templated examples built from fictional
targets and accusations. They are never put in validation/test, so the
evaluation stays on real data only.

All names are fictional placeholders; no real person is referenced.

Usage:
    python scripts/make_synthetic_defamation.py            # writes datasets/annotation/synthetic_defamation.csv
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "annotation" / "synthetic_defamation.csv"
SEED = 42

# Identifiable targets (fictional)
TARGETS = [
    "@user_nok", "@pim_2545", "@bank.official", "@kaew_shop", "@tonkla99",
    "นายสมชาย", "นางสาวมะลิ", "ครูวิชัย", "อาจารย์ปรีชา", "หมอสมศักดิ์", "ผู้ใหญ่บ้านหมู่ 3",
    "เจ้าของร้านกาแฟหน้ามอ", "หัวหน้าแผนกบัญชี", "ผจก.สาขาบางนา", "ร้านเสื้อผ้า @kaew_shop",
    "แม่ค้าเพจ ลดล้างสต็อก", "พี่ตั้มห้องข้างๆ", "เพื่อนร่วมงานชื่อแนน", "แฟนเก่ากู", "น้องเอ็มปีหนึ่ง",
    "ไอ้โจ้ ม.5/2", "อีเจน หอใน", "คนขับวินหน้าซอย 7", "ตำรวจโรงพักนี้คนนึง", "บริษัท ABC ทัวร์",
]

# Factual-sounding allegations (verifiable act that damages reputation)
ALLEGATIONS = [
    "โกงเงินลูกค้าไปเป็นแสน", "ยักยอกเงินกองกลางห้อง", "ขโมยของในร้านสะดวกซื้อ", "แอบรับเงินใต้โต๊ะ",
    "ติดยาบ้ามาหลายปีแล้ว", "เคยติดคุกคดีฉ้อโกง", "มีเมียน้อยอยู่ต่างจังหวัด", "นอกใจแฟนไปมีกิ๊ก",
    "หลอกให้คนโอนเงินแล้วบล็อก", "ขายของปลอมแต่บอกว่าของแท้", "ยืมเงินเพื่อนแล้วไม่ยอมคืน",
    "ปลอมเอกสารสมัครงาน", "ซื้อวุฒิปริญญามา", "ทำร้ายร่างกายแฟน", "แอบถ่ายรูปผู้หญิงในห้องน้ำ",
    "เอาเงินบริจาคไปใช้ส่วนตัว", "รับส่วยจากบ่อนแถวนี้", "ลอกงานคนอื่นไปส่ง", "เบี้ยวค่าจ้างลูกน้อง",
    "ชักดาบค่าอาหารร้านป้า", "ทิ้งลูกทิ้งเมียหนีหนี้", "ซื้อเสียงตอนเลือกตั้ง", "แชร์ลูกโซ่หลอกคนในหมู่บ้าน",
]

# Sentence frames: {t} target, {a} allegation
FRAMES = [
    "{t} {a}",
    "{t} {a} รู้กันทั้งซอย",
    "บอกไว้ก่อนนะ {t} {a}",
    "ระวัง {t} ด้วย คนนี้{a}",
    "เตือนภัย {t} {a} อย่าไปหลงเชื่อ",
    "ใครรู้จัก {t} บ้าง มัน{a}",
    "ได้ยินมาว่า {t} {a} จริงป่ะ",
    "แฉ {t} {a} มีหลักฐานด้วย",
    "{t} อย่ามาทำเป็นคนดี {a}",
    "คนที่ชื่อ {t} อะ {a} นะ",
    "{t} {a} แล้วยังมีหน้ามาโพสต์อีก",
    "รู้ยัง {t} {a}",
    "{t} ที่ทุกคนชมกันนักหนา จริงๆ{a}",
    "ไม่อยากพูดเยอะ แต่ {t} {a}",
    "{t} {a} สมควรโดนไล่ออก",
    "อีห่า {t} {a}",
    "{t} มึง{a} ใช่มั้ย ทุกคนรู้หมดแล้ว",
]

N = 300


def main() -> None:
    rng = random.Random(SEED)
    seen: set[str] = set()
    rows = []
    while len(rows) < N:
        text = rng.choice(FRAMES).format(t=rng.choice(TARGETS), a=rng.choice(ALLEGATIONS))
        if text in seen:
            continue
        seen.add(text)
        rows.append({"text": text, "label": 2, "source": "Synthetic", "note": "synthetic_train_only"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
