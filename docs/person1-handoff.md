# ส่งงานคนที่ 1 → คนที่ 2 (Dataset & Preprocessing)

## สร้างไฟล์ใหม่ทั้งหมดด้วยคำสั่งเดียว
```bash
pip install pandas scikit-learn pythainlp emoji rapidfuzz tabulate
python scripts/make_synthetic_defamation.py   # สร้างตัวอย่างสังเคราะห์ (ใช้ใน train เท่านั้น)
python scripts/prepare_dataset.py             # ได้ train/validation/test + data_report.md (~15 วินาที)
python -m pytest tests/test_evasion_normalizer.py tests/test_prepare_dataset.py
```

## ไฟล์ที่ส่งให้
| ไฟล์ | คืออะไร |
|---|---|
| `datasets/processed/train.csv` | 19,477 แถว (รวมตัวอย่างสังเคราะห์ 296 แถว) |
| `datasets/processed/validation.csv` | 4,110 แถว ใช้ตอนเลือกโมเดลและจูน hyperparameter |
| `datasets/processed/test.csv` | 4,111 แถว **ใช้ครั้งเดียวตอนจบ** |
| `datasets/processed/data_report.md` | ตัวเลขทุกขั้นตอนของการ clean + class distribution (เอาไปใส่รายงานได้เลย) |
| `datasets/annotation_guidelines.md` | นิยาม label และวิธีตัดสิน |
| `src/thai_mod_api/text_processing.py` | `preprocess(text)`, `tokenize(text)` |
| `src/thai_mod_api/evasion_normalizer.py` | `normalize_evasion(text)`, `find_evasions(text)` |

### คอลัมน์
`text_raw` (ข้อความต้นฉบับ ใช้ไฮไลต์), `text_norm` (ผ่าน `preprocess()` แล้ว ใช้เทรน), `label` (0/1/2), `label_name`, `source`

### Label
| label | ชื่อ | train | val | test |
|---|---|---|---|---|
| 0 | normal | 11,920 | 2,554 | 2,555 |
| 1 | insult | 7,193 | 1,541 | 1,542 |
| 2 | defamation | 364 | 15 | 14 |

## วิธีใช้ฝั่งโมเดล
```python
import pandas as pd
from src.thai_mod_api.text_processing import preprocess, tokenize

train = pd.read_csv("datasets/processed/train.csv")

# Naive Bayes / TF-IDF
CountVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2))   # fit บน train["text_norm"]

# WangchanBERTa / PhayaThaiBERT: ส่ง train["text_norm"] เข้า tokenizer ของโมเดลได้เลย ไม่ต้องตัดคำ

# ตอน inference (API) ต้องเรียก preprocess() ก่อนเสมอ
model.predict([preprocess(user_text)])
```

## สิ่งที่คนที่ 2 ต้องรู้
1. **ข้อมูลไม่สมดุลมาก** (หมิ่นประมาทน้อยกว่า 2%) ให้ใช้ `class_weight="balanced"` หรือ weighted loss และ **รายงาน precision/recall/F1 รายคลาส กับ macro-F1** อย่าดู accuracy อย่างเดียว
2. **test มีหมิ่นประมาทแค่ 14 ข้อความ** ให้รายงานจำนวนตัวอย่าง (support) คู่กับค่า F1 เสมอ
3. **เลือกโมเดลด้วย validation** ส่วน `scripts/train_wangchanberta.py` ตัวเดิมเลือก epoch จาก test set ซึ่งทำให้ข้อมูล test รั่ว ต้องแก้ให้ใช้ validation
4. **ห้ามเอาผล 83.85% ของ repo เดิมไปอ้าง** เพราะเป็นโจทย์ toxic/non-toxic คนละโจทย์
5. ตัวอย่างสังเคราะห์มี `source == "Synthetic"` ถ้าอยากทดลองว่าช่วยไหม ให้เทรนแบบตัดออกด้วย `--no-synthetic` แล้วเทียบผล (เป็นหัวข้อที่ใส่ในรายงานได้)

## สิ่งที่ทำในขั้นตอน preprocessing
| ขั้น | ตัวอย่าง |
|---|---|
| ลบ URL และลิงก์ pic.twitter | `ดูนี่ https://x.com` → `ดูนี่` |
| @mention → `<USER>` | `@somchai_99` → `<USER>` |
| emoji → ข้อความ | 🤬 → `:face_with_symbols_on_mouth:` |
| ตัวอังกฤษเป็นตัวเล็ก, ลบ `#` | `KUY #ซีเกมส์` → `kuy ซีเกมส์` |
| ลบตัวคั่นระหว่างตัวอักษร | `ค-ว-ย`, `ค.ว.ย`, `ค ว ย` → `ควย` |
| ย่อตัวอักษรที่ซ้ำ | `ควายยยย` → `ควาย`, `โง่ววววว` → `โง่`, `5555555` → `555` |
| คำคาราโอเกะ | `kuy` → `ควย`, `sus` → `สัส` |
| คำเลี่ยง/สะกดเพี้ยน | `เชี่ย` → `เหี้ย`, `มรึง` → `มึง`, `อีควัย` → `อีควาย` |
| edit distance ≤ 1 กับคำใน `toxic_keywords.csv` (เฉพาะคำที่ไม่อยู่ในพจนานุกรมไทย) | `เหี่ย` → `เหี้ย` แต่ `ความ` ยังเป็น `ความ` |

ข้อความ 41% ของ dataset ถูกแก้โดยขั้นตอน normalize อย่างน้อยหนึ่งจุด

## สิ่งที่พบระหว่างทำ (ใส่ในรายงานได้)
- dataset5 (LGBT) มี**สแปมโฆษณาหนังโป๊**ปนอยู่ประมาณ 320 แถว ลบออกแล้ว
- `neg` ของ Wisesight คืออารมณ์ลบ เช่น บ่นเน็ตช้า ไม่ใช่คำดูหมิ่น จึงไม่เอามาใช้เป็นคลาสดูหมิ่น
- dataset4 เป็นรีวิวสินค้า ส่วน dataset6–8 เป็นภาษาอังกฤษ อยู่นอกขอบเขตของโจทย์
