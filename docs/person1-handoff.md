# ส่งงานคนที่ 1 → คนที่ 2 (Dataset & Preprocessing)

## สร้างไฟล์ใหม่ทั้งหมดด้วยคำสั่งเดียว
```bash
pip install pandas scikit-learn pythainlp emoji rapidfuzz tabulate
python scripts/make_synthetic_defamation.py   # สร้างตัวอย่างสังเคราะห์ (ใช้ใน train เท่านั้น)
python scripts/make_evasion_testset.py        # ชุดทดสอบคำเลี่ยง 60 ประโยค
python scripts/prepare_dataset.py             # ได้ train/validation/test + data_report.md (~15 วินาที)
python scripts/ablation_naive_bayes.py        # ทดลอง NB: มี/ไม่มี normalization, มี/ไม่มีตัวอย่างสังเคราะห์
python scripts/evaluate_rule_analyzer.py      # วัดผลกฎ target + allegation
python -m pytest tests/test_evasion_normalizer.py tests/test_prepare_dataset.py tests/test_target_allegation.py
```

## ไฟล์ที่ส่งให้
| ไฟล์ | คืออะไร |
|---|---|
| `datasets/processed/train.csv` | 19,484 แถว (รวมตัวอย่างสังเคราะห์ 296 แถว) |
| `datasets/processed/validation.csv` | 4,112 แถว ใช้ตอนเลือกโมเดลและจูน hyperparameter |
| `datasets/processed/test.csv` | 4,112 แถว **ใช้ครั้งเดียวตอนจบ** |
| `datasets/processed/data_report.md` | ตัวเลขทุกขั้นตอนของการ clean + class distribution (เอาไปใส่รายงานได้เลย) |
| `datasets/annotation_guidelines.md` | นิยาม label และวิธีตัดสิน |
| `src/thai_mod_api/text_processing.py` | `preprocess(text)`, `tokenize(text)` |
| `src/thai_mod_api/evasion_normalizer.py` | `normalize_evasion(text)`, `find_evasions(text)` |
| `src/thai_mod_api/target_extractor.py` | `extract_targets(text)` หาเป้าหมาย (@mention, ชื่อ, "ร้านนี้", มึง, อิตุ๊ด) พร้อมตำแหน่ง |
| `src/thai_mod_api/allegation_analyzer.py` | `analyze(text)` คืน targets + การกล่าวหา + `defamation_risk` + คำอธิบายภาษาไทย ใช้ทำ explainability/highlight |
| `datasets/annotation/evasion_testset.csv` | ชุดทดสอบความทนต่อคำเลี่ยง (60 ประโยค, 35 แบบสะกดเลี่ยง) |
| `datasets/annotation/insult_audit.csv` | ผลตรวจ label ดูหมิ่น 200 แถว (`is_real_insult`) |
| `docs/reports/ablation_naive_bayes.md` | ผลทดลอง Naive Bayes 4 แบบ |
| `docs/reports/rule_analyzer.md` | ผลวัดกฎ target + allegation |

### คอลัมน์
`text_raw` (ข้อความต้นฉบับ ใช้ไฮไลต์), `text_norm` (ผ่าน `preprocess()` แล้ว ใช้เทรน), `label` (0/1/2), `label_name`, `source`

### Label
| label | ชื่อ | train | val | test |
|---|---|---|---|---|
| 0 | normal | 11,926 | 2,555 | 2,556 |
| 1 | insult | 7,190 | 1,541 | 1,541 |
| 2 | defamation | 368 | 16 | 15 |

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

# Explainability: เป้าหมาย + ข้อความที่เป็นการกล่าวหา (ตำแหน่งอิงข้อความต้นฉบับ)
from src.thai_mod_api.allegation_analyzer import analyze
analyze("ครูวิชัยแอบมีเมียน้อย")
# {'targets': [{'text': 'ครูวิชัย', 'type': 'named', 'start': 0, 'end': 8}],
#  'allegation': {'type': 'factual_claim', 'evidence': [{'text': 'เมียน้อย', ...}], ...},
#  'defamation_risk': True, 'explanation': 'พบการกล่าวหา (เมียน้อย) ที่ระบุตัวบุคคล (ครูวิชัย)'}
```

## สิ่งที่คนที่ 2 ต้องรู้
1. **ข้อมูลไม่สมดุลมาก** (หมิ่นประมาทน้อยกว่า 2%) ให้ใช้ `class_weight="balanced"` หรือ weighted loss และ **รายงาน precision/recall/F1 รายคลาส กับ macro-F1** อย่าดู accuracy อย่างเดียว
2. **test มีหมิ่นประมาทแค่ 15 ข้อความ** ให้รายงานจำนวนตัวอย่าง (support) คู่กับค่า F1 เสมอ
3. **เลือกโมเดลด้วย validation** ส่วน `scripts/train_wangchanberta.py` ตัวเดิมเลือก epoch จาก test set ซึ่งทำให้ข้อมูล test รั่ว ต้องแก้ให้ใช้ validation
4. **ห้ามเอาผล 83.85% ของ repo เดิมไปอ้าง** เพราะเป็นโจทย์ toxic/non-toxic คนละโจทย์
5. ตัวอย่างสังเคราะห์มี `source == "Synthetic"` ผลทดลองกับ NB อยู่ใน `docs/reports/ablation_naive_bayes.md` แล้ว ลองทำแบบเดียวกันกับ BERT ได้
6. **label ดูหมิ่นมี noise สูง** (ดูหัวข้อด้านล่าง) ค่า F1 ของคลาสดูหมิ่นจึงวัด "ความเหมือน label เดิม" มากกว่าความเข้าใจคำดูหมิ่นจริง ควรเขียนข้อนี้ในรายงาน
7. ใช้ `evasion_testset.csv` วัดโมเดลทุกตัวด้วย เพื่อเทียบว่า BERT ทนคำเลี่ยงได้ดีกว่า NB หรือไม่

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

## ผลทดลอง (ใส่ในรายงานได้)

### Naive Bayes ablation (วัดบน validation + ชุดคำเลี่ยง)
| normalization | synthetic | macro-F1 (val) | ความแม่นบนข้อความที่สะกดเลี่ยง |
|---|---|---|---|
| ไม่มี | ไม่มี | 0.640 | 74.3% |
| ไม่มี | มี | 0.655 | 85.7% |
| มี | ไม่มี | 0.640 | 68.6% |
| **มี** | **มี** | **0.653** | **97.1%** |

- บน validation ค่าใกล้กันมาก เพราะข้อความจริงใน dataset มีคำเลี่ยงไม่มาก
- พอเจอข้อความที่สะกดเลี่ยงจริงๆ การใช้ normalization คู่กับตัวอย่างสังเคราะห์เพิ่มความแม่นจาก 74% เป็น 97%
- แบบที่มี normalization แต่ไม่มีตัวอย่างสังเคราะห์ได้ต่ำสุด เพราะในชุดคำเลี่ยงมีประโยคหมิ่นประมาทที่โมเดลแทบไม่เคยเห็นตอนเทรน

### กฎ target + allegation (วัดบน val + test ที่เป็นข้อมูลจริง)
- recall ของคลาสหมิ่นประมาท **0.84** (จับได้ 26 จาก 31 ข้อความ)
- ไปติดธงข้อความปกติหรือดูหมิ่นแค่ประมาณ **2%**
- precision ต่ำ (0.13) เพราะกฎหาแค่ "ผู้ต้องสงสัย" ให้โมเดลตัดสิน และใช้ไฮไลต์อธิบายผล

### ตรวจ label ดูหมิ่น 200 แถว (สุ่มจาก val + test)
| แหล่ง | แถวที่ตรวจ | เป็นคำดูหมิ่นจริง |
|---|---|---|
| ThaiToxicityTweet | 23 | 91% |
| HateThaiSent | 10 | 50% |
| ThaiCyberbullyingLGBT | 167 | **33%** |
| รวม | 200 | **40.5%** |

dataset5 (LGBT) ติด label `neg` ให้เกือบทุกข้อความที่มีคำว่า "อิตุ๊ด/อีเกย์" รวมถึงตอนที่เพื่อนเรียกกันแบบสนิทสนม เช่น "ขอให้มึงมีความสุขมากๆนะอิตุ๊ด" ซึ่งตามไกด์ไลน์ของเราไม่นับเป็นการดูหมิ่น นี่คือข้อจำกัดใหญ่ที่สุดของ dataset นี้ ทางแก้ระยะยาวคือติด label คลาสดูหมิ่นของ dataset5 ใหม่

### การขุดหาข้อความหมิ่นประมาท
อ่านข้อความที่ผ่านตัวกรองคำกล่าวหาไปทั้งหมด 966 ข้อความ เจอหมิ่นประมาทจริง 103 ข้อความ รอบสุดท้ายเจอแค่ 6 จาก 381 แปลว่าข้อความหมิ่นประมาทใน dataset ชุดนี้ถูกขุดจนเกือบหมดแล้ว ถ้าอยากได้มากกว่านี้ต้องหาแหล่งข้อมูลใหม่

## สิ่งที่พบระหว่างทำ (ใส่ในรายงานได้)
- dataset5 (LGBT) มี**สแปมโฆษณาหนังโป๊**ปนอยู่ประมาณ 320 แถว ลบออกแล้ว
- `neg` ของ Wisesight คืออารมณ์ลบ เช่น บ่นเน็ตช้า ไม่ใช่คำดูหมิ่น จึงไม่เอามาใช้เป็นคลาสดูหมิ่น
- dataset4 เป็นรีวิวสินค้า ส่วน dataset6–8 เป็นภาษาอังกฤษ อยู่นอกขอบเขตของโจทย์
