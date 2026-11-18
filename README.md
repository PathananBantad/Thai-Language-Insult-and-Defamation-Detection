# Thai Insult and Defamation Detection

ระบบตรวจจับข้อความดูหมิ่นและข้อความเข้าข่ายหมิ่นประมาทภาษาไทย · CS374 Natural Language Processing, ภาคเรียนที่ 1/2569

แยกข้อความทำร้ายภาษาไทยบนโซเชียลมีเดียเป็น 2 คลาส คือ **ดูหมิ่น / เข้าข่ายหมิ่นประมาท** ตามขอบเขตใน proposal รองรับคำสะกดเลี่ยง (ค-ว-ย, kuy, เชี่ย) และอธิบายผลด้วยการระบุตัวบุคคลกับข้อกล่าวหา

| สมาชิก | รหัสนักศึกษา | หน้าที่ |
|---|---|---|
| แพรพลอย พิพัฒน์ภาธรกุล | 6709650540 | Dataset, annotation, preprocessing, target/allegation analysis |
| พัทธนันท์ บรรทัด | 6709650516 | Model training, evaluation, system |

> ระบบนี้ใช้คัดกรองความเสี่ยงเบื้องต้น ไม่ใช่การตัดสินความผิดตามกฎหมาย

## ผลลัพธ์หลัก: 2 คลาส (test, n = 1,557, หมิ่นประมาท 16)

| วิธี | Accuracy | Macro-F1 | F1 ดูหมิ่น | หมิ่นประมาท P / R / F1 |
|---|---|---|---|---|
| ComplementNB | 0.987 | 0.612 | 0.994 | 0.30 / 0.19 / 0.23 |
| WangchanBERTa | 0.989 | 0.628 | 0.995 | 0.43 / 0.19 / 0.26 |
| **กฎระบุตัว + ข้อกล่าวหา** | 0.981 | **0.736** | 0.990 | 0.33 / **0.88** / **0.48** |

- **ดูหมิ่น:** ทุกวิธีแยกได้แม่นยำ (F1 ≈ 0.99)
- **หมิ่นประมาท:** กฎที่เขียนจากนิยามกฎหมาย (ระบุตัวได้ + กล่าวหาการกระทำ) ดีที่สุด ส่วนโมเดลที่เรียนรู้จากข้อมูลจับได้แค่ 3 จาก 16 เพราะมีตัวอย่างจริงให้เรียนเพียง 103 ข้อความ การปรับ threshold ของ BERT บน validation ช่วยได้เพียง 0.26 เป็น 0.29
- **คำสะกดเลี่ยง:** normalizer คืนคำได้ 34/35 แบบ · NB ตอบถูก 33/33 · WangchanBERTa 28/33 ประโยค
- **ข้อจำกัด:** label ดูหมิ่นจาก dataset เดิมถูกจริง 40.5% (สุ่มตรวจ 200 แถว) และระบบจำแนกเฉพาะข้อความที่เป็นการโจมตีอยู่แล้ว

### การทดลองเสริม: 3 คลาส (ปกติ / ดูหมิ่น / หมิ่นประมาท)
บน `datasets/processed/` (test n = 4,112): ComplementNB macro-F1 0.667 · WangchanBERTa macro-F1 0.843 (หมิ่นประมาท F1 0.62)

metrics ทั้งหมด: [`docs/reports/model_metrics/`](docs/reports/model_metrics/) · รายงานการทดลอง: [`docs/reports/`](docs/reports/)

## Pipeline

```
ข้อความโจมตี → preprocess() → ComplementNB / WangchanBERTa → ดูหมิ่น / หมิ่นประมาท
             │                                          
             └→ analyze(): ระบุตัว + ข้อกล่าวหา → คำอธิบาย เช่น “พบการกล่าวหา (เมียน้อย) ที่ระบุตัวบุคคล (ครูวิชัย)”
```

## โครงสร้างที่สำคัญ

| Path | คืออะไร |
|---|---|
| `datasets/processed_2class/` | **หลัก**: train / validation / test แบบ ดูหมิ่น vs หมิ่นประมาท (Git LFS) + `data_report.md` |
| `datasets/processed/` | การทดลองเสริมแบบ 3 คลาส |
| `datasets/annotation_guidelines.md` | นิยาม 3 คลาสและผังการตัดสิน |
| `datasets/annotation/` | label หมิ่นประมาทที่ติดด้วยมือ, ตัวอย่างสังเคราะห์, ชุดทดสอบคำเลี่ยง, ผลตรวจ label |
| `src/thai_mod_api/text_processing.py` | `preprocess()`, `tokenize()` |
| `src/thai_mod_api/evasion_normalizer.py` | แปลงคำสะกดเลี่ยง |
| `src/thai_mod_api/target_extractor.py` · `allegation_analyzer.py` | ระบุตัวบุคคล + ข้อกล่าวหา, `analyze()` |
| `scripts/prepare_dataset.py` | สร้าง dataset 3 คลาส |
| `scripts/train_2class.py` | เทรนแบบ 2 คลาส (`nb` / `bert`) โดยใช้สคริปต์ 3 คลาสเดิม แค่สลับ label |
| `scripts/train_complement_nb_3class.py` · `train_wangchanberta_3class.py` | สคริปต์เทรนหลัก (เลือกด้วย validation, วัด test ครั้งเดียว) |
| `scripts/tune_threshold_2class.py` | เลือก threshold หมิ่นประมาทของ BERT บน validation |
| `scripts/ablation_naive_bayes.py` · `evaluate_bert_evasion.py` · `evaluate_rule_analyzer.py` | การทดลองเสริม |
| `docs/person1-handoff.md` | รายละเอียดฝั่ง dataset และ preprocessing |

## วิธีรัน

```bash
pip install -r requirements.txt
pip install rapidfuzz tabulate
git lfs pull

# หลัก: 2 คลาส
python scripts/prepare_dataset.py --two-class --out-dir datasets/processed_2class
python scripts/train_2class.py nb              # ~1 นาที
python scripts/train_2class.py bert --run-dir models/training_runs/wcb_2class   # ~20 นาที บน GTX 1660 Ti
python scripts/evaluate_bert_evasion.py --two-class
python scripts/tune_threshold_2class.py

# เสริม: 3 คลาส
python scripts/prepare_dataset.py
python scripts/train_complement_nb_3class.py
python scripts/train_wangchanberta_3class.py   # ~1 ชม.

python -m pytest tests/test_evasion_normalizer.py tests/test_prepare_dataset.py tests/test_target_allegation.py tests/test_three_class_training.py
```

ไฟล์โมเดลที่เทรนแล้วอยู่ใน `models/` ซึ่งไม่ได้เก็บใน git เพราะใหญ่เกิน ต้องเทรนใหม่ด้วยสคริปต์ข้างบน

## อ้างอิง

- วรพงศ์ บำรุงศรี (2021). *Cyberbullying Identification Using NLP in Thai Language*. จุฬาลงกรณ์มหาวิทยาลัย
- Lowphansirikul et al. (2021). *WangchanBERTa: Pretraining Transformer-based Thai Language Models*. arXiv:2101.09635
- โครงสร้างโค้ดเริ่มต้นจาก [THAI-MOD](https://github.com/Takedaxz/THAI-MOD-Multilingual-Online-Toxicity-Detection-System) (Toxic/Non-toxic) ซึ่งยังมี API, หน้าเว็บ และเอกสารเดิมอยู่ใน `src/thai_mod_api/` และ `docs/`
