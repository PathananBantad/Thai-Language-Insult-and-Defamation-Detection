# Thai Insult and Defamation Detection

ระบบตรวจจับข้อความดูหมิ่นและข้อความเข้าข่ายหมิ่นประมาทภาษาไทย · CS374 Natural Language Processing, ภาคเรียนที่ 1/2569

แยกข้อความภาษาไทยบนโซเชียลมีเดียเป็น 3 คลาส คือ **ปกติ / ดูหมิ่น / เข้าข่ายหมิ่นประมาท** รองรับคำสะกดเลี่ยง (ค-ว-ย, kuy, เชี่ย) และอธิบายผลด้วยการระบุตัวบุคคลกับข้อกล่าวหา

| สมาชิก | รหัสนักศึกษา | หน้าที่ |
|---|---|---|
| แพรพลอย พิพัฒน์ภาธรกุล | 6709650540 | Dataset, annotation, preprocessing, target/allegation analysis |
| พัทธนันท์ บรรทัด | 6709650516 | Model training, evaluation, system |

> ระบบนี้ใช้คัดกรองความเสี่ยงเบื้องต้น ไม่ใช่การตัดสินความผิดตามกฎหมาย

## ผลลัพธ์ (test, n = 4,112)

| โมเดล | Accuracy | Macro-F1 | F1 ปกติ | F1 ดูหมิ่น | F1 หมิ่นประมาท |
|---|---|---|---|---|---|
| ComplementNB | 0.910 | 0.667 | 0.930 | 0.893 | 0.179 |
| **WangchanBERTa** | **0.957** | **0.843** | **0.967** | **0.942** | **0.621** |

- **คำสะกดเลี่ยง:** normalizer คืนคำเดิมได้ 34/35 แบบ, WangchanBERTa ตอบถูก 35/35 ประโยคที่สะกดเลี่ยง
- **กฎระบุตัว + ข้อกล่าวหา:** recall คลาสหมิ่นประมาท 0.84 บน val+test
- **ข้อจำกัด:** หมิ่นประมาทจริงมี 103 ข้อความ (test 15) และ label ดูหมิ่นจาก dataset เดิมถูกจริง 40.5% (สุ่มตรวจ 200 แถว)

รายละเอียด: [`docs/reports/`](docs/reports/) · metrics ดิบ: [`docs/reports/model_metrics/`](docs/reports/model_metrics/)

## Pipeline

```
ข้อความ → preprocess() → ComplementNB / WangchanBERTa → ปกติ / ดูหมิ่น / หมิ่นประมาท
             │                                          
             └→ analyze(): ระบุตัว + ข้อกล่าวหา → คำอธิบาย เช่น “พบการกล่าวหา (เมียน้อย) ที่ระบุตัวบุคคล (ครูวิชัย)”
```

## โครงสร้างที่สำคัญ

| Path | คืออะไร |
|---|---|
| `datasets/processed/` | train / validation / test (Git LFS) + `data_report.md` |
| `datasets/annotation_guidelines.md` | นิยาม 3 คลาสและผังการตัดสิน |
| `datasets/annotation/` | label หมิ่นประมาทที่ติดด้วยมือ, ตัวอย่างสังเคราะห์, ชุดทดสอบคำเลี่ยง, ผลตรวจ label |
| `src/thai_mod_api/text_processing.py` | `preprocess()`, `tokenize()` |
| `src/thai_mod_api/evasion_normalizer.py` | แปลงคำสะกดเลี่ยง |
| `src/thai_mod_api/target_extractor.py` · `allegation_analyzer.py` | ระบุตัวบุคคล + ข้อกล่าวหา, `analyze()` |
| `scripts/prepare_dataset.py` | สร้าง dataset 3 คลาส |
| `scripts/train_complement_nb_3class.py` · `train_wangchanberta_3class.py` | เทรนโมเดล (เลือกด้วย validation, วัด test ครั้งเดียว) |
| `scripts/ablation_naive_bayes.py` · `evaluate_bert_evasion.py` · `evaluate_rule_analyzer.py` | การทดลองเสริม |
| `docs/person1-handoff.md` | รายละเอียดฝั่ง dataset และ preprocessing |

## วิธีรัน

```bash
pip install -r requirements.txt
pip install rapidfuzz tabulate
git lfs pull

python scripts/prepare_dataset.py              # สร้าง datasets/processed/
python scripts/train_complement_nb_3class.py   # ~1 นาที
python scripts/train_wangchanberta_3class.py   # ~1 ชม. บน GTX 1660 Ti (3 epochs)
python scripts/evaluate_bert_evasion.py

python -m pytest tests/test_evasion_normalizer.py tests/test_prepare_dataset.py tests/test_target_allegation.py tests/test_three_class_training.py
```

ไฟล์โมเดลที่เทรนแล้วอยู่ใน `models/` ซึ่งไม่ได้เก็บใน git เพราะใหญ่เกิน ต้องเทรนใหม่ด้วยสคริปต์ข้างบน

## อ้างอิง

- วรพงศ์ บำรุงศรี (2021). *Cyberbullying Identification Using NLP in Thai Language*. จุฬาลงกรณ์มหาวิทยาลัย
- Lowphansirikul et al. (2021). *WangchanBERTa: Pretraining Transformer-based Thai Language Models*. arXiv:2101.09635
- โครงสร้างโค้ดเริ่มต้นจาก [THAI-MOD](https://github.com/Takedaxz/THAI-MOD-Multilingual-Online-Toxicity-Detection-System) (Toxic/Non-toxic) ซึ่งยังมี API, หน้าเว็บ และเอกสารเดิมอยู่ใน `src/thai_mod_api/` และ `docs/`
