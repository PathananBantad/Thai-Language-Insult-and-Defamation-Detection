# Thai Insult and Defamation Detection

ระบบตรวจจับข้อความดูหมิ่นและข้อความเข้าข่ายหมิ่นประมาทภาษาไทย (Decision Support System for Thai Insult & Defamation Detection)  
CS374 Natural Language Processing, ภาคเรียนที่ 1/2569

แยกข้อความภาษาไทยเป็น 2 คลาส คือ **ดูหมิ่น (Insult)** และ **เข้าข่ายหมิ่นประมาท (Defamation)** ตาม Proposed Workflow (Slide 5) รองรับคำสะกดเลี่ยง (Evasion Normalization) พร้อมระบุเป้าหมายบุคคล/องค์กร (Target Identification), การวิเคราะห์ข้อกล่าวหา (Allegation Analysis) และคำอธิบายเชิงหลักฐาน (Rule-based Evidence Highlighting)

| สมาชิก | รหัสนักศึกษา | หน้าที่ |
|---|---|---|
| แพรพลอย พิพัฒน์ภาธรกุล | 6709650540 | Dataset, annotation, preprocessing, target/allegation analysis |
| พัทธนันท์ บรรทัด | 6709650516 | Model training, evaluation, system |

> **ข้อควรระวัง (Disclaimer):** ระบบนี้ใช้เพื่อการคัดกรองความเสี่ยงเบื้องต้น (Preliminary Risk Screening) สำหรับผู้ดูแลเนื้อหา (Content Moderator) เท่านั้น **ไม่ใช่คำตัดสินความผิดหรือคำแนะนำทางกฎหมาย**

---

## 1. ผลการประเมินโมเดลหลัก: Dataset V2

ใช้ชุดข้อมูลหลัก: `datasets/processed/binary_insult_defamation_v2/`  
Label Mapping: `0: Insult (ดูหมิ่น)`, `1: Defamation (หมิ่นประมาท)` (คลาสข้อความทั่วไปถูกคัดออกตามขอบเขตภารกิจ)

### ผลลัพธ์บน Test Set (n = 1,470; Insult = 1,456, Defamation = 14)
*(อ้างอิงจากรายงานการประเมินจริงใน `reports/binary_insult_defamation_v2/evaluations/completed_models.json`)*

| โมเดล | Accuracy | Macro-F1 | Insult F1 (P / R) | Defamation F1 (P / R) | Mean Confidence |
|---|---|---|---|---|---|
| **WangchanBERTa V2 (Default)** | **0.9932** | **0.7483** | **0.9966** (0.9939 / 0.9993) | **0.5000** (**0.8333** / 0.3571) | **0.9966** |
| ComplementNB (Fallback) | 0.9898 | 0.6403 | 0.9949 (0.9925 / 0.9973) | 0.2857 (0.4286 / 0.2143) | 0.9409 |

### ผลลัพธ์บน Validation Set (n = 1,474; Insult = 1,460, Defamation = 14)
- **WangchanBERTa V2:** Macro-F1 = 0.7252 (Insult F1 = 0.9959, Defamation F1 = 0.4545, Precision = 0.6250, Recall = 0.3571)
- **ComplementNB:** Macro-F1 = 0.6167 (Insult F1 = 0.9935, Defamation F1 = 0.2400, Precision = 0.2727, Recall = 0.2143)

### ข้อจำกัดสำคัญของผลการประเมิน (Important Limitations)
1. **ตัวอย่างคลาสหมิ่นประมาทมีจำกัดมาก:** Test Set มีตัวอย่าง Defamation ในข้อมูลจริงเพียง **14 ตัวอย่าง** (และ Validation Set มี 14 ตัวอย่าง) ทำให้ค่า Recall และ F1-score ของคลาสนี้มีความอ่อนไหวสูงมาก การทายถูก/ผิดเพียง 1-2 ข้อความส่งผลต่อตัวเลขอย่างมีนัยสำคัญ
2. **สถานะ PhayaThaiBERT:** โมเดล PhayaThaiBERT ยังไม่ได้รับการประเมินบน Dataset V2 และไม่รวมอยู่ในตารางเปรียบเทียบโมเดลในรอบนี้
3. **Evidence Highlighting:** สแปนคำที่ไฮไลต์บนหน้าเว็บ (เป้าหมายและข้อกล่าวหา) เป็น **Rule-based evidence cues** ที่สกัดจากกฎ Regular Expression เพื่อช่วยผู้ดูแลตัดสินใจ **ไม่ใช่ Attention หรือคำอธิบายเหตุผลภายในของโครงข่ายประสาท (Not model rationales)**

---

## 2. โครงสร้าง Pipeline 6 ขั้นตอน (Slide 5 Workflow)

```
[Raw Text]
   │
   ├─► 1. Evasion Normalization: แปลงคำสะกดเลี่ยง (ค-ว-ย -> ควย, kuy -> ควย, ควายยยย -> ควาย)
   │
   ├─► 2. Target Identification: สกัดชื่อบุคคล, องค์กร, @mention (Conservative: สรรพนามลอย ๆ -> Unknown)
   │
   ├─► 3. Allegation Analysis: แยก Factual Claim (โกง, ยักยอก) ออกจาก Opinion/Insult (ขี้โกง, เลว)
   │      ตรวจคำปฏิเสธ (Negation) และคำออกตัว/ข่าวลือ (Hedges)
   │
   ├─► 4. Scope Gate: หากไม่พบเป้าหมายชัดเจน หรือไม่มีข้อกล่าวหา/คำด่า -> Abstain ส่งต่อ Human Review
   │
   ├─► 5. Binary Classification: WangchanBERTa V2 จำแนก Insult vs Defamation (เกณฑ์ความมั่นใจ >= 55%)
   │
   └─► 6. Explainability & Disclaimer: คืนค่า Evidence Spans, คำอธิบายภาษาไทย และ Legal Disclaimer
```

---

## 3. โครงสร้างไฟล์และโมเดลที่สำคัญ

| Path | รายละเอียด |
|---|---|
| `datasets/processed/binary_insult_defamation_v2/` | **ชุดข้อมูลหลัก**: `train.csv` (7,008), `validation.csv` (1,474), `test.csv` (1,470) พร้อมสรุปใน `dataset_summary.json` |
| `models/experiments/binary_insult_defamation_v2/wangchanberta_v2/` | **โมเดลหลักเริ่มต้น (Default)**: โมเดล WangchanBERTa fine-tuned พร้อม `metadata.json` |
| `models/experiments/binary_insult_defamation_v2/complement_nb/` | **โมเดลสำรอง (Fallback)**: ComplementNB pipeline (`model.joblib`) |
| `reports/binary_insult_defamation_v2/evaluations/` | รายงานผลการประเมินอย่างเป็นทางการ (`model_comparison.csv`, `completed_models.json`) |
| `src/thai_mod_api/evasion_normalizer.py` | กฎถอดรหัสคำสะกดเลี่ยง 5 ระดับ พร้อม `find_evasion_spans()` |
| `src/thai_mod_api/target_extractor.py` | สกัดเป้าหมาย (บุคคล, องค์กร, @mention) พร้อมฟังก์ชัน Conservative check `classify_target()` |
| `src/thai_mod_api/allegation_analyzer.py` | จำแนกข้อกล่าวหา 5 หมวด ตรวจจับ Negation และ Hedges |
| `src/thai_mod_api/binary_preprocessing.py` | Versioned preprocessing contract (`binary_preprocess_v1`, `USE_FUZZY=False`) |
| `src/thai_mod_api/binary_inference.py` | `BinaryPredictionService` โหลดโมเดล, ตรวจสอบ Scope Gate, ควบคุมเกณฑ์ความมั่นใจ (55%) |
| `src/thai_mod_api/main.py` | FastAPI backend สำหรับ Binary Screening Endpoint และ Legacy Endpoint |
| `src/thai_mod_api/static/` | Frontend Web UI: `index.html`, `app.js`, `styles.css` |

---

## 4. วิธีการติดตั้งและใช้งาน (Installation & Setup)

### 1) การติดตั้ง Dependencies
```bash
pip install -r requirements.txt
git lfs pull
```

### 2) การรันระบบ Backend & Frontend UI
```bash
uvicorn src.thai_mod_api.main:app --reload
```
เมื่อเซิร์ฟเวอร์เริ่มทำงานแล้ว สามารถเปิดใช้งานได้ที่:
* **Web Moderator UI:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* **Interactive API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3) การเรียกใช้งาน Binary API ผ่าน cURL
```bash
curl -X POST "http://127.0.0.1:8000/api/binary/insult-defamation/predict" \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"@somchai โกงเงินบริษัทไปสองล้าน\"}"
```

**ตัวอย่าง Response:**
```json
{
  "status": "predicted",
  "reason": null,
  "predicted_label": "Defamation",
  "probabilities": {
    "Insult": 0.12,
    "Defamation": 0.88
  },
  "confidence": 0.88,
  "target": {
    "status": "Identified",
    "entities": [
      {
        "text": "@somchai",
        "type": "mention",
        "start": 0,
        "end": 8
      }
    ]
  },
  "allegation_analysis": {
    "result": "Factual allegation cue",
    "hedged": false,
    "cues": [
      {
        "text": "โกงเงิน",
        "category": "fraud_theft"
      }
    ],
    "rule_based_only": true
  },
  "evidence_spans": [
    {
      "kind": "target_rule",
      "text": "@somchai",
      "start": 0,
      "end": 8,
      "source_text": "raw"
    },
    {
      "kind": "allegation_rule",
      "text": "โกงเงิน",
      "start": 9,
      "end": 16,
      "source_text": "raw"
    }
  ],
  "highlighting_note": "สแปนไฮไลต์แสดงเฉพาะคำที่ตรวจพบตามกฎ ไม่ใช่เหตุผลภายในของโครงข่ายประสาท (Rule-supported spans only; not model rationales)",
  "explanation": "เข้าข่ายหมิ่นประมาท (Defamation): พบ ข้อกล่าวหา (โกงเงิน) พุ่งเป้าไปที่ @somchai (ความมั่นใจ 88.0%)",
  "disclaimer": "Preliminary risk screening only; not legal advice or a legal determination."
}
```

---

## 5. การรัน Unit Tests
```bash
pytest tests/test_target_allegation.py tests/test_binary_preprocessing.py tests/test_binary_inference.py tests/test_binary_metrics.py tests/test_evasion_normalizer.py
```
*(ระบบผ่านชุดการทดสอบการแยกเป้าหมาย, ข้อกล่าวหา, การคำนวณ Metrics และการแปลงคำสะกดเลี่ยงเรียบร้อยแล้ว)*
