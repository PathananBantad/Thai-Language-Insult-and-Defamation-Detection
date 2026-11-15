# Rule-based defamation flag (target + allegation)

Data: real validation + test rows (n = 8,224, defamation n = 31).

| metric | value |
|---|---|
| precision | 0.133 |
| recall | 0.839 |
| F1 | 0.230 |
| TP / FP / FN | 26 / 169 / 5 |

Share of each gold class flagged as defamation risk:

| label_name   |   flagged_rate |
|:-------------|---------------:|
| defamation   |          0.839 |
| insult       |          0.022 |
| normal       |          0.020 |

High recall + low precision is expected: the rules find *candidates* (target + accusation words) and the classifier decides. The flag and its evidence spans are meant for explanation/highlighting, not as the final label.

## Missed defamation examples (FN)

- ประชาชนชาวไทยเหมือนตายทั้งเป็น นับแต่มีปีศาจทหารขี้ฉ้อตอแหลยึดอำนาจ หลอกลวงปวงชนว่าขอเวลาอีกไม่นานจะคืนความสุขให้ประชาชน
- เอาบริษัทคนอื่นไปแอบอ้าง แล้วจดชื่อสินค้าตัวเองใน อย.นี่โครตทุเรศ
- เอาผิด บริษัท เบียร์สิงห์ด้วยที่เป็นเจ้าของ เบียร์ลีโอ ฐานสนับสนุน เซว่น ยินยอมให้เอาเบียร์สด ที่เป็นสินค้าของบริษัทมาวา
- ท่านจะอยู่ในตำแหน่งเพื่อทำหน้าที่นี้อีกนานไหมครับ 😂 #ส่วยเบียร์ช้าง
- ก็ไปถามไอ้ประยุทธ์กับไอ้ประวิทย์สิว่ามึงใช้ให้ลูกน้องมึงรับเกินจากคนจีนพม่าแรงงานพม่าลาวกัมพูชารับมาจนมึงมึงไม่มีสิทธิ์ท
