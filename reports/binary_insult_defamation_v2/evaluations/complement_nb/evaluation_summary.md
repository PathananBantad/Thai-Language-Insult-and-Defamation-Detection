# Evaluation: PyThaiNLP CountVectorizer + TF-IDF + ComplementNB

This is a direct binary Insult/Defamation classifier. Test was opened only after the checkpoint was selected using Validation macro F1. No test threshold tuning was performed.

- Validation macro F1: 0.6167; support Insult=1460, Defamation=14.
- Test macro F1: 0.6403; support Insult=1456, Defamation=14.
- Real-only Test Defamation recall: 0.21428571428571427 on 14 real/other Defamation samples.
- Test Defamation support is 14; one missed Defamation changes recall by 7.14 percentage points.
The result is preliminary and does not establish generalization.
