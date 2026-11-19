# Evaluation: PhayaThaiBERT

This is a direct binary Insult/Defamation classifier. Test was opened only after the checkpoint was selected using Validation macro F1. No test threshold tuning was performed.

- Validation macro F1: 0.7672; support Insult=1460, Defamation=14.
- Test macro F1: 0.9224; support Insult=1456, Defamation=14.
- Real-only Test Defamation recall: 0.7857142857142857 on 14 real/other Defamation samples.
- Test Defamation support is 14; one missed Defamation changes recall by 7.14 percentage points.
The result is preliminary and does not establish generalization.
