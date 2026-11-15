# Naive Bayes ablation

Model: CountVectorizer(newmm, 1-2 gram) + TF-IDF + ComplementNB, balanced sample weights.
Scored on the **validation** split (n = 4,112, defamation n = 16) and on the evasion robustness set (`evasion_testset.csv`, 60 rows, 35 disguised).

| normalization   | synthetic   |   val_macro_f1 |   val_normal_f1 |   val_insult_f1 |   val_defamation_f1 |   val_defamation_recall |   evasion_acc_disguised |   evasion_acc_all |
|:----------------|:------------|---------------:|----------------:|----------------:|--------------------:|------------------------:|------------------------:|------------------:|
| no              | no          |          0.640 |           0.926 |           0.878 |               0.117 |                   0.375 |                   0.743 |             0.733 |
| no              | yes         |          0.655 |           0.931 |           0.894 |               0.140 |                   0.188 |                   0.857 |             0.850 |
| yes             | no          |          0.640 |           0.926 |           0.879 |               0.117 |                   0.375 |                   0.686 |             0.683 |
| yes             | yes         |          0.653 |           0.931 |           0.894 |               0.133 |                   0.188 |                   0.971 |             0.917 |

- **normalization** = evasion normalization on (`preprocess`) vs off (`clean_text` only)
- **synthetic** = synthetic defamation rows kept in train vs removed
- **evasion_acc_disguised** = accuracy on the disguised sentences only

Note: validation has only 16 defamation examples, so defamation F1 moves a lot with a single prediction. Treat differences there as indicative, not conclusive.
