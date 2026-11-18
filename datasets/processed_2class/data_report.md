# Data preparation report

## 1. Sources

| source | rows in file | rows used |
|---|---|---|
| ThaiToxicityTweet (dataset2) | 2,160 | 2,160 |
| HateThaiSent (dataset3) | 4,953 | 4,953 |
| ThaiCyberbullyingLGBT (dataset5) | 16,749 | 16,749 |
| WisesightSentiment (dataset1) | 23,545 | 4,000 |

## 2. Cleaning steps

| step | rows |
|---|---|
| loaded (after label mapping) | 27,862 |
| + manually labelled rows not in the loaded sources | 27,881 |
| drop empty text | 27,881 |
| drop adult-content spam | 27,558 |
| drop texts shorter than 3 chars after cleaning | 27,528 |
| deduplicate on normalized text (116 duplicates, 0 conflicting texts dropped) | 27,412 |
| two-class mode: drop normal texts (17,037 rows) | 10,375 |

Manual labels: 86 rows relabelled in place, 19 rows added from `manual_labels.csv`.
Synthetic defamation rows added to **train only**: 296 (validation and test contain real data only).

## 3. Class distribution (stratified 70/15/15, seed 42)

| split | insult (0) | defamation (1) | total |
|---|---|---|---|
| train | 7,190 (95.1%) | 368 (4.9%) | 7,558 |
| validation | 1,541 (99.0%) | 15 (1.0%) | 1,556 |
| test | 1,541 (99.0%) | 16 (1.0%) | 1,557 |
| all (real only) | 10,272 (99.0%) | 103 (1.0%) | 10,375 |

## 4. Source x label

| source                |   defamation |   insult |
|:----------------------|-------------:|---------:|
| HateThaiSent          |            7 |      522 |
| ThaiCyberbullyingLGBT |           57 |     8435 |
| ThaiToxicityTweet     |           18 |     1315 |
| WisesightSentiment    |           21 |        0 |
