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

Manual labels: 86 rows relabelled in place, 19 rows added from `manual_labels.csv`.
Synthetic defamation rows added to **train only**: 296 (validation and test contain real data only).

## 3. Class distribution (stratified 70/15/15, seed 42)

| split | normal (0) | insult (1) | defamation (2) | total |
|---|---|---|---|---|
| train | 11,926 (61.2%) | 7,190 (36.9%) | 368 (1.9%) | 19,484 |
| validation | 2,555 (62.1%) | 1,541 (37.5%) | 16 (0.4%) | 4,112 |
| test | 2,556 (62.2%) | 1,541 (37.5%) | 15 (0.4%) | 4,112 |
| all (real only) | 17,037 (62.2%) | 10,272 (37.5%) | 103 (0.4%) | 27,412 |

## 4. Source x label

| source                |   defamation |   insult |   normal |
|:----------------------|-------------:|---------:|---------:|
| HateThaiSent          |            7 |      522 |     4420 |
| ThaiCyberbullyingLGBT |           57 |     8435 |     7831 |
| ThaiToxicityTweet     |           18 |     1315 |      827 |
| WisesightSentiment    |           21 |        0 |     3959 |
