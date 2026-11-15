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
| + manually labelled rows not in the loaded sources | 27,880 |
| drop empty text | 27,880 |
| drop adult-content spam | 27,557 |
| drop texts shorter than 3 chars after cleaning | 27,526 |
| deduplicate on normalized text (124 duplicates, 0 conflicting texts dropped) | 27,402 |

Manual labels: 81 rows relabelled in place, 18 rows added from `manual_labels.csv`.
Synthetic defamation rows added to **train only**: 296 (validation and test contain real data only).

## 3. Class distribution (stratified 70/15/15, seed 42)

| split | normal (0) | insult (1) | defamation (2) | total |
|---|---|---|---|---|
| train | 11,920 (61.2%) | 7,193 (36.9%) | 364 (1.9%) | 19,477 |
| validation | 2,554 (62.1%) | 1,541 (37.5%) | 15 (0.4%) | 4,110 |
| test | 2,555 (62.2%) | 1,542 (37.5%) | 14 (0.3%) | 4,111 |
| all (real only) | 17,029 (62.1%) | 10,276 (37.5%) | 97 (0.4%) | 27,402 |

## 4. Source x label

| source                |   defamation |   insult |   normal |
|:----------------------|-------------:|---------:|---------:|
| HateThaiSent          |            7 |      522 |     4419 |
| ThaiCyberbullyingLGBT |           53 |     8438 |     7826 |
| ThaiToxicityTweet     |           17 |     1316 |      827 |
| WisesightSentiment    |           20 |        0 |     3957 |
