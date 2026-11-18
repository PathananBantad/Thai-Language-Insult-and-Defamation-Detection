# Binary Insult vs Defamation Experiment v1

## Objective

Direct binary classification: `0 = Insult`, `1 = Defamation`. Original Normal rows are excluded. This is a preliminary risk-screening experiment, not a legal determination.

## Data source and mapping

Derived from the existing processed `train.csv`, `validation.csv`, and `test.csv`, preserving split membership and source provenance. Original labels map as `1 → Insult (0)`, `2 → Defamation (1)`, and `0 → excluded`. The derived records retain raw text, source text normalization, experiment text normalization, original label/name, stable sample ID, source row, split, source, and synthetic flag.

The current processed source files include historical HateThaiSent-attributed rows. They remain in these derived data because this experiment was specified against the existing processed splits and their expected full-split counts; no HateThaiSent candidate additions were used. The raw HateThaiSent source file and candidate experiment artifacts had previously been removed, while the original processed files were preserved.

## Exclusion procedure

The 200 unique Insult samples from the Phase 4.1 queue are excluded from Train only. Phase 5.1 is checked for the same sample ID set and matching lineage. Stable sample IDs are generated from split and physical CSV row; each exclusion is checked against source row, raw/normalized text, source, and original label. Excluded items remain recoverable in `exclusion_log.csv` and are not relabeled. Validation and Test are not filtered.

## Dataset counts

| Split | Insult (0) | Defamation (1) | Total |
|---|---:|---:|---:|
| Train before exclusion | 7,190 | 368 | 7,558 |
| Train excluded | 200 | 0 | 200 |
| Train after exclusion | 6,990 | 368 | 7,358 |
| Validation | 1,541 | 16 | 1,557 |
| Test | 1,541 | 15 | 1,556 |

Train Defamation has 296 Synthetic and 72 Real/Other samples. Validation and Test contain no Synthetic Defamation rows.

## Preprocessing

The binary experiment uses `binary_preprocess_v1`, separate from the shared API preprocessing. It applies the existing URL/emoji/mention/hashtag cleaning and deterministic evasion rules, with fuzzy keyword correction disabled. Representative tested changes include `ค-ว-ย → ควย` and `kuy → ควย`; ordinary `focus` remains unchanged. `text_raw` is retained. There is no character-offset mapping from normalized text back to raw text.

Training and the binary inference service use the same `normalize_binary_text()` function and version metadata. Existing legacy API behavior is unchanged.

## Model configurations and status

| Model | Checkpoint | Configuration | Status |
|---|---|---|---|
| ComplementNB | PyThaiNLP newmm + CountVectorizer + TF-IDF | alpha selected from 0.1, 0.3, 0.5, 1.0 by Validation macro F1; balanced sample weights | Pending final Evaluation |
| WangchanBERTa | `airesearch/wangchanberta-base-att-spm-uncased` | pinned revision; CUDA mixed precision; class-weighted loss; 3 epochs; best Validation macro F1 | Epoch 1 complete; Validation macro F1 0.6477; epochs 2–3 in progress |
| PhayaThaiBERT | [`clicknext/phayathaibert`](https://huggingface.co/clicknext/phayathaibert) base | pinned revision; Adafactor for memory; CUDA mixed precision; class-weighted loss; 3 epochs; best Validation macro F1 | Pending |

All models use `text_norm`, the `0=Insult, 1=Defamation` mapping, and Train-only fitting. Test is not used in training or checkpoint selection.

## Evaluation

Metrics are pending. Completed runs will be reported with accuracy, class precision/recall/F1/support, macro F1, weighted F1, confusion matrix, classification report, and Real/Other-only Defamation recall. Defamation has only 16 Validation and 15 Test examples; one Test error changes recall by 6.67 percentage points. No generalization claim should be made from this small Test support.

## Target identification and allegation analysis

The baseline reuses existing rule lists but only treats `@mention`, named targets, and explicit reference patterns as identified targets. Broad addressee/third-person pronouns and slur nicknames do not establish a specific target. Allegation cues are rule-based; they distinguish a factual-claim cue, an insult/opinion cue, and Unclear. No labeled NER or allegation dataset is available, so no performance metric is claimed.

## Explainability and API integration

Model explanations are not available. The service returns model probabilities and only lists raw-text spans found by the separate target/allegation rules; those spans are not model rationales. Low-confidence predictions below the fixed 0.55 screening threshold abstain; that threshold was not tuned on Test. The ComplementNB artifact loaded successfully and returned a predicted result for a mention-plus-allegation input and abstentions for ordinary and empty inputs. The new API route is opt-in via `THAI_MOD_BINARY_MODEL_DIR`; legacy routes remain unchanged. API runtime verification is pending because FastAPI is unavailable in the specified interpreter and package installation is prohibited.

## Limitations and next steps

- Confirm the intended treatment of historical HateThaiSent-attributed rows still present in the protected processed splits.
- Complete WangchanBERTa and PhayaThaiBERT runs, then evaluate only completed artifacts.
- Keep the 15-row Defamation Test limitation prominent and consider a larger independently reviewed real-only set in a future phase.
- Human review is still needed for labels and rule-based target/allegation cues.
- Do not use this screening output as a legal conclusion.
