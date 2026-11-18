# Known Issues — Binary Insult vs Defamation v1

1. **Processed source scope:** The protected original processed splits still contain rows attributed to HateThaiSent (including 346 Train Insults and 4 Train Defamations in this binary view). The new raw source and candidate work are gone, and no candidate additions were used. The dataset counts requested for this phase include these pre-existing processed rows; confirm this source policy before treating the run as the definitive dataset.
2. **Defamation imbalance:** Only 72 of 368 Train Defamation examples are Real/Other; 296 are Synthetic. Validation/Test have 16/15 Defamation rows and all are Real/Other.
3. **Label reliability:** Insult labels were inherited from source labels except for the 200 pending-review samples excluded from Train. No automatic relabeling was performed.
4. **Preprocessing:** Evasion normalization is rule-based and can still alter ambiguous spellings; fuzzy correction is disabled. Raw-to-normalized character offsets are unavailable.
5. **Target/allegation baseline:** Rules have no labeled evaluation set. A cue is not proof of an allegation or target identity. Broad pronouns are treated as Unknown.
6. **Model explanations:** No model-specific feature attribution is provided. Listed spans come only from transparent rules.
7. **API runtime:** FastAPI is missing from the specified interpreter, so the new route cannot be imported or exercised here without installing a package, which is prohibited.
8. **First WangchanBERTa attempt:** Preserved as a failed pre-training attempt with error log; no model checkpoint was saved from that attempt. A corrected attempt is in progress in a separate output directory.
