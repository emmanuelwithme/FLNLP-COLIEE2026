# ChunkAgg Version and Experiment History

[中文](EXPERIMENT_HISTORY.md) | [v5 guide](README.en.md) | [v6 guide](../modernBert-fp-chunkAgg-v6/README.en.md)

This document records the ChunkAgg history that can be verified as of 2026-10-20 from Git, checkpoint `trainer_state.json` files, saved adaptive-negative JSON files, and the current implementation. It intentionally does not invent missing version history.

## Version and metric conventions

Repository release labels such as `[Ver 4.4]` and `[Ver 5.0]` are not ChunkAgg architecture versions. No independently identifiable v1/v2 artifacts survive. The first trained run is therefore called `legacy (v3-era)`: its input has the special-token defect later associated with the v3 input contract, but its checkpoint has no explicit architecture-version metadata.

The scores below are the best epoch-level full scoped valid-retrieval micro top-5 F1 values saved by each run. They are historical training metrics, not fresh canonical `eval.py` results. Re-run the current inference, ranking, and evaluation pipeline before using them in a paper or final report.

| Run | Main change | Rows/epoch | Best epoch / checkpoint | Fused F1 | Mean F1 | First F1 | Change from prior run |
|---|---|---:|---|---:|---:|---:|---:|
| legacy (v3-era) | Initial 1–3 chunk model with a learned `[DOC]` fusion token | 6573 | 5 / `checkpoint-8220` | 0.065735 | not logged | not logged | — |
| v4 | Corrected chunk input; added residual/evidence/ESS while fusion remained unbounded | 6262 | 4 / `checkpoint-6264` | 0.092577 | 0.119419 | 0.122706 | +0.026842 (+40.83%) |
| v5 | Added a bounded/zero-init fusion contract and content filter on top of v4 | 6257 | 7 / `checkpoint-10955` | **0.145365** | 0.138782 | 0.140428 | **+0.052787 (+57.02%)** |
| v6 | Kept the v5 network/loss; fixed `softmax(score/1.0)`; restored eligible out-of-scope positives | 6568 | 10 / `checkpoint-16420` | 0.129457 | 0.119583 | 0.124520 | -0.015908 (-10.94%) |
| single-chunk adaptive emptyExclude | Reference, not part of the ChunkAgg sequence | 6573 | 7 / `checkpoint-11508` | 0.164519 | n/a | n/a | 0.019154 above v5 |

v5 is the best saved ChunkAgg run, although it remains 0.019154 F1 below the historical single-chunk baseline. v6 changed both negative sampling and positive eligibility, so its regression cannot be attributed to either change in isolation.

## What each revision taught us

### Legacy (v3-era)

The initial model encoded up to three 4096-token chunks, projected each chunk `[CLS]`, and fused them with a standalone learned `[DOC]` token and one transformer block. It kept the first three chunks and used document-level contrastive loss. Adaptive negatives followed full-scope `softmax(score / 1.0)` sampling.

ModernBERT's fast-tokenizer path was later found not to insert `[CLS]/[SEP]` through the Python `build_inputs_with_special_tokens()` call used by this implementation. The run nevertheless proved that the pipeline could train end to end, but peaked at only 0.065735 and degraded after epoch 5.

Conclusion: more chunks and fusion capacity are not sufficient. The token contract must be tested explicitly, and an initially content-free `[DOC]` representation is fragile.

### v4

v4 corrected the explicit `[CLS] + body + [SEP]` chunk contract. Its checkpoint keys and trainer logs confirm that residual projection, a fusion output, chunk-evidence loss, target-ESS sampling, and fused/mean/first ablations were already present; these were not introduced by v5. The learned fusion path could still overwrite the stable representation without a residual bound. v4 also required positives to belong to the query scope, removing 333 out-of-scope positive pairs and leaving 6262 rows.

Fused F1 improved to 0.092577, but mean pooling reached 0.119419 and first-chunk retrieval reached 0.122706 at the same checkpoint.

Conclusion: the input fix helped, but learned fusion hurt validation retrieval. Every future run must report fused/mean/first ablations together.

### v5

v5 retained v4's evidence loss and ESS curriculum, formalized the residual projector, normalized mean base, and mean-seeded `[DOC]` as its architecture contract, used conservative zero-output initialization, smoothly capped the applied fusion residual at 0.5, added a raw-residual penalty, and introduced a 100-non-whitespace-character content filter. Positives still had to be in scope, leaving 6257 rows.

Fused F1 reached 0.145365 and, for the first time, beat both mean (by 0.006583) and first (by 0.004937).

Conclusion: conservative bounded fusion is the strongest architectural direction so far. However, v5 also formalized initialization behavior and added a content filter, so the complete gain cannot be attributed to the cap alone. The peak at epoch 7 followed by oscillation confirms that the best checkpoint, not the last epoch, must be selected.

### v6

v6 retained the v5 model, loss, content filter, and 15 unique negatives, but replaced ESS calibration with fixed `softmax(score / 1.0)`. It also restored the baseline positive policy: an eligible labeled positive remains a training target even outside scope. Scope constrains only negatives and ranking. This produces 6568 rows, including 311 eligible out-of-scope positives.

Its best fused F1 was 0.129457, 0.015908 below v5, while still beating its own mean and first ablations.

Conclusion: the complete fixed-temperature v6 contract did not beat v5, but this was not a strict one-variable ablation because positive eligibility changed too. The next informative run should combine v5 ESS sampling with v6 positive eligibility.

## Did the 100-character content filter help?

The filter **did execute and change the data flow**, but no existing one-variable experiment proves that it improved F1. Both v5 and v6 use it; v4 lacks it but also has a different fusion contract, while the single-chunk baseline differs in model and training procedure. Their scores therefore cannot isolate the filter's causal effect.

Its measurable effects are:

- It excludes only 5 of 7708 `processed` documents (0.065%): `015738(0)`, `039404(0)`, `058076(10)`, `092745(10)`, and `085084(11)`, where parentheses show non-whitespace character counts.
- It removes 5 of 6573 otherwise constructible training pairs (0.076%): three positives belonging to short query `085084`, plus one pair each whose positive is `039404` or `058076`.
- These five documents collectively occupy 6533 candidate positions across 1445 query scopes. Although the document count is tiny, filtering them can prevent blank or near-blank embeddings from contaminating adaptive-negative sampling or top-5 ranking for many queries.
- It makes 3 of 1656 valid gold pairs (0.181%) explicitly unreachable: two labels for blank query `015738` and one label whose near-blank candidate is `092745`. Those documents contain almost no usable retrieval evidence.

The current evidence therefore supports treating the threshold as a **data-integrity and training-stability guardrail**, not as a proven score improvement. It may prevent pathological embeddings, but v5's +0.052787 gain cannot be credited to it. A causal ablation should hold v5 initialization, sampling, seed, and all other settings fixed while comparing `min_chars=0`, `1`, and `100`, and should report filtered gold counts, negative-sampling incidence, and canonical F1.

## Meaning of `emptyExclude` and positive eligibility

`scopeFilteredRaw_plus0_emptyExclude` describes scope construction only: years come from raw files; a known-year query admits candidates no later than the query year; unknown-year queries receive an empty scope; and unknown-year candidates are excluded from known-year query scopes. It does not determine whether an out-of-scope gold positive is eligible for training.

| Pipeline | Eligible out-of-scope positive | Empty-scope query | Retained pairs |
|---|---|---|---:|
| single-chunk adaptive emptyExclude | **kept** | skipped because no scores/negative pool exist | 6573 / 6595 |
| legacy ChunkAgg | **kept** | skipped | 6573 / 6595 |
| ChunkAgg v4 | excluded | skipped | 6262 / 6595 |
| ChunkAgg v5 | excluded, then content-filtered | skipped | 6257 / 6595 |
| ChunkAgg v6 | **kept, then content-filtered** | skipped | 6568 / 6595 |

Therefore, single-chunk adaptive emptyExclude does keep out-of-scope positives. Its saved epoch-7 data contains 6573 unique `(query, positive)` rows, including all 311 out-of-scope positives available to non-empty-scope queries. The 22 missing label pairs belong to six empty-scope queries and cannot form contrastive rows because no scoped negatives exist.

The recommended long-term policy is:

```text
gold label + existing document + content eligibility => eligible positive
scope - self - all labeled positives                 => negative/ranking pool
empty scope                                           => no contrastive row, but keep gold in evaluation
```

## Guidance for the next experiments

1. Run v5 model/loss plus v5 ESS sampling plus v6 positive eligibility, changing no other major factor.
2. Never treat an out-of-scope labeled positive as a negative; keep content quality and year-scope rules separate.
3. Record architecture, positive eligibility, sampling policy, and corpus source independently in a run manifest.
4. Always report fused, mean, and first retrieval together.
5. Select checkpoints by full-denominator retrieval F1, not contrastive `eval_loss/acc` or the last epoch.
6. Repeat important comparisons across seeds and generate fresh canonical inference/ranking/`eval.py` results before publication.
