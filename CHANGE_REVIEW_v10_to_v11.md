# Change review: v10 → v11

Everything altered in response to the pre-submission review, with the evidence for
each change and the numbers it moved. Organized so that the two code defects come
first, because they are the reason every table changed.

**State at the close:** 717 automated checks, zero failures. All experiments
re-run under the corrected pipeline. 21 figures regenerated. 16 body tables and 3
supplementary tables, all renumbered consecutively after the split; every figure,
table and equation reference resolves.

---

## Part 1. Code defects found and fixed

Three, of which the reviewer identified two. All three were real, and all three
were in code rather than in the description of it.

### 1.1 The two-sided embargo was never implemented (reviewer 4.1)

`retrain/raw_pipeline.py:block_wise_holdout`

The manuscript claimed, and Fig. 2 drew, a 20-window embargo on each side of the
held-out block. The code placed one band before the test block and resumed
training at the index immediately after it:

```python
train_idx.extend(block[:start])
train_idx.extend(block[start + n_test + n_gap:])   # resumes at the test block's end
test_idx.extend(block[start + n_gap:start + n_gap + n_test])
```

The trailing training segment began exactly where the test block ended, so the
first post-test training window shared up to 19 of its 20 raw readings with the
last test window. The manuscript's own arithmetic was the evidence and nobody
read it: 1,581 − 316 − 20 = 1,245 per class, ×4 = 4,980, which is one band, not
two.

**Fix.** The test block is placed at `lo`, `gap` windows are reserved on each
side, and training is taken from `[0, lo−gap)` and `[lo+n_test+gap, n)`.

**Verification.** Minimum separation between any training and any test window of
the same class measured at 21 window starts, against the 20 needed for zero
shared readings, on every seed used in the paper.

**Effect.** Partition moves from 4,980 / 1,264 to **4,900 / 1,264**. Every
experiment re-run.

### 1.2 The anomaly path had test exposure (reviewer 4.2)

`retrain/raw_pipeline.py:build_anomaly_model`

The reviewer identified the scaler. There were two exposures, not one:

1. `StandardScaler` was fitted on **every NoGas row in the corpus**, before any
   partition existed.
2. The autoencoder checkpoint it then loaded had **itself been trained on every
   NoGas row**.

The reviewer's alternative remedy, treating it as an externally pretrained frozen
extractor with demonstrably disjoint training data, was unavailable: the training
data are the whole corpus.

**Fix.** New `partition_anomaly()` fits the input scaler and trains the
autoencoder from scratch on the nominal windows of each training partition alone,
then scores every window under that model. Called from `prepare()`, so every
driver that forms a partition is covered, and called explicitly in both
class-disjoint drivers, where the fitting set is the three training classes.
Memoized per distinct partition. `build_anomaly_model` is retained, marked
deprecated, and used only for regenerating pre-correction numbers.

**Effect — the largest single change in the paper:**

| Quantity | v10 (leaked) | v11 (corrected) |
|---|---|---|
| Held-out anomaly ROC-AUC | 0.9928 ± 0.0053 | **0.9251 ± 0.0618** |
| Held-out TPR | 0.7363 ± 0.0949 | 0.8015 ± 0.1205 |
| Held-out FPR | **0.0000 ± 0.0000** | **0.1101 ± 0.0992** |

Seven points of discrimination and the entire zero-false-positive claim were
artifacts of the anomaly model having seen the nominal windows it was later
scored on. The spread across seeds widened from ±0.005 to ±0.062, running 0.8228
to 0.9898, which the pooled figure had concealed.

This correction propagates into every learned comparator, because the anomaly
score is one of the twenty-two inputs to all of them.

### 1.3 CUSUM carried its statistic across class-block boundaries (found while answering a minor comment)

`retrain/new_baselines.py:CusumDetector`

The reviewer asked us to *state* the reset semantics. Writing them down revealed
there were none worth stating: `S_t` was zeroed once per `predict` call, so on a
test partition ordered NoGas → Smoke → Mixture → Perfume the statistic
accumulated across every boundary and the class following a hazardous one
inherited an already-elevated statistic.

**Fix.** `S_t` resets at every change of class block. Never carried across a
boundary, the embargo band, or between partitions.

**Effect.** Accuracy 0.4994 → 0.4535. Missed-hazard 0.0013 → 0.0000. Escalation
0.9987 → 1.0000. False-alarm rate **0 → 0.1861**, which is the honest cost of the
escalation: one threshold and no class structure buys full escalation by alarming
on roughly one clean window in five.

---

## Part 2. How the headline numbers moved

| Claim | v10 | v11 | Direction |
|---|---|---|---|
| Random-split inflation, temporal models | 1.66–4.08 pts | **3.23–5.57 pts** | Worse leakage than reported |
| Protocol effect vs model spread (temporal) | 3.09 vs 2.48 | **4.04 vs 2.42** | Claim strengthens, ×1.7 |
| Protocol effect vs model spread (all six) | 2.45 vs 4.72 (failed) | **3.24 vs 3.28** (tie) | Still restricted to temporal models |
| Ten-comparator accuracy spread | 5.66 pts | **4.43 pts** | Tighter |
| Cost-weighted policy accuracy rank | 6th | **5th** | — |
| Equivalence, unweighted network | 0.960 | **0.992** | Stronger |
| Class-disjoint miss separation | >2 orders | **factor of 40** | Smaller but same conclusion |
| Worst class-disjoint model | MLP, 9.49% | **MLP, 6.41%** | Same model, same reversal |
| Its accuracy rank | 5th of 10 | **3rd of 10** | Reversal sharper |
| Mixture escalation spread | ×198 | **×203** | — |
| Anomaly-input dependence, GBM | 60.6% | **64.2%** | — |
| GBM miss with anomaly forced to zero | 0.3405 | **0.4203** | Worse |
| Cost-ratio accuracy peak | 6:1 | **1:1** | Negative result strengthens |
| Blocked-validation selection | 2:1 | **6:1** | 8:1 still not selected |
| Lowest calibration ECE | Deep ensemble | **MC dropout** | Ordering not stable → don't read one |
| Anomaly-only ablation | 75.1% | **65.8%** | — |
| Removing per-sensor σ | −0.11 pts | **+0.40 pts** | State is not minimal |

Two conclusions are unchanged and now rest on corrected numbers: accuracy ranking
does not predict class-disjoint safety behavior, and missed-hazard rate alone
clears models that notify nobody.

---

## Part 3. New analyses added

### 3.1 Eventized alarm burden (reviewer 4.9) — new Table 16

`retrain/run_alarm_episodes.py`

Contiguous alarm-grade windows on clean test windows collapsed into annunciated
episodes under a stated rule: 3-window on-delay (6 s persistence), 15-window
off-delay (30 s). Stated, not tuned.

This is the addition that most changes the engineering reading, and it partly
corrects us:

- Compression runs **5.5× to 316×**, so the hourly indication figures were one to
  two orders of magnitude above the rate an operator would be interrupted at.
- Once eventized, every condition tested falls within or near the EEMUA envelope,
  where on indications several sat two orders of magnitude above it. Fig. 16 is
  now framed as indication pressure, and the Experiment 6 text is reconciled.
- **The ordering reverses.** At single-channel loss the shallow tree produces 360
  indications/h against the cost-weighted policy's 231, yet annunciates 1.14
  times/h against 5.70. The tree fails into one standing alarm; the policy
  chatters. The same distinction appears within one model: the cost-weighted
  policy produces more episodes at four lost channels (10.25/h) than at seven
  (5.70/h) despite five times fewer indications.

Nothing else in the paper distinguishes a standing alarm from a chattering one.

### 3.2 ROPE sensitivity (reviewer 4.6) — new Table 10

`retrain/run_rope_sensitivity.py`

Both analyses re-run at ±0.5, ±1 and ±2 accuracy points. The result is
uncomfortable and is reported as such: **7 of 10 comparisons change their most
probable outcome** across that range. Only two survive the full sweep — the
unweighted network is equivalent at every width, and the decision tree and
ordinal objective are worse at every width. Everything between is
width-dependent, and the manuscript now says so.

The ROPE width carries considerably more than ρ does, which inverts the emphasis
of the previous version.

### 3.3 Partition-local per-class reconstruction error

`retrain/run_anomaly_by_class.py`. The descriptive "does the anomaly score track
hazard" figure had still been computed with the corpus-wide autoencoder. Now
computed under the partition-local model, averaged over five partitions: clean
air 0.023, vapor 0.107, mixture 80.6, smoke 164.7.

---

## Part 4. Framing and text changes

| Reviewer comment | What changed |
|---|---|
| 4.4 Runtime ≠ evaluated object | New subsection *What Is Evaluated, and What Is Not* names the **ARXIS sensor-policy core**, states the runtime's guardrails (anomaly override, thermal promotion) and why excluding them is what makes models separable, and says the full-runtime experiment is not in this paper. Repeated in the README |
| 4.5 "Exact" Clopper-Pearson | Renamed **independence-reference binomial upper bound** in the Summary, contributions, method, nomenclature, abbreviations and every table caption reporting one. Method explains why the binomial model does not hold; Table 12's caption carries the warning where the numbers are; the class-disjoint case is called out as the worst (1,581 overlapping windows from one episode) |
| 4.6 Bootstrap framing | Rewritten: the two analyses share every observation, 30 runs over one session are not 30 physical experiments, the agreement establishes only robustness to the assumed ρ |
| 4.7 Open-set and alarm analytics | Two new subsections. *Unknown Gases Are Not a New Problem* (Qu et al. 2022; Ma et al. 2024) draws the line: open-set work is scored on rejection accuracy, none asks what action an unfamiliar hazard produces, and our experiment is deliberately not an open-set method. *Alarm Analytics Is Also an Established Field* (Wang, J. et al. 2016; EEMUA 2024; ISA 2016) draws the other: that work configures an existing alarm system, this screens a model before it becomes a source of alarms |
| 4.8 Synthetic action ladder | Action 4 renamed ***Recommend ESD assessment***. New passage before the metric definitions on the ladder's constructed status; new limitation *The Action Ladder Is a Construct*. Summary no longer says two quantities "decide whether a monitoring system is safe to install" |
| 4.10 Length and tone | Supporting-information file created carrying the perception component, explanation component, cost-asymmetry experiment, calibration experiment and full comparator specifications. Every self-commenting construction the reviewer named was located and rewritten |

### Minor comments

| Comment | Action |
|---|---|
| "Five-quantity safety metric set" | → "five-quantity evaluation set" |
| CQL needs detail | Supplementary S5: behavior policy, coverage, reward and cost structure, γ = 0.99, CQL(H) at α = 1.0, one-step successor, and the absence of episodes and terminal states stated plainly |
| CUSUM reset semantics | Specified **and fixed** (§1.3) |
| Where channel zeroing is applied | New paragraph: all three perturbation families act on the **standardized** state, so a dropped channel sits at its training-set mean, not an electrical zero — the milder of the two faults. The untested case named |
| Other fault modes | Named as untested: stuck-at-last-value, bias, saturation, NaN, slow drift |
| Small figure labels | Base type raised, annotations floored at 8 pt, widest figures 13.4 → 11.5 in, 300 dpi. **A first, more aggressive attempt overlapped labels and was reverted**; the current settings were checked by rendering |
| Wide tables | Model-name abbreviations retained where they fit; Table 16 uses short names with a caption key |
| Perception/explanation space | Both moved to supporting information |
| "Safe" / "fit for service" | Four passages rewritten; fitness now explicitly a facility decision no benchmark settles |
| Standards editions | EEMUA → **revised fourth edition, 2024**. Caveat: edition and contents verified, the paywalled numeric envelope not re-checked |

---

## Part 5. Repository artifacts (reviewer 4.3, Section 6)

| File | Purpose |
|---|---|
| `Makefile` | One target per manuscript table and figure, plus `all`, `verify`, `manifest`, `release-check` |
| `make_manifest.py` | `RUN_MANIFEST.json` (commit, host, CPU count, Python and package versions, seeds, protocol constants, SHA-256 of every driver, result file and figure) and `requirements.lock`, both from the live interpreter |
| `verify_v11.py` | 717 checks: every table value against stored results, SPE house style, a cross-reference proof pass, a coverage audit. Reads both the manuscript and the supporting information |
| `README.md` | Rewritten. Stale RL terminology, old class-disjoint numbers and the 8:1 language removed; the paper-versus-runtime distinction on the first screen |
| `RELEASE_CHECKLIST.md` | Freeze, tag, Zenodo DOI, cold-clone test |
| `RESPONSE_TO_REVIEWER.md` | Point-by-point response |

---

## Part 6. Numbering and structural integrity

The supplementary split removed Table 7, Table 13 and Table 16 and Figs. 5, 14
and 19 from the body, leaving gaps, and the new ROPE table collided with an
existing Table 11. All body objects were renumbered consecutively and every
in-text reference rewritten.

**Verified after renumbering:** table captions 1–16 with no dangling or uncited
references; figure captions 1–17 likewise; equations 1–12 sequential, with the
ordinal objective moved to the supplement as Eq. S1 and the duty-cycle conversion
renumbered from 13 to 12.

---

## Part 7. Text integrity, re-checked

- n-gram overlap against the structural model paper: **0.00% at 5, 8 and 12
  grams**, longest shared run 0 words.
- Stylometrics against the same human-written comparator: hedges 1.5 per 1k
  (human 2.5), repeated trigrams 17.5% (human 21.0), lexical diversity 0.547
  (human 0.504), semicolons 1.4 (human 2.8). Every feature at or beyond the human
  paper on the safe side.

---

## Part 8. What remains open

Three items, all author-side, all in `RELEASE_CHECKLIST.md`.

1. **Tag and archive the release.** This workspace is not a git checkout. The
   Zenodo DOI must be reserved before the manuscript's final pass, because it
   goes in Data and Code Availability.
2. **The Ultralytics version** for the thermal classifier. The manifest cannot
   observe it because that training happened outside this pipeline.
3. **The EEMUA 191 fourth-edition numeric envelope.** Edition verified, figures
   paywalled.

Two scientific gaps are stated in the manuscript rather than closed:

- **The thermal split is still randomized.** The images are not in this
  workspace, so the block-wise re-split is outstanding. The manuscript states the
  asymmetry and rests no result on the perception number.
- **No block-based uncertainty treatment.** The bounds are labeled for what they
  are and the improvement is named as the one this design most needs, but it is
  not implemented.
