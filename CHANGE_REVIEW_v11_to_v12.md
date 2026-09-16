# Change review: v11 → v12 (and the third-pass blockers)

Response to the second red-team review, which ran five passes over the
manuscript, the code, the novelty claim, the statistics, and a final 53-item
must-fix list. Recommendation received: Major Revision, 68/100.

**State at the close:** 751 automated checks, one deliberate failure — the Ultralytics version gate described in `SUBMISSION_GATE.md`, which blocks submission until the real string is supplied. Everything else passes. Two new experiments.
One further code defect found and fixed. Body cut from 18,491 to 15,913 prose
words with the overflow moved to a supporting-information file.

---

## Part 1. The credibility blockers (A1–A5)

The reviewer was right that these were the most damaging findings, because
Appendix A claims automated verification of every quoted value and these had
survived it. Each is now checked by a rule that fails loudly.

| Finding | Was | Now | Root cause |
|---|---|---|---|
| A1. Alarm burden at one lost channel | 117 in two places, 128 in the table | **128 everywhere**, and the hourly figure verified as exactly 1.8× it | 117 was a pre-correction value that three occurrences kept |
| A2. Anomaly probe | prose 0.0092→0.4203, figure caption 0.0098→0.3405 | **0.0092→0.4203 in both** | the caption string lived in the figure script and was never regenerated |
| A3. Temporal model spread | 2.48 in the Conclusion, 2.42 in Results | **2.42** | stale Conclusion |
| A4. Escalation ratio | "a factor of about 203" from displayed values reading 200 | **"about 200"**, with the unrounded minimum 0.0049 now displayed | false precision |
| A5. "Factor of 40" | attributed to missed-hazard *rate* | **"a factor of 40 in the 95% independence-reference upper bound, 0.19% against 7.49%"** | the 40 was a ratio of bounds, not of rates |

The verifier now checks all five, including that the superseded 117 and 0.3405
cannot reappear anywhere.

---

## Part 2. A third code defect, found while answering a minor comment

The reviewer asked (A12) what the calibration bootstrap resamples, and whether
the calibration split is temporally separated. Both answers were bad.

**The calibration bootstrap resampled individual windows i.i.d.** — exactly the
independence assumption this paper exists to criticize, applied inside the
paper's own analysis. Replaced with a moving-block bootstrap over blocks of 20
windows.

**The calibration split was carved with `rng.permutation`** — a uniformly random
subset of the training partition, over windows overlapping by 19 of 20 raw rows.
The temperature was therefore fitted on windows nearly identical to the ones it
was fitted against. Replaced with a contiguous per-class band with the same
20-window embargo used for the outer holdout.

The correction reverses the previous finding. Under the leaky split the fitted
temperature was consistently below 1, and the manuscript concluded the network
was under-confident. Under a properly separated band it is above 1: the network
is over-confident, which is the ordinary result the literature predicts and the
opposite of what the paper previously reported.

---

## Part 3. Two new experiments

### 3.1 Uncertainty that does not assume independent windows (Tier 1 item 4)

`retrain/run_block_bootstrap.py`. The reviewer's preferred remedy was a block or
episode bootstrap. An episode bootstrap is impossible on this corpus, since each
class was acquired in one continuous run and there is one episode per class; that
is stated rather than worked around. What is implemented is a moving-block
bootstrap inside each test partition, block lengths swept at 20, 100 and 316
windows, plus the quantity that actually carries the information for zero counts:

**Only 31 of 632 hazardous windows and 15 of 316 clean windows have pairwise
disjoint raw support.** Evaluating the bound on those counts rather than the
nominal ones:

| Quantity | Nominal | Disjoint-support |
|---|---|---|
| Bound on a zero, hazardous windows | 0.47% | **9.21%** |
| Bound on a zero, clean windows | 0.94% | **18.10%** |
| Alarm burden a clean zero supports | 9.4 per 1,000 | **181 per 1,000** |
| Bound on a zero, class-disjoint | 0.19% | **3.72%** |

A factor of about twenty. The manuscript now states the consequence directly:
**on this corpus an observed zero missed-hazard rate is consistent with a true
rate near one in eleven**, and no experiment here can distinguish a genuinely safe
model from a lucky one at the resolution the window counts suggest. That is the
strongest argument in the paper for acquiring more sessions, and it is now made
in the paper rather than left for a referee to make.

### 3.2 Episode-level hazard metrics and detection latency (Tier 1 item 5)

`retrain/run_episode_metrics.py`. The reviewer's Major Concern 25 was that false
alarms were eventized while hazards stayed at the window level. They now receive
the same treatment: each contiguous run of same-class windows is one hazard
episode, detected if any window draws an alarm-grade action, sustained if three
consecutive windows do.

**The null result first.** In distribution, and under drift and noise at every
severity tested, every model detects every hazard episode and every detection
latency is zero. The metric cannot discriminate there at all, because each class
was recorded as one steady-state run with no onset transient. **No detection-latency
claim can be made from this corpus, and the manuscript now says so explicitly**
rather than leaving latency unmentioned, which was the reviewer's Major Concern 24.

**Under channel loss it discriminates, and it disagrees with the window-level
column.** At total sensor loss the perceptron's window-level missed-hazard rate of
0.5000 corresponds to five of ten hazard episodes never drawing any response.
Gradient boosting is sharper: a window rate of 0.1953 sounds like partial
degradation, but **three of ten hazard episodes go entirely undetected**, and it
already loses one episode at a single lost channel where its window rate of
0.0516 would not obviously fail a review. The window rate understates how many
distinct events pass unnoticed.

---

## Part 4. Reframing (A6, A7, Tier 1 items 1–3, 7–8)

| Item | Change |
|---|---|
| Title | Now **"A Safety-Relevant Evaluation Protocol for Learned Gas-Monitoring Models: Class-Disjoint Hazards and Graded Sensor Perturbation on a Laboratory MOX Benchmark."** "Distribution shift" and "sensor degradation" both overstated what the experiments contain |
| Summary | Rewritten to the structure the reviewer asked for: problem, contribution, three results, limits, implication. It now opens by naming the contribution as an evaluation methodology demonstrated on a laboratory testbed, and carries the three scope limits plus the one-in-eleven uncertainty figure |
| Eq. 2 | A set-off declaration immediately before it: the action labels are constructed evaluation targets, not facility alarm requirements, not an SIS specification, and derived from no concentration, LEL fraction, exposure limit or consequence model |
| Experiment 3 | Explicitly "class exclusion within a single acquisition session, not demonstrated domain shift", with the reason spelled out |
| RQ4 | Now asks about "synthetic sensor-fault perturbations of the decision state", and says physical sensor failures are not tested |
| Thermal branch | Named as outside the evaluated sensor-policy core at first mention, with "multimodal describes the system, not the object these experiments characterize" |
| Keywords | Dropped "distribution shift" and "cost-sensitive learning"; now five, within SPE's limit |

---

## Part 5. The A8–A53 list

| Item | Change |
|---|---|
| A8, A29–A31 | "Fail-loud"/"fail-silent" replaced with alarm-biased / silent-failure, declared study-specific behavioral patterns carrying no IEC 61511 meaning. The "wrong high-severity response is almost as good" sentence rewritten |
| A11 | Every headline bound now reads "95% independence-reference upper bound", not "95% upper bound" |
| A13 | Hazardous-window calibration stated as an observed figure inheriting the same window dependence, not a population guarantee |
| A14 | The ROPE's "procurement decision" justification removed; it is now declared a study-specific choice with the sweep doing the work |
| A15 | "Survives every sensitivity check" narrowed to "across the whole of the evaluated ρ and equivalence-width ranges" |
| A16, A17 | The Limitations sentence claiming every comparison is ρ-stable contradicted Table 9 and the Results text. Corrected: eight of ten are stable, the random forest and gradient boosting are not, and both are named |
| A18 | "The two that escalate fully on both" → "the two strongest escalators", with the 0.983 value given |
| A19 | "None produces a single high-severity false alarm" scoped to the Table 8 comparator set, with CUSUM's 0.1861 named as the exception |
| A21 | "Four-stage" eliminated; the pipeline is five-component throughout |
| A25, A26 | "Only the second is novel" removed; the engineering contribution stated positively as a pre-deployment screen |
| A27 | The repeated "what accuracy does not measure" refrain cut to one occurrence |
| A32–A35 | Sweeping openers narrowed: "rarely preceded by silence", "hardly at all", "the benchmark has saturated", "the only route to an external-validity claim" |
| A36 | The conformal-prediction claim conditioned on the exchangeability assumptions it requires, which this paper's dependence structure puts in question |
| A37, A38 | The "block bootstrap over episodes" suggestion corrected, since there is one episode; and "the partition is the experimental unit" qualified as right for this analysis, wrong for physical inference |
| A39 | Five seeds now "a descriptive robustness check", not "adequate" |
| A40 | Every table caption names its source of variability |
| A41 | "Expected cost" → "study-defined expected cost", marked study-internal |
| A42, A43, A44 | Ordinal result scoped to this matrix; "deployable decision tree" → "interpretable decision-tree comparator"; "comparators a facility would recognize" → "interpretable and conventional process-monitoring comparators" |
| A45–A47 | The alarm hierarchy is now announced where the metric is defined, before any number is quoted: indication pressure, eventized burden, operational interpretation. EEMUA marked contextual rather than a compliance test |
| A48 | The "conservative relative to typical process practice" claim removed as uncited |
| A49 | Summary restructured and shortened |
| A51 | Keywords revised |

---

## Part 6. Length

The first review asked for a 25–30% cut; the second added required material.
Both were addressed by moving rather than deleting.

| | v11 | v12 |
|---|---|---|
| Body prose | 18,491 | **15,913** |
| Supporting information | 2,488 | **~6,000** |
| Body tables | 17 | **14** |
| Body figures | 17 | **15** |
| Supplementary sections | 6 | **9** |

Moved to supporting information this round: the ρ and ROPE sensitivity sweeps
with their tables, the decision-state ablation and upstream-input probe, and the
protocol adoption-cost analysis. Each leaves a result summary and a pointer in
the body.

The body remains longer than the model paper, at 15,913 against 11,229 prose
words. It carries seven experiments, and the two referee passes required
substantial additions. That gap is a judgment call rather than an oversight, and
it is the one item in this response where we have not fully met a reviewer
request.

---

## Part 7. What still cannot be fixed from the data

Stated in the manuscript, not worked around.

1. **No detection latency evidence.** One steady-state run per class means no
   onsets. A corpus with instrumented releases is required.
2. **No episode-level replication.** One acquisition episode per class means no
   episode bootstrap and no physical replicate.
3. **The action ladder remains constructed.** No facility alarm philosophy exists
   for these analytes, so every metric on it measures agreement with a
   researcher-imposed target.
4. **The analytes are still surrogates.** No hydrocarbon, one device, one session.
5. **The thermal split is still randomized.** The images are not in this
   workspace.

Author-side, from the first review and still open: tag and archive the release
for a DOI, recover the Ultralytics version, and check the paywalled EEMUA 191
fourth-edition envelope.


---

## Part 8. Third-pass blockers

A further review pass found four blockers and six wording items after the v12
draft. All are addressed; each blocker now has a verifier gate so it cannot
recur.

| Blocker | Finding | Resolution |
|---|---|---|
| 1. "Factor of 43" | The reviewer's arithmetic was right and mine was wrong. 0.4203/0.0092 = **45.68** from displayed values, **45.79** unrounded. The 43 came from a superseded baseline | Restated as **"from 0.0092 to 0.4203, a factor of about 46"**. The gate now checks the exact ratio *and* the ratio recomputed from the two displayed values, at ±0.8, so a stale figure cannot pass |
| 2. Ultralytics version | The appendix contained a note telling its own authors an item was unfinished, inside the appendix that claims machine-checked values | Sentence removed. Replaced with a placeholder that **fails verification**, so the manuscript cannot pass its own check until the value is real. See `SUBMISSION_GATE.md` |
| 3. Availability wording | "result files behind Tables 7 through 13" implied reproducibility stopped before the two tables carrying the new contribution | Rewritten to cover every table and figure in both documents explicitly, and to name the Makefile, manifest and verifier |
| 4. "Every number" claim | Untrustworthy until run against the final file | The full sequence was run in order: manuscript saved, tables regenerated, figures regenerated, `verify_v12.py` run against that exact file. 751 checks, one deliberate gate |

| Wording item | Resolution |
|---|---|
| "the worst in the field" | → "the worst of the evaluated models on the held-out hazardous class". Gate: `"in the field"` must not appear |
| Opening sentence | "far more abnormal indications in a week than any operator can act on" → "can generate sustained streams of abnormal indications that contribute to operator alarm burden", now cited |
| "cheap to make and worth measuring" | Deleted |
| "One objection deserves an answer" / "One counter-reading deserves an answer" | → neutral technical framing |
| Five-level vs four-action | Stated once, in the Methods under Table 4, and removed from the contributions, the Discussion and the Limitations |
| Nomenclature | *U* now reads "one-sided independence-reference binomial upper bound", matching the methodology. *n* split into **N_test**, **N_opp** and **N_run**, and Eqs. 4, 10 and 11 updated to match |

### Two things this pass confirms were already fixed

The 117/128 contradiction is resolved: the single-channel figure is 128
everywhere, and the eventized table's 231.3/h is exactly 1.8 × 128.5. The
remaining "117" is the random forest's compression factor at total sensor loss,
a different quantity. And 2.48 no longer appears anywhere; the spread is 2.42.
