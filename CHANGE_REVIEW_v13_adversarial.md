# Change review: the five-part adversarial review

What the review asked for, what was changed, and what was deliberately left alone.

The review consolidates five adversarial passes: scientific validity, experimental design, statistics, novelty, and reproducibility. Its verdict is **major revision**, and its core judgment is that the paper is publishable as a methodological benchmark rather than as a safety qualification study. That judgment was accepted, and the manuscript had already been reframed that way before this round; what follows is the delta against that starting point.

Verification after the changes: **`verify_v13.py` — 802 checks, 1 failure**, the failure being the deliberate Ultralytics gate. **`audit_release.py` — 0 problems.**

---

## Part 1. Findings that required new computation

### 1.1 The persistence rule was a convention nobody had tested (review 3.12)

The review's sharpest methodological point. Table 13 collapses alarm-grade windows into annunciated episodes under one rule, an on-delay of three windows and an off-delay of fifteen, and the manuscript then uses the eventized view to reorder the models. The review observed that if the ordering depends on the rule, the reordering is a property of the convention.

The previous version said "the effect of that choice is bounded below" and then never bounded it. That was a promise the paper did not keep.

A new driver, `retrain/run_persistence_sweep.py`, fits the models once, computes the alarm-grade flag sequence once per model, seed and condition, and re-eventizes the same sequences at twenty-five on-delay by off-delay settings with nothing refitted. Rank agreement with the deployed rule is measured per condition by Kendall's τ and every pairwise reversal is recorded. Results in `v3_persistence_sweep.csv` and `v3_persistence_rank.json`; reported as **Table S7** and a new Supplementary Section S9.

The answer is mixed, and two of the three parts cost the paper something.

| Claim | Outcome under the sweep |
|---|---|
| Compression is 5× to 316× | Holds. |
| The single-channel reordering: the shallow tree annunciates less than the cost-weighted policy | **Holds at every one of the 25 settings.** Strictly less at 20, equal at 5, reversed at 0. |
| The cost-weighted policy annunciates more at four lost channels than at seven | Holds at 21, ties at 2, **reverses at 2**, both at an off-delay of 60 windows. Now reported with that qualification. |
| "Once eventized, every condition tested falls within or near the EEMUA reference" | **Fails.** True at the deployed rule only. At noise σ = 0.5 the grid reaches 109 episodes per hour; at σ = 0.3 the deployed rule reports **zero** annunciated episodes from every comparator while a shorter on-delay reports up to 50 per hour. |

The last row is the substantive correction of this round. A convention chosen by the authors decided whether one tested condition looked benign or looked like an alarm flood. The claim is now stated as conditional on the rule, in the body and in a new Limitations entry, **The Persistence Rule Is a Convention**.

### 1.2 A cited result had no driver (review 5.1)

Supplementary Section S8 quoted a measured cost for the metric set against accuracy alone, citing `v3_metric_cost.json`. The file existed; **nothing in the repository produced it.** For a paper whose Data and Code Availability section claims every artifact needed to reproduce it is released, that is exactly the gap the review's Review 5 is about, and the earlier release audit had not caught it because the audit checks that named files exist, not that they can be regenerated.

`retrain/run_metric_cost.py` now produces it, timing both quantities over the real 1,264-window test partition across 200 repetitions after warm-up. The regenerated figures replace the old ones in S8: **0.455 ms against 0.290 ms, a ratio of 1.57×**, previously quoted as 0.478 ms, 0.264 ms and 1.81×. The section now says the absolute figures are machine-dependent and the ratio is the transferable quantity.

---

## Part 2. Claims that outran their evidence

Each of these is a wording change, and each is gated so it cannot drift back.

**"One in eleven" (review 3.3).** The manuscript said an observed zero missed-hazard rate "is consistent with a true rate near one in eleven," which reads as an estimate. It is an upper bound. Both occurrences now read "compatible with true rates as high as about one in eleven," and the longer passage states the direction explicitly: the figure says what the data fail to rule out, and a reader who carries away "about one in eleven" as the rate has inverted the claim.

**Cross-model correlation intervals (review 3.9).** The Fisher-transform intervals on the ten-comparator Pearson correlations were reported as 95% confidence intervals. Ten comparators chosen by the authors to span inductive biases, sharing one corpus and one protocol, are not a sample from a population of models. They are now labelled **exploratory reference intervals rather than confidence intervals**, with the reason given.

**No reject option (review 2.3).** The class-disjoint experiment was described accurately but never stated the structural fact plainly. A new paragraph does: no comparator has a reject option, each is forced to emit one of its trained actions on every window of the withheld class, and the experiment therefore measures how a closed-set policy *routes* an unfamiliar hazard rather than whether it can *recognize* one. It is named a class-exclusion, forced-classification stress test, and the point that a comparator might score well by mapping the unknown analyte onto a conveniently alarming neighbor is made rather than left for a referee.

**Ten episodes are counts, not probabilities (review 2.6).** Table 14's entries are out of ten. A new paragraph gives the arithmetic: 7/10 carries a one-sided 95% lower bound on detection of 0.39, and even 10/10 establishes only that detection exceeds 0.74. Nothing in that table supports a ranking and none is asserted.

**Computational versus physical replication (reviews 3.4, 3.6, 3.8).** Three statements added to Limitations. Run-level standard deviations quantify computational variability across seeds, block positions and initializations, not device-to-device or session-to-session sampling uncertainty. Because ρ is imported from a cross-validation setting rather than identified by this design, the Bayesian correlated *t*-test is offered as a sensitivity analysis over an assumed dependence structure, not as a calibrated posterior. And the agreement between it and the paired bootstrap establishes robustness to the analysis choice, not independent confirmation, because the two analyses share every observation.

**An internal contradiction found while acting on review 2.5.** Experiment 6 said channel dropout places the dropped channel at its training-set mean; Limitations said it was "zeroing of the standardized channel features." Both are true under a training-partition standard scaler, where standardized zero *is* the training mean, but the second reads as an electrical zero, which the first explicitly says it is not. Limitations now says so in one sentence.

**Scope statement in the Conclusion.** The review's Appendix C asks for a consolidated interpretation statement. Everything in it was already in Limitations, but the Conclusion is where a reader lands. A closing paragraph now states the scope: a controlled methodological benchmark, independence-reference bounds, computational rather than physical replication, class exclusion with no reject option, dimensionless perturbations of standardized features. One stray "confidence bound" in that section was corrected to "independence-reference bound," the term used everywhere else.

---

## Part 3. Reproducibility and the repository

### 3.1 A provenance table (review, required action A.7)

The review asks for "a single provenance table mapping every headline number to commit, script, result file, seed set and split manifest." Appendix A previously carried a prose list of artifacts. It is now **Table A-1**, twenty-three rows, mapping every reported object to its driver, its stored result file, its protocol and its `make` target. The commit and package versions stay in `RUN_MANIFEST.json`, which `make_manifest.py` writes from the live interpreter rather than from the table, so a disagreement between the two is a real disagreement.

### 3.2 Deprecated drivers (review 5.6)

Four earlier drivers remain in the repository and produce no number in the paper: `run_exp1_real.py`, `run_exp3_loco.py`, `run_exp4_calibration.py`, `run_expzoo.py`. Each now carries a deprecation header, and Appendix A names them and says what differs. The honest version of that difference is narrower than the review supposed: they are not running a no-embargo split, because they call the same corrected pipeline. What is true is that their headers describe the decision network in reinforcement-learning terms, as a Dueling DQN, although it is trained by cost-weighted cross-entropy; and one of them fixes the anomaly-refit seed at zero rather than varying it with the run seed. Both are stated as such rather than dressed up.

The review's related point, that the manuscript should not call the main checkpoints DQN-trained, was already satisfied: the manuscript says the network is "trained by cost-weighted cross-entropy. Not by reinforcement learning, despite the architecture's provenance." The code headers were the remaining mismatch.

### 3.3 The finding the repository cannot answer from here

Review 5 is largely an audit of the **public GitHub repository**, and it is correct on its own terms: that repository documents five seeds, lacks the degradation, ablation, alarm-episode and block-bootstrap artifacts, and does not expose the verification script. All of those exist and have existed for several versions. **They were never pushed.** The workspace where every version since v10 was built is not a git repository, so nothing has been committed from it, and the last version to physically reach the repository folder was v10.

This is not a manuscript defect and no wording change addresses it. It is the single largest item standing between this paper and an independent verification, and it is resolved by pushing. Everything needed is staged in the repository folder with `commit_v13.bat`.

---

## Part 4. Novelty and literature

The review's verdict, **adequate incremental novelty**, was accepted. It asks for wording that claims the combination rather than the ingredients. The manuscript already carried that wording: "We did not find a prior study combining these elements on a MOX gas-detection benchmark." No change was needed.

Of the eight papers the review names as the strongest novelty pressure, four were already cited: Qu et al. (2022), Dennler et al. (2022), Ma et al. (2024) and Zhang and Zhang (2025). **Yao et al. (2024)** has been added, at the sentence describing the joint treatment of drift and the unknown class, which previously made that claim without a citation; the sentence now also says outright that the combination is established rather than open.

Four remain uncited, and deliberately so: Yao et al. (2023) in *Chemometrics and Intelligent Laboratory Systems*, Du et al. (2026) in the *Journal of Chemometrics*, Wen and Khan (2026) in *Process Safety and Environmental Protection*, and Parvez et al. (2025) in *Control Engineering Practice*. Each was located and each appears to exist, but the bibliographic metadata services needed to confirm authors, volumes and page ranges are not reachable from this environment, and a reference list is the wrong place to guess. They are listed here so they can be checked and inserted from a machine with journal access. Du et al. (2026) is the most worth adding: coverage-aware selective classification under controlled gas-sensor shift is the nearest neighbor to this paper's calibration and deferral discussion.

---

## Part 5. What was not changed, and why

Three of the review's recommendations would require data this study does not have, and the manuscript states each as a limitation rather than working around it.

**Independent sessions, devices and acquisition days.** The review is right that single-session confounding is the central experimental limitation and that class, time, device and acquisition condition are not independently replicated. There is one acquisition. This is stated in the Summary, in Limitations, and first in Future Work.

**Physically calibrated degradation.** Mapping σ and the gain shift onto manufacturer specifications or field failure data would require those specifications and a physical fault injection rig. The manuscript instead says exactly what the perturbations are: dimensionless, applied to standardized features, with the milder of the two channel-loss variants tested and the electrical-zero case untested.

**A reject-option baseline.** The review offers "relabel and/or add reject-option baseline." The relabelling is done. The baseline is a real experiment and is named as the natural extension rather than attempted, because choosing which action a rejection maps to is itself a facility decision this corpus cannot supply.

One recommendation was considered and declined. The review suggests either replacing the Bayesian correlated *t*-test with a dependence structure appropriate to the repeated-block design, or presenting it strictly as a sensitivity analysis. The second was taken. The first would mean deriving a new correlation model for a design with seed, block position and hyperparameter draw all varying, which is a methodological contribution in its own right and not one this paper is in a position to make well.

---

## Verification

| Check | Before | After |
|---|---|---|
| `verify_v13.py` | 770 checks, 1 deliberate failure | **802 checks, 1 deliberate failure** |
| `audit_release.py` | 0 problems | **0 problems** |
| New result files | — | `v3_persistence_sweep.csv`, `v3_persistence_rank.json`, regenerated `v3_metric_cost.json` |
| New drivers | — | `run_persistence_sweep.py`, `run_metric_cost.py` |

Thirty-one checks were added, one per claim introduced or corrected in this round. Among them: that the persistence sweep is reported rather than promised, and that the phrase "bounded below" no longer appears unfulfilled; that the single-channel reordering really does survive all twenty-five settings, recomputed from the sweep rather than trusted from the sentence; that the deployed rule really does report zero at noise σ = 0.3; that the metric-cost figures match the file the new driver writes; that each deprecated driver carries its header; and that the estimate phrasing of the missed-hazard bound cannot return.

The one remaining failure is the Ultralytics version string in Appendix A, unchanged from the previous round and documented in `SUBMISSION_GATE.md`.
