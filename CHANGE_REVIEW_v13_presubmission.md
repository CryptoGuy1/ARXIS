# Change review: the complete must-fix pre-submission plan

The plan consolidated five audits into 51 numbered items across three tiers, with a verdict of **DO NOT SUBMIT YET**. What follows is what changed, what did not, and why.

Verification after this round: **`verify_v13.py` — 838 checks, 1 failure**, the failure being the deliberate Ultralytics gate. **`audit_release.py` — 0 problems.** The check count was 802 before this round.

---

## The finding that changed the paper's numbers

### Item A1. Table 10's bounds were computed from an average, not from a count

This is the one that mattered, and the plan was right about it in a way that is worth spelling out.

The manuscript says Eq. 10 is evaluated with the partition as the unit and the worst case across partitions reported. What the code actually did was take the mean missed-hazard **rate** across five seeds, multiply by 1,581, round to an integer, and feed that pseudo-count to Clopper-Pearson. That is the count an *average* partition produced. Where the seeds agree the two are identical. Where they disagree the averaged version is much smaller, and on this corpus they disagree badly.

`retrain/run_final.py` now writes `v3_loco_perseed.csv`, the integer miss count each partition actually produced. The counts are not close to uniform:

| Held-out class, model | Misses per partition | Old bound | Worst-partition bound |
|---|---|---|---|
| Smoke, multilayer perceptron | 4, 23, 30, 216, 234 | 7.49% | **16.35%** |
| Mixture, shallow decision tree | 0, 0, 489, 489, 988 | 26.71% | **64.51%** |
| Mixture, k-nearest neighbors | 24, 28, 31, 38, 80 | 3.28% | **6.06%** |
| Mixture, multilayer perceptron | 0, 0, 0, 24, 54 | 1.53% | **4.27%** |
| Smoke, k-nearest neighbors | 8, 12, 15, 17, 43 | 1.76% | **3.49%** |

The shallow decision tree is the clearest case. Its average missed-hazard rate of 0.2487 reads as a model that is uniformly mediocre. The counts say something quite different: flawless on two partitions, catastrophic on three. An average is a poor summary of behavior like that, so **Table 10 now carries the per-partition counts as a column of their own** alongside the corrected bound, and the caption states that bounds come from counts and never from an averaged rate.

`retrain/run_partition_bounds.py` was carrying the same defect and is corrected the same way. It now also prints the old averaged-pseudocount figure beside the new one, so the size of the correction is visible in the artifact rather than only in this document.

**Dependent claims, all updated (items A2, A3):**

- The headline separation was "a factor of 40" (7.49 / 0.19). It is now **a factor of about 86** (16.35 / 0.19), gated against both the exact ratio (86.37) and the ratio a reader recomputes from the two printed figures (86.05).
- "Ten of eleven comparators pass a miss-rate screen at or below 3.3%" was false under the stated method. Eight of eleven hold worst-partition bounds at or below 3.3%; the multilayer perceptron (4.27%), k-nearest neighbors (6.06%) and the shallow tree (64.51%) do not. The sentence now says **eight of eleven**, and the word "pass" is gone from the paper (item A21): the 3.3% figure is described as a threshold picked to have something to sort on, carrying no engineering authority.

### Item A5. The correlation term disagreed with the paper's own counts

Eq. 11 said ρ = *n*test/(*n*train + *n*test) = 0.202. The counts the paper reports, 4,900 and 1,264, give 0.20506. The discrepancy traced to a hard-coded pair `(1264, 4980)` in the comparison driver; 4,980 is not a number the pipeline produces under any seed.

`retrain/run_split_counts.py` now derives ρ from the partition, checks that the counts do not move across all 35 protocol seeds, and writes `v3_split_counts.json`. Every consumer reads that file; nothing carries its own copy. `retrain/run_bayes_recompute.py` recomputes the posteriors at the corrected ρ **from the stored per-run accuracies**, so no model was retrained to change an arithmetic constant, and the superseded value is recorded in the artifact as `rho_superseded`.

Seven of ten equivalence probabilities moved, all by less than 0.003. Table 8's final column, Table S4 and Table S5 are regenerated. No conclusion changes, which is the expected outcome and is worth reporting as such.

---

## Errors of fact

**Item A12.** The Conclusion said the early-train, late-test protocol cost three models "between 27 and 29 points." Table 7 gives 41.2 for the random forest, 37.9 for gradient boosting and 29.5 for the shallow tree. The sentence now names all three with the right numbers.

**Item A13.** The manuscript referred to **Fig. 18** in Experiment 6 and to **Table 15** in Data and Code Availability. The paper has fifteen figures and fourteen tables. Both are corrected, and a cross-reference sweep confirms Figs. 1 to 15 and Tables 1 to 14 are all cited and all exist.

**Item A29.** Table 5 described the bound as an "exact upper bound" while the Methods narrative, two pages earlier, explains at length why it is not exact. The table row now reads "independence-reference upper bound and disjoint-support sensitivity, Eq. 10."

**Item A46.** The Introduction cited Dehnaw et al. (2024) for "compact local language models for on-site explanation." That paper is about optimizing deep networks for edge gas detection, which is a different claim. It now supports the embedded-hardware sentence where it belongs, and Gemma Team (2025) carries the local-language-model claim.

**Found while working, not in the plan.** The supporting information still carried the old title, which included the phrase "Under Distribution Shift" that the manuscript explicitly disclaims. Synchronized.

---

## Item A6. The training target and the acceptable set were the same symbol

Eq. 2 defined *a*\*(Perfume) = {1, 2}, and Eq. 3 then wrote log *p*θ(*a*\*(*g*) | φ), taking the log-probability of a set. The code was never ambiguous: `RULE_ORACLE` maps Perfume to 1 for training, `CORRECT_ACTIONS` accepts {1, 2} for scoring. The paper conflated them.

Now there are two objects. **Eq. 2a** gives the single-valued training target *y*(*g*); **Eq. 2b** gives the acceptable set *A*(*g*). The loss uses *y*, decision accuracy uses membership in *A*, and the manuscript states outright that action 2 is never a positive training target for any class, so nothing here is trained to emit it. It is an admissible evaluation response and a rung a model can land on by accident.

---

## Claims narrowed

**Item A4 (and a correction to the previous round).** The last revision added binomial lower bounds to Table 14's episode counts, quoting 0.39 for 7/10 and 0.74 for 10/10. The plan is right that this is pseudo-replication: the ten entries are five runs times two classes over **two** physical acquisitions, and repeating a partition with a new seed does not sample a new event. The bounds are removed and a verifier gate now fails if they return.

**Item A20.** Experiment 3 is a class-exclusion, forced-routing stress test. It has no reject option, and the manuscript says so plainly.

**Items A24, A25, A26, A27.** Five-seed results are descriptive; run-level SDs are computational rather than physical replication; the correlated *t*-test is a sensitivity analysis over an assumed dependence structure; and the paired bootstrap "preserves within-run pairing and makes no parametric assumption about the paired-difference distribution" but does not remove the dependence of reusing one recording thirty times. The earlier phrase "assumes nothing about within-run correlation" was too strong and is gone.

**Item A28.** The disjoint-support counts are conservative reference counts, never an effective sample size. A gate now counts uses of that phrase and allows exactly the one that negates it.

**Item A23.** Every per-hour alarm figure comes from a 316-window clean segment, about 10.5 minutes at the 2 s cadence. The manuscript now says so where the conversion is introduced and again in Table 13's caption, and calls the hourly numbers illustrative conversions.

**Item A22.** Table 13 is retitled "alarm-grade indication density against eventized alarm rate," and Fig. 14's EEMUA annotation now reads as a contextual reference rather than a threshold, with the panel's units named on the figure.

**Items A14, A15.** "What is not settled is how any of it behaves when ranked by something other than accuracy" and "none asks whether the response would have notified anybody" were broader than the literature supports. Both are replaced with the narrower forms the plan recommends. "More than a benchmark: an evaluation framework a facility can apply" becomes a benchmarked evaluation template that might suggest how a facility could structure a review after substituting its own inputs.

**Items A17, A43, A45.** The title drops "Safety-Relevant" and "Class-Disjoint Hazards" for **"A Safety-Oriented Evaluation Protocol for Learned Gas-Monitoring Models: Leakage-Controlled Class Exclusion and Synthetic Sensor Perturbation on a Laboratory MOX Benchmark."** Keywords drop "sensor degradation," which implied physical fault testing, for "synthetic sensor perturbation" and "temporal data leakage."

**Item A40.** "Every model in Table 8 is a defensible choice" excluded nothing, including the ordinal objective the paper elsewhere calls untuned and undeployable. Now scoped to the conventional comparators.

**Items A47, A48.** Repetitive caveats and argumentative meta-commentary were cut: the runtime-boundary statement appeared three times and now appears once, the constructed-ladder caveat twice and now once beside its blockquote, and phrases like "the table earns its place" and "that cuts both ways" are gone. The Proposed Method section came down from 4,246 words to under 4,100 with nothing deleted that a referee asked for.

---

## Reproducibility

Every artifact named in this round is in `retrain/results_v2/` and listed in the provenance table (Table A-1): `v3_loco_perseed.csv` and `v3_split_counts.json` are new, `v3_partition_bounds.json`, `v3_powered_bayes.json`, `v3_rho_sensitivity.json` and `v3_rope_sensitivity.json` are regenerated. New `make` targets: `splitcounts`, `bayes`. The release audit checks both directions and reports zero problems.

**Item A51** asked for the verifier to be extended into a manuscript-wide consistency checker that would fail on the 27-29 sentence, the nonexistent Fig. 18 and Table 15, and a wrong ρ. It now does. Thirty-six checks were added this round, among them: Table 10's per-partition counts recomputed from `v3_loco_perseed.csv` and compared cell by cell against the printed column; the worst-partition bound recomputed from the largest count; the headline ratio gated against both its exact and its displayed-value form; ρ checked against the split counts and against the value stored in the Bayesian artifact; and a gate that fails if binomial inference reappears on Table 14.

---

## What was not done

**Items A7, A8, A9, A10, A32, A33, A34.** The repository work. The v13 file set is prepared and staged, but the public repository still shows an older experiment suite, and nothing here can push it. The Ultralytics version string remains the single failing check.

**Item A18, preferred path.** Rerunning Experiment 6 in raw seven-channel sensor space with physically motivated fault modes would be the stronger paper. That is a new experiment. The alternative path the plan offers was taken instead: the regimes are named as standardized-feature perturbations everywhere, including in the title and keywords, and no physical sensor-failure robustness is claimed.

**Item A19.** Isolating overlap contamination from temporal-position effects needs a control the corpus does not currently support cleanly. The claim is narrowed to a partition-protocol effect, which is what the experiment measures.

**Item A16.** Adding the closest prior art to Table 1 is the right thing to do and is only half done. Yao et al. (2024) is now cited. Four further papers the audits name, Yao et al. (2023) in *Chemometrics and Intelligent Laboratory Systems*, Du et al. (2026) in the *Journal of Chemometrics*, Wen and Khan (2026) in *Process Safety and Environmental Protection* and Parvez et al. (2025) in *Control Engineering Practice*, were located but the metadata services needed to confirm authors, volumes and pages are unreachable from this environment. Inventing those fields into a reference list is not an acceptable trade, so they are left for insertion from a machine with journal access. **Du et al. (2026) is the most worth adding**, since coverage-aware selective classification under controlled gas-sensor shift sits closest to Experiment 7.

**Item A50.** Figure and table legibility at journal page scale needs a rendered PDF read page by page, which is an author-side check.

---

## Prose

The revision was written to the style brief supplied with the plan: varied sentence length, hedged where hedging is honest, specific rather than generic, and without the defensive register that item A47 objected to. Repeated sentence openings were reduced (the commonest fell from 24 to 14 uses), and no em-dash appears in prose. Hedging was deliberately **not** applied to verified computations. Writing "the bound appears to be 16.35%" would be worse prose and worse science, since that number is recomputed from stored counts by the verification run on every execution.
