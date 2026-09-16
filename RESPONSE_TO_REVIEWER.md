# Response to the pre-submission review

**Manuscript:** ARXIS: Safety-Relevant Evaluation of Gas-Hazard Monitoring Models Under Distribution Shift and Sensor Degradation
**Recommendation received:** Major revision, invite resubmission

We are grateful for a review that read the code as well as the paper. Both P0 findings were correct, both were defects in the implementation rather than in the description of it, and finding them by reading `block_wise_holdout()` against the manuscript's own arithmetic is the kind of check we should have run on ourselves.

Reviewer comments are quoted in italics. Every number in the manuscript has been regenerated under the corrected pipeline; where a value moved, the response says so.

---

## P0 comments

### 4.1 The two-sided embargo was not implemented

> *The public block_wise_holdout() implementation appears to construct one explicit 20-window gap before the test block and then resumes the training sequence immediately after the test block.*

**Confirmed, and fixed.** The reviewer's reading of the code and of our own arithmetic was exactly right. The old implementation was

```python
train_idx.extend(block[:start])
train_idx.extend(block[start + n_test + n_gap:])   # resumes at the test block's end
test_idx.extend(block[start + n_gap:start + n_gap + n_test])
```

The trailing training segment began at `start + n_test + n_gap`, which is the index immediately after the last test window, so the first post-test training window shared up to 19 of its 20 raw readings with the last test window. The 4,980 figure in the manuscript was itself the evidence: 1,581 − 316 − 20 = 1,245 per class, a single embargo band.

The corrected function places the test block at `lo`, reserves `gap` windows on each side of it, and takes training data from `[0, lo−gap)` and `[lo+n_test+gap, n)`. Per-class training rows fall from 1,245 to 1,225 and the corpus partition is now **4,900 train / 1,264 test**. A direct check on the corrected split confirms a minimum separation of 21 window starts between any training and any test window of the same class, against the 20 needed for zero shared readings, on every seed used in the paper.

Every experiment that depends on the partition has been re-run.

### 4.2 The anomaly path used information from outside the training partition

> *The public code for the anomaly component appears to fit the autoencoder input scaler using all NoGas rows before the train/test partition is formed.*

**Confirmed, and worse than described, and fixed.** Two exposures existed, not one. `build_anomaly_model()` fitted `StandardScaler` on every NoGas row in the corpus, and the checkpoint it then loaded had itself been trained on every NoGas row. Both happened before any partition existed. The reviewer's second option, treating it as an externally pretrained frozen extractor with demonstrably disjoint training data, is not available to us, because the training data are not disjoint: they are the whole corpus.

We therefore took the first option. A new `partition_anomaly()` fits the input scaler and trains the autoencoder from scratch on the nominal windows of each training partition alone, then scores every window under that model. It is called from `prepare()`, so every driver that forms a partition gets the corrected path, and it is called explicitly in the class-disjoint drivers, where the fitting set is the three training classes. The autoencoder is memoized per distinct partition within a process.

The size of the exposure is visible in the one number that measures the anomaly component directly. The held-out anomaly ROC-AUC reported in the previous version was 0.9928 ± 0.0053. Under the corrected path it falls substantially, and the revised manuscript reports the new figure. We would rather publish the smaller number.

The downstream effects are reported throughout the revised manuscript rather than summarized here, since they touch every table.

### 4.3 The repository does not match the manuscript

> *Appendix A currently promises a reproducibility package that the public repository does not provide.*

**Accepted in full.** This was the correct diagnosis and it deserved the 4/10. The following now ship with the submission:

| Artifact | What it does |
|---|---|
| `Makefile` | one target per manuscript table and figure, plus `all`, `verify`, `manifest`, `release-check` |
| `make_manifest.py` | writes `RUN_MANIFEST.json` (commit, branch, dirty flag, host, CPU count, Python and package versions, seeds, protocol constants, SHA-256 of every driver, result file and figure) and `requirements.lock` from the live interpreter |
| `verify_v11.py` | parses the manuscript's and the supporting information's own tables and fails loudly on any value that disagrees with the stored results |
| `README.md` | rewritten; the stale reinforcement-learning terminology, the old class-disjoint numbers and the 8:1 language are gone, and the paper-versus-runtime distinction is on the first screen |
| `RELEASE_CHECKLIST.md` | the author-side sequence: freeze, tag, archive to Zenodo, reserve the DOI, cold-clone test |

`requirements.lock` is generated from the interpreter that produced the results, which removes the manuscript-versus-requirements disagreement the reviewer noticed; Appendix A now quotes it. Two items in the checklist remain genuinely author-side and are flagged there rather than claimed here: tagging and archiving the release against the real remote, and recovering the Ultralytics version used to train the thermal classifier, which is the one version the manifest cannot observe because that training happened outside this pipeline.

---

## Statistical and framing comments

### 4.4 The runtime and the evaluated object are not the same system

> *Either rename the experimental object throughout as the "ARXIS sensor-policy core" and state that the runtime adds out-of-scope guardrails, or add a final experiment evaluating the full runtime decision logic.*

**First option taken.** A new subsection, *What Is Evaluated, and What Is Not*, names the evaluated object the **ARXIS sensor-policy core** and states what the runtime adds: an anomaly-threshold override and a thermal-classifier promotion rule. It also states why the guardrails are excluded rather than merely that they are: a rule that raises actions would lift escalation adequacy and missed-hazard rate toward their best attainable values on every comparator simultaneously, which is the saturation the evaluation exists to break out of. The README repeats the boundary, and the manuscript says plainly that evaluating the full runtime against the same metric set is an experiment this paper does not contain.

### 4.5 Clopper-Pearson is not exact for overlapping windows

> *At minimum, rename the quantity an independence-reference binomial upper bound and keep the dependence caveat adjacent to every headline use.*

**Renamed, throughout.** The quantity is an **independence-reference binomial upper bound** in the Summary, the contributions, the method, the nomenclature, the abbreviations and every table caption that reports one. The method section now states why it is not exact, that the binomial model does not hold for windows overlapping by 19 of 20 raw rows, and that the quantity is reported to fix a recomputable reference point rather than as a valid confidence statement. The class-disjoint case is called out specifically: 1,581 overlapping windows from one class episode in one session are not 1,581 independent hazard opportunities, and the caption of Table 12 says so where the numbers are.

A block or event-based treatment is named in the manuscript as the single statistical improvement this design most needs. We have not added one, because doing it properly means defining the episode structure of a corpus that carries no timestamp column, and we would rather flag that than estimate it.

### 4.6 The bootstrap is robustness, not replication

> *Agreement between the correlated t-test and paired bootstrap should be described as numerical robustness to the analysis choice on this dataset, not as independent replication.*

**Accepted, and the paragraph is rewritten.** The manuscript now states that the two analyses share every observation, that 30 runs over one acquisition session are not 30 independent physical experiments, and that what the agreement establishes is only that the conclusion does not depend on the assumed ρ.

**The ROPE sweep is added.** A new driver runs both analyses at ±0.5, ±1 and ±2 accuracy points and reports, per comparison, the three posterior masses and the most probable outcome at each width, so a reader can see exactly which conclusions are threshold-dependent. The new table and its discussion are in the manuscript.

### 4.7 Engage with open-set gas recognition and alarm analytics

> *The novelty is not "unknown gases exist" and not "alarms matter."*

**Accepted, and two subsections added.** *Unknown Gases Are Not a New Problem* engages the open-set gas-recognition literature (Qu et al. 2022; Ma et al. 2024) and the open-set/drift-adaptation line, and then draws the distinction the reviewer asked for: those methods are scored on rejection accuracy and unknown-class detection, which are recognition quantities, and none of them asks what action an unfamiliar hazardous condition produces. The manuscript states explicitly that our class-disjoint experiment is deliberately *not* an open-set method, and that the question of what action a rejection should map to is left open by that literature.

*Alarm Analytics Is Also an Established Field* does the same for alarm management (Wang, J. et al. 2016; EEMUA 2024; ISA 2016), and draws the corresponding line: that work configures an alarm system that already exists, while the screen proposed here asks what to measure about a model *before* it becomes a source of alarms.

### 4.8 The action hierarchy is synthetic

> *I recommend renaming Action 4 to something like "highest-severity escalation / recommend ESD assessment".*

**Renamed to *Recommend ESD assessment*** in the action table and everywhere the label appears. Two passages now frame the ladder as a construct: one before the metric definitions, stating that no concentration, release rate, LEL fraction, toxic-exposure limit or consequence model is available in this corpus and none was used, and a new limitation, *The Action Ladder Is a Construct*, stating that every quantity built on the ladder measures agreement with a constructed target rather than consequence avoided.

The Summary sentence has been changed. It no longer says that missed hazards and false alarms "decide whether a monitoring system is safe to install"; it says they are two quantities any installation decision has to weigh.

### 4.9 Alarm-grade windows are not alarm events

> *I strongly recommend adding one eventized analysis: collapse contiguous alarm-grade windows into alarm episodes under a clearly stated persistence/debounce rule and report episodes per hour.*

**Added.** A new driver collapses contiguous alarm-grade windows on clean test windows into annunciated episodes under an explicit rule: an on-delay of 3 consecutive windows (6 s of persistence) before an episode starts, and an off-delay of 15 windows (30 s) before it ends. It reports indications and episodes side by side, per 1,000 clean windows and per hour under the same illustrative duty cycle, together with the compression factor between them, across clean and degraded conditions. The new table and its discussion are in the manuscript, and Fig. 18 is now framed as indication pressure rather than as a plant alarm-rate estimate.

**EEMUA updated.** The citation moves from the 2013 third edition to the **revised fourth edition, 2024**. One caveat we would rather state than paper over: we verified the edition and its contents structure, but not the specific numeric envelope in the fourth edition text, which is paywalled. That check is on the release checklist as an author-side item before submission.

### 4.10 Too long, and occasionally over-defensive

> *The paper would benefit from a 25-30% reduction... so many sentences are explicitly self-commenting on the writing rather than advancing the technical argument.*

**Both accepted.** A supporting-information file now carries the perception component, the explanation component, the cost-asymmetry and ordinal-objective experiment, the calibration experiment and the full comparator specifications, each reduced in the body to its result and a pointer. The self-commenting constructions the reviewer named were located and rewritten: "we would rather state", "worth saying plainly", "a reader should", "we are deliberately", "is not pedantry" and the rest are gone or replaced with direct statements.

We note one tension the reviewer's list creates, and how we resolved it: comments 4.6, 4.7 and 4.9 each require new material, so the body absorbed roughly 2,600 words of additions before any cutting began. The reduction was taken against the expanded draft, which is why the net figure is smaller than 25% while the cut itself is larger.

---

## Minor comments

| Comment | Action |
|---|---|
| "Five-quantity safety metric set" is awkward | Now "five-quantity evaluation set" |
| Conservative Q-learning needs more detail | Full specification in Supplementary S5: behavior policy (the deterministic target map), coverage, reward definition and cost structure, γ = 0.99, CQL(H) penalty at α = 1.0, one-step successor construction, and the absence of episodes and terminal states, stated as such |
| CUSUM requires explicit reset semantics | Specified, and the implementation changed. *S*ₜ now resets at every change of class block, so it is never carried across a block boundary, the embargo band, or between partitions. The previous implementation reset only per evaluation set; the reported CUSUM numbers change as a result |
| State precisely where channel zeroing is applied | A new paragraph under Experiment 6: all three perturbation families act on the **standardized** decision state, so a dropped channel sits at its training-set mean rather than at an electrical zero, which is the milder of the two channel-loss faults. Stated, with the untested case named |
| Additional sensor-fault modes | Named explicitly as untested in the limitations: stuck-at-last-value, bias, saturation, NaN, and slow accumulating drift |
| Figures have small labels at print size | Base typography raised, every annotation floored at 9 pt and scaled by 1.25, the widest figures brought from 13.4 in to 9.8 in so the print downscale stops shrinking the type, and output moved to 300 dpi. Checking the final SPE PDF at print size is on the release checklist |
| Large tables wrap model names | Abbreviated model names with a key in the caption, in the two widest tables |
| Perception and explanation get disproportionate space | Both moved to supporting information |
| Distinguish benchmark behavior from a qualification decision | Four passages rewritten; "fit for service" now explicitly names it as a facility decision that no benchmark result of this kind settles |
| Check standards citations and editions | EEMUA moved to the 2024 fourth edition; the remaining editions are on the release checklist for a final pass |

---

## What we have not done

Three things, stated plainly rather than buried.

**The thermal split is still randomized.** The images are not in the working environment, so the block-wise re-split remains outstanding. The manuscript states the asymmetry, rests no result on the perception number, and reports the figure as an optimistic upper bound.

**The full runtime is still unevaluated.** We took the reviewer's first option for 4.4 rather than the second. Evaluating the guardrails against the same metric set is roughly one driver's work and we would add it if the editor asks.

**There is no block-based uncertainty treatment.** The bounds are labeled for what they are and the improvement is named in the manuscript, but it is not implemented.
