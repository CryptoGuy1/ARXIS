# ARXIS

> ### Reproducing the SPE Journal manuscript
>
> **Reproduce the paper from the tagged release `v13-submission`, not from the tip of
> this branch.** The default branch carries development and historical material that
> does not correspond to the submitted manuscript, and its numbers may differ.
>
> ```
> git checkout v13-submission
> make check        # verify_v13.py, self_audit.py, test_safety_metrics.py,
>                   # check_figure_text.py, audit_release.py
> make all          # regenerate every reported object from the raw corpus
> ```
>
> The submitted manuscript is titled *A Safety-Oriented Benchmark for Learned
> Gas-Monitoring Models Under Class Exclusion and Sensor Degradation*. The archival
> deposit of that tag carries the DOI recorded in the paper's Data and Code
> Availability statement.

**Safety-oriented benchmarking of gas-hazard monitoring models under class exclusion and sensor degradation.**

This repository is the research artifact for the manuscript, submitted to *SPE Journal*, Data Science and Engineering Analytics. Every number in the paper is produced by the scripts here, from the raw corpus, with no manual steps.

---

## What this is, and what it is not

The contribution is an **evaluation methodology**, not a gas-detection architecture. The claim is not that the pipeline in this repository detects gas better than published alternatives. It does not: the decision accuracies reported here sit below most published figures on the same corpus, for reasons the paper's first experiment measures. The claim is that accuracy, and even missed-hazard rate on its own, can rank models in ways that reverse under class-disjoint hazards and sensor degradation, and that a five-quantity action-level evaluation set makes that visible.

Two distinctions matter for reading the code.

**The evaluated object is the sensor-policy core.** Every table and figure in the manuscript characterizes the anomaly component, the coordination component and the decision component, taking a 22-dimensional sensor state and emitting an action in {0,…,4}. The running system in `src/` is larger: it wraps the learned policy in deterministic guardrails, including an anomaly-threshold override and a promotion rule driven by the thermal classifier. **Those guardrails are not evaluated in the paper.** They would lift escalation adequacy and missed-hazard rate toward their best attainable values on every comparator at once, which is exactly the saturation the evaluation exists to break out of. Do not read the manuscript's tables as a characterization of `src/`.

**The analytes are surrogates.** The corpus contains incense smoke, alcohol-based vapor and a mixture of the two. Nothing here establishes detection performance for methane or any hydrocarbon.

---

## Corpus

[MultimodalGasData](https://data.mendeley.com/datasets/zkwgkjkjn9/2) (Narkhede et al. 2022, CC BY 4.0): 6,400 samples in four classes, a seven-channel MQ-series MOX array paired with 206×156 thermal frames, logged at 2 s intervals.

Place `Gas_Sensors_Measurements.csv` in `data/`. The thermal images are needed only for the perception component and are not used by any safety result.

---

## Protocol

Sliding windows of length 20, labeled by the final reading, with boundary-straddling windows discarded: **6,324 windows, 1,581 per class**.

Partitioning is a **block-wise holdout with a two-sided embargo**. For each class, a contiguous 20% is held out, separated from the training data by 20 excluded windows *on each side*, so that no training window shares a raw reading with any test window. The block position is drawn from the run seed, so different seeds give genuinely different partitions. This yields **4,900 training and 1,264 test windows**.

> An earlier version of `block_wise_holdout()` placed the embargo only before the test block and resumed training immediately after it, which let the first post-test training window share up to 19 raw readings with the last test window. That is fixed. Per-class training rows moved from 1,245 to 1,225, and every experiment was re-run.

The **anomaly model is refit inside each training partition**. The autoencoder's input scaler and the autoencoder itself are trained on the nominal (NoGas) windows of the training partition alone, then used to score every window.

> An earlier version fitted the scaler on all NoGas rows in the corpus and loaded a checkpoint trained on all NoGas rows, both before any partition existed. The anomaly feature therefore carried test-period exposure. That is fixed, and every experiment was re-run. The held-out anomaly ROC-AUC falls substantially as a result, which is the size of the leakage.

Uncertainty on zero counts is reported as an **independence-reference binomial upper bound**, computed by the Clopper-Pearson construction with the *partition* as the unit. It is not called an exact bound: windows overlap by 19 of 20 raw rows, so the binomial model does not hold and the bound is optimistic.

---

## Reproducing the paper

```bash
pip install -r requirements.lock
make all          # every experiment, figure, the manifest and verification
```

Individual targets map to manuscript objects. The mapping below is the summary; the
full provenance table, one row per reported object, is Appendix A of the manuscript.

| Command | Produces |
|---|---|
| `make leakage` | Table 7, Fig. 7, the partitioning-protocol experiment |
| `make final` | Table 8, Fig. 8, the 30-run comparison and the equivalence test |
| `make bayes` | recomputes the posteriors at the corrected ρ without retraining |
| `make splitcounts` | `v3_split_counts.json`, from which ρ is derived |
| `make calibration` | Table 9, Fig. 9, Tables S3, and the ROC curves |
| `make classdisjoint` | Table 10, Figs. 10 to 12 |
| `make comparison` | Table 11, Figs. 13 and 14, the ablation and the cost sweep |
| `make reject` | Table 12 and Table S9, the reject-option baseline |
| `make rawfaults` | Table 13 and Table S10, faults in raw sensor space |
| `make episodes` | Table 14, eventized alarm burden |
| `make hazardepisodes` | Table 15 |
| `make overlap` | Table S8, the overlap-versus-position control |
| `make persistence` | Table S7, the persistence-rule sweep |
| `make rho` / `make rope` | Tables S4 and S5, the two sensitivity sweeps |
| `make bounds` / `make blockboot` / `make bootstrap` | partition-level bounds; block and paired bootstrap |
| `make figures` | Figs. 1 to 15 and Figs. S1 to S5 |
| `make setupfig` | Fig. 3 alone, the acquisition schematic |
| `make verify` | checks every value in the manuscript against the result files |
| `make selfaudit` / `make test` / `make check` | the audit, the metric unit tests, and all three together |
| `make manifest` | records commit, host, package versions, file hashes |

`make verify` runs `verify_v13.py`, which parses the manuscript's own markdown tables
and fails loudly on any value that disagrees with `retrain/results_v2/`. It is the
check that keeps the paper and the code from drifting apart, and it should be green
before any release is tagged. It currently reports exactly one deliberate failure,
the Ultralytics version placeholder in Appendix A, which stays open until the version
string from the machine that trained the thermal classifier is filled in; nothing in
the repository records it and it is not going to be guessed.

`make selfaudit` runs `self_audit.py`, an independent pass that looks for superseded
numbers, stale cross-references and prose that has drifted from the tables. `make test`
runs `test_safety_metrics.py` over the metric definitions and the bound construction.

## A note on two names

Two labels in this repository read as stronger claims than the code supports, and both
are worth flagging before anyone reads them literally.

`SupervisedDQN` in `retrain/comparators.py` is **not** a reinforcement-learning agent.
It is the `DuelingDQN` network architecture trained with plain cross-entropy on actions,
with no reward, no bootstrap and no environment. The name is historical. The manuscript
calls the same object a cost-weighted cross-entropy policy, which is what it is. The one
genuinely offline-RL comparator in the study is `CQL`, and it is single-step.

Action 4 is **`Recommend ESD assessment`**, not `Emergency shutdown`. The pipeline is
advisory and actuates nothing: action 4 refers a condition for emergency-shutdown
assessment under a facility procedure. Any older figure, comment or configuration string
in this repository that says `Emergency shutdown` is stale and should be read as the
former.

Expect several hours on two cores. The autoencoder is retrained once per distinct training partition and memoized within a process.

---

## Layout

```
data/                   raw corpus (not redistributed here)
models/                 legacy autoencoder checkpoint, no longer used by experiments
retrain/
  raw_pipeline.py       windowing, partitioning, partition-local anomaly, scaling
  safety_metrics.py     the five-quantity metric set and the binomial bounds
  new_baselines.py      k-NN, shallow decision tree, CUSUM
  zoo.py, comparators.py, agent_rl.py, rewards.py, calibration.py
  run_*.py              experiment drivers, one per Makefile target
  make_figs_v2.py       the result figures
  make_fig_setup.py     Fig. 3, the acquisition schematic
  results_v2/           every CSV and JSON behind a reported number
figures_v2/             every figure in the manuscript
src/                    the running advisory system (NOT the evaluated object)
verify_v13.py           manuscript-to-results verification
self_audit.py           independent audit for superseded numbers and stale text
test_safety_metrics.py  unit tests for the metric set and the bounds
make_manifest.py        run manifest and environment lock
```

---

## Cost-ratio selection

The 8:1 loss-weight ratio that appeared in earlier versions of this repository was selected on test-partition performance, which is not a defensible selection procedure. The manuscript now reports the ratio chosen on a **blocked validation band carved out of the training partition**, and reports 8:1 only as a reference point alongside it. Text or configuration in this repository that still presents 8:1 as an optimized deployment setting is stale; the manuscript's Experiment 4 supersedes it.

---

## Citing

Nweke, B. C., Ramezan, G., and Saraji, S. ARXIS: Safety-Oriented Evaluation of Learned Gas-Monitoring Models Under Class Exclusion and Sensor Degradation. Manuscript submitted to *SPE Journal*, Data Science and Engineering Analytics.

**Archived release.** The version of this repository that produced the submitted
numbers is tagged `v13-submission`. A Zenodo deposit of that tag carries the DOI; the
DOI is minted at deposit time and is recorded here and in the manuscript's Data and
Code Availability statement once it exists. Until then the tag is the citable object,
and a reviewer reproducing the paper should check out the tag rather than the tip of
the default branch.

Corpus: Narkhede, P., Walambe, R., Chandel, P., et al. 2022. MultimodalGasData: Multimodal Dataset for Gas Detection and Classification. *Data* 7 (8): 112.

## License

Code under the repository license. The corpus is CC BY 4.0 and is redistributed under its own terms.
