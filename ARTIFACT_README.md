# ARXIS

**Safety-relevant evaluation of gas-hazard monitoring models under distribution shift and sensor degradation.**

This repository is the research artifact for the manuscript of the same name, submitted to *SPE Journal*, Data Science and Engineering Analytics. Every number in the paper is produced by the scripts here, from the raw corpus, with no manual steps.

---

## What this is, and what it is not

The contribution is an **evaluation methodology**, not a gas-detection architecture. The claim is not that the pipeline in this repository detects gas better than published alternatives. It does not: the decision accuracies reported here sit below most published figures on the same corpus, for reasons the paper's first experiment measures. The claim is that accuracy, and even missed-hazard rate on its own, can rank models in ways that reverse under class-disjoint hazards and sensor degradation, and that a five-quantity action-level screen makes that visible.

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

Individual targets map to manuscript objects:

| Command | Produces |
|---|---|
| `make comparison` | Tables 9, 13, 14, 15; Figs. 9, 10, 14–18 |
| `make classdisjoint` | Table 12; Figs. 11–13 |
| `make leakage` | Table 8; Fig. 8 |
| `make calibration` | Table 16; Figs. 6, 7, 19 |
| `make episodes` | Table 18, eventized alarm burden |
| `make final` | the 30-run protocol and the equivalence test |
| `make rho` / `make rope` | Tables 10 and 11, the two sensitivity sweeps |
| `make bounds` / `make bootstrap` | partition-level bounds; paired bootstrap |
| `make figures` | Figs. 1–20 |
| `make verify` | checks every value in the manuscript against the result files |
| `make manifest` | records commit, host, package versions, file hashes |

`make verify` runs `verify_v11.py`, which parses the manuscript's own markdown tables and fails loudly on any value that disagrees with `retrain/results_v2/`. It is the check that keeps the paper and the code from drifting apart, and it should be green before any release is tagged.

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
  make_figs_v2.py       all 20 figures
  results_v2/           every CSV and JSON behind a reported number
figures_v2/             every figure in the manuscript
src/                    the running advisory system (NOT the evaluated object)
verify_v11.py           manuscript-to-results verification
make_manifest.py        run manifest and environment lock
```

---

## Cost-ratio selection

The 8:1 loss-weight ratio that appeared in earlier versions of this repository was selected on test-partition performance, which is not a defensible selection procedure. The manuscript now reports the ratio chosen on a **blocked validation band carved out of the training partition**, and reports 8:1 only as a reference point alongside it. Text or configuration in this repository that still presents 8:1 as an optimized deployment setting is stale; the manuscript's Experiment 4 supersedes it.

---

## Citing

Nweke, B. C., Ramezan, G., and Saraji, S. ARXIS: Safety-Relevant Evaluation of Gas-Hazard Monitoring Models Under Distribution Shift and Sensor Degradation. Manuscript submitted to *SPE Journal*.

Corpus: Narkhede, P., Walambe, R., Chandel, P., et al. 2022. MultimodalGasData: Multimodal Dataset for Gas Detection and Classification. *Data* 7 (8): 112.

## License

Code under the repository license. The corpus is CC BY 4.0 and is redistributed under its own terms.
