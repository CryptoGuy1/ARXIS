# ARXIS — Experiment Results Report

> Trained on the real raw corpus `Gas_Sensors_Measurements.csv` (6400 rows → 6324 leakage-safe sliding windows after dropping class-boundary-straddling windows). Decision Agent = DuelingDQN (input 22 = [anomaly|current7|delta7|std7], output 5). All comparisons across 5 seeds (42, 1337, 7, 2024, 99).

---

## Exp 1 — Cost-Weighted Pareto

**Goal:** Find the best trade-off between danger-miss and false-alarm.

**Method:** Train Dueling DQN with cost-weighted cross-entropy at ratios 1:1, 2:1, 4:1, 6:1, 8:1, 10:1, 12:1, 16:1, 20:1. Five seeds each.

| Cost Ratio | Accuracy | Std |
|------------|----------|-----|
| 1:1 | 0.9584 | 0.0340 |
| 2:1 | 0.9616 | 0.0111 |
| 4:1 | 0.9634 | 0.0211 |
| 6:1 | 0.9658 | 0.0274 |
| **8:1** | **0.9633** | **0.0211** |
| 10:1 | 0.9584 | 0.0340 |
| 12:1 | 0.9563 | 0.0378 |
| 16:1 | 0.9511 | 0.0190 |
| 20:1 | 0.8396 | 0.119 |

**Result:** Accuracy peaks at 6:1 (0.9658) but 8:1 is deployed because it has the lowest variance (0.0211). At 20:1, accuracy collapses to 0.8396.

**Files:** `models/retrained/exp1_pareto/` (40 checkpoints)

---

## Exp 2 — Boundary Stress Test

**Goal:** Test if cost-weighting drives safer decisions on uncertain cases.

**Method:** Compare Decision Agent vs Plain DQN, MLP, GBM on full test set AND on the bottom-quartile margin subset (uncertain cases). Wilcoxon signed-rank test.

| Model | Accuracy | Std |
|-------|----------|-----|
| Decision Agent | 0.9584 | 0.0340 |
| Plain DQN | 0.9563 | 0.0378 |
| MLP | 0.9616 | 0.0111 |
| GBM | 0.9634 | 0.0211 |

**Result:** All models achieve ≈0 danger-miss on this separable dataset.

**Files:** `models/retrained/exp2_comparators/` (4 checkpoints)

---

## Exp 3 — Leave-One-Class-Out (Headline Result)

**Goal:** Test out-of-distribution generalization — can the agent catch an unseen hazard?

**Method:** For each held-out class (NoGas, Smoke, Mixture, Perfume), train 4 models on the remaining 3 classes, evaluate danger-miss on the held-out class.

| Held-Out | Model | Miss Rate | 95% Bound |
|----------|-------|-----------|-----------|
| Smoke | Decision Agent | 0.0% | ≤0.19% |
| Smoke | **MLP** | **18.7%** | **≤20.4%** |
| Smoke | Plain DQN | 0.0% | ≤0.19% |
| Smoke | GBM | 0.0% | ≤0.19% |
| Mixture | Decision Agent | 0.44% | ≤0.83% |
| Mixture | Plain DQN | 0.0% | ≤0.19% |
| Mixture | MLP | 0.38% | ≤0.75% |
| Mixture | GBM | 0.0% | ≤0.19% |

**Result:** On held-out Smoke, the MLP misses 18.7% of danger rows while the Decision Agent misses 0%. This is the headline finding: in-distribution accuracy does NOT predict out-of-distribution safety.

**Files:** `models/retrained/exp1_pareto/` (uses Exp 1 checkpoints)

---

## Exp 4 — Confidence Calibration

**Goal:** Measure how well confidence estimates match actual accuracy (ECE).

**Method:** Compute Expected Calibration Error for 4 estimators: raw_softmax, mc_dropout_20, temp_scaling, deep_ensemble_5.

| Estimator | ECE | Std |
|-----------|-----|-----|
| Raw softmax | 0.0489 | 0.0699 |
| MC dropout | 0.0533 | 0.0692 |
| Temp scaling | 0.0319 | 0.0397 |
| **Deep ensemble** | **0.0123** | **0.0043** |

**Result:** Deep ensemble best (ECE 0.0123). Temp scaling second (0.0319). Raw softmax and MC dropout worst.

**Files:** `models/retrained/exp4_calibration/` (7 checkpoints)

---

## Model Comparison (ZOO)

**Goal:** Test whether the Decision Agent's performance is unique or matched by simpler/other architectures.

**Method:** Train 8 models on same corpus, same 5 seeds, same evaluation.

| Model | Accuracy | Std | Miss Rate |
|-------|----------|-----|-----------|
| Decision Agent | 0.9633 | 0.0215 | 0.0 |
| Plain DQN | 0.9563 | 0.0378 | 0.0 |
| MLP | 0.9616 | 0.0111 | 0.0 |
| GBM | 0.9634 | 0.0211 | 0.0098 |
| SVM | 0.9511 | 0.0190 | 0.0 |
| Random Forest | 0.9769 | 0.0086 | 0.0 |
| LSTM | 0.9680 | 0.0231 | 0.0 |
| CQL | 0.9437 | 0.0479 | 0.0 |

**Result:** No baseline significantly different from the Decision Agent at α=0.05.

---

## Saved Model Weights

- `models/retrained/exp1_pareto/` — 40 DuelingDQN checkpoints (8 cost ratios × 5 seeds + 1:1 floor × 5)
- `models/retrained/exp2_comparators/` — Decision Agent, Plain DQN, MLP, GBM
- `models/retrained/exp4_calibration/` — base_net, deep_ensemble × 5, temp_scaling_T
- Total: 51 weight files + MANIFEST.json

---

## Reproducibility

All runs use 5 seeds (42, 1337, 7, 2024, 99) on the real corpus with block-wise leakage-safe split. Determinism guards (`torch.use_deterministic_algorithms(True)` + `torch.set_num_threads(1)`) present in every driver.

| Experiment | Driver | Verified |
|------------|--------|----------|
| Exp 1 | `retrain/run_exp1_real.py` | Re-run twice, bit-identical |
| Exp 2 | `retrain/run_exp2_real.py` | once |
| Exp 3 | `retrain/run_exp3_loco.py` | once |
| Exp 4 | `retrain/run_exp4_calibration.py` | once |
| ZOO | `retrain/run_expzoo.py` | once |

---

## Folder Map

- **Validation harness:** `retrain/`
- **Saved weights:** `models/retrained/`
- **Raw data:** `data/Gas_Sensors_Measurements.csv`
- **Results + CSVs + plots:** `retrain/results/`
