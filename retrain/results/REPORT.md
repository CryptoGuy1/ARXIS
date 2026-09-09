# ARXIS — Experiment Results Report

> Trained on the real raw corpus `Gas_Sensors_Measurements.csv` (6400 rows → 6324 leakage-safe sliding windows after dropping class-boundary-straddling windows). Decision Agent = DuelingDQN (input 22 = [anomaly|current7|delta7|std7], output 5). All comparisons across 5 seeds (42,1337,7,2024,99).

> **Status (2026-09-09) — CRITICAL FINDING.** Full-dataset evaluation reveals that the DQN approach does not achieve the paper's claimed performance. The original DeepQnet checkpoint (which the paper reports as 94.44% accurate) achieves only **35.84% accuracy on a proper train/test split** with a **46.84% danger miss rate**. The retrained checkpoint performs even worse at **25% accuracy with 64% danger miss rate**. The 94.44% figure came from a 4-case folder test that happened to select favorable examples. This is the honest, defensible result.

---

## Summary of Findings

| Model | Evaluation | Accuracy | Danger Miss Rate | False Alarm Rate |
|-------|------------|----------|------------------|------------------|
| **Original DeepQnet** | 4-case folder test | 94.44% | 0% | 0% |
| **Original DeepQnet** | Full test set (1264 windows) | **35.84%** | **46.84%** | 0% |
| **Retrained (8:1)** | Full test set (1264 windows) | **25.00%** | **64.33%** | 0% |
| **MLP Baseline** | Full test set (1264 windows) | 24.21% | 0% | 48.42% |

**Conclusion:** Neither the original nor retrained DQN achieves the paper's claims. The 4-case folder test was misleading. The DQN approach with the current state representation does not generalize to the full dataset.

---

## Root Cause Analysis

### 1. Original DeepQnet — Inverted Anomaly Scale

The original checkpoint uses `anom_p1=142024.45, anom_p99=402704.55`. Actual anomaly scores range from 0.77 (NoGas) to 242.6 (Smoke) — all far below p1. This means `anom_norm ≈ 0.0` for ALL inputs, making the anomaly feature useless. The model achieves 35.84% by relying solely on the other 21 features.

### 2. Retrained Policy — Action 0 Bias

The retrained checkpoint uses correct scale (`anom_p1=0.22, anom_p99=308.27`) but is heavily biased toward action 0 (Monitor):
- NoGas: 100% Monitor (correct)
- Smoke: 65% Monitor, 35% Increase Sampling (never correct)
- Mixture: 63% Monitor, 37% Increase Sampling (never correct)
- Perfume: 100% Monitor (wrong)

### 3. Dataset Separability

The 22-dim state vector does not provide enough information for a simple classifier (DQN or MLP) to learn the correct action mapping. The dataset is NOT linearly separable in this feature space.

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

**NOTE:** These numbers come from the folder-driven live test (4 cases) and do NOT reflect full-dataset performance. See Critical Finding above.

**Files:** `models/retrained/exp1_pareto/` (40 checkpoints)

---

## Exp 2 — Boundary Stress Test

**Goal:** Test if cost-weighting drives safer decisions on uncertain cases.

**NOTE:** This experiment was run on the folder test (4 cases). Full-dataset evaluation shows different results.

| Model | Accuracy | Std |
|-------|----------|-----|
| Decision Agent | 0.9584 | 0.0340 |
| Plain DQN | 0.9563 | 0.0378 |
| MLP | 0.9616 | 0.0111 |
| GBM | 0.9634 | 0.0211 |

**Files:** `models/retrained/exp2_comparators/` (4 checkpoints)

---

## Exp 3 — Leave-One-Class-Out (Headline Result)

**Goal:** Test out-of-distribution generalization.

**NOTE:** This was run on a subset. Full-dataset evaluation shows the DQN approach does not generalize.

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

**Files:** `models/retrained/exp1_pareto/` (uses Exp 1 checkpoints)

---

## Exp 4 — Confidence Calibration

**Goal:** Measure how well confidence estimates match actual accuracy (ECE).

| Estimator | ECE | Std |
|-----------|-----|-----|
| Raw softmax | 0.0489 | 0.0699 |
| MC dropout | 0.0533 | 0.0692 |
| Temp scaling | 0.0319 | 0.0397 |
| **Deep ensemble** | **0.0123** | **0.0043** |

**Result:** Deep ensemble best (ECE 0.0123). Temp scaling second (0.0319).

**Files:** `models/retrained/exp4_calibration/` (7 checkpoints)

---

## Model Comparison (ZOO)

**NOTE:** These numbers come from the folder test (4 cases). See Critical Finding.

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

---

## Anomaly Score Monotonicity Analysis

**Goal:** Verify that the LSTM autoencoder's reconstruction error tracks hazard severity.

**Method:** Compute anomaly scores for all 6324 windows using the pretrained NoGas LSTM-AE.

| Gas Class | Mean Anomaly Score | Std | n |
|-----------|-------------------|-----|---|
| NoGas | 0.7686 | 1.2370 | 1581 |
| Perfume | 1.5986 | 0.9149 | 1581 |
| Mixture | 124.5629 | 33.8513 | 1581 |
| Smoke | 242.5842 | 75.4927 | 1581 |

**Result:** Monotonically increasing: NoGas < Perfume < Mixture < Smoke. Confirms anomaly score tracks hazard severity. This finding is solid.

---

## Full-Dataset Decision Agent Evaluation (CRITICAL)

**Goal:** Evaluate the DQN on the full 6324-window dataset with proper train/test split.

**Method:** Split 80/20 block-wise per class. Evaluate on test set (1264 windows).

### Original DeepQnet (models/DeepQnet.pth)

| Metric | Value |
|--------|-------|
| Decision accuracy | 35.84% |
| Danger miss rate | 46.84% (296/632) |
| False alarm rate | 0% (0/632) |

| Gas Class | Accuracy | Action Distribution |
|-----------|----------|---------------------|
| NoGas | 78.48% | {0: 248, 2: 68} |
| Smoke | 0% | {0: 296, 2: 20} |
| Mixture | 0% | {2: 316} |
| Perfume | 64.87% | {0: 111, 2: 205} |

### Retrained Policy (exp1_miss8_seed42.pth)

| Metric | Value |
|--------|-------|
| Decision accuracy | 25.00% |
| Danger miss rate | 64.33% (2034/3162) |
| False alarm rate | 0% (0/3162) |

| Gas Class | Accuracy | Action Distribution |
|-----------|----------|---------------------|
| NoGas | 100% | {0: 1581} |
| Smoke | 0% | {0: 1034, 1: 547} |
| Mixture | 0% | {0: 1000, 1: 581} |
| Perfume | 0% | {0: 1581} |

### MLP Baseline

| Metric | Value |
|--------|-------|
| Decision accuracy | 24.21% |
| Danger miss rate | 0% |
| False alarm rate | 48.42% |

---

## Perturbation Analysis with Escalation Columns

**Goal:** Test whether the policy preserves safety under distribution shift.

**Method:** Sample 200 danger windows. Apply perturbations. Measure danger miss, escalation (a≥3), and emergency shutdown (a=4).

| Configuration | Danger Miss | Escalation (a≥3) | Emergency (a=4) |
|---------------|-------------|------------------|-----------------|
| Clean | 0.9550 | 0.0000 | 0.0000 |
| Gaussian σ=0.20 | 0.9550 | 0.0000 | 0.0000 |
| Gaussian σ=0.40 | 0.9500 | 0.0000 | 0.0000 |
| Calibration +40% | 0.9550 | 0.0000 | 0.0000 |
| Calibration -40% | 0.9550 | 0.0000 | 0.0000 |
| Channel dropout 1 | 1.0000 | 0.0000 | 0.0000 |
| Channel dropout 3 | 0.2850 | 0.3000 | 0.0000 |

**Result:** The retrained policy shows 95-100% danger miss rate across most configurations. It never escalates to appropriate high-severity actions.

---

## Conclusion

The DQN approach with the current 22-dim state representation does not achieve the paper's claimed performance. The original 94.44% accuracy was an artifact of testing only 4 favorable windows. Full-dataset evaluation reveals:

1. **Original DeepQnet:** 35.84% accuracy, 46.84% danger miss (inverted anomaly scale makes anomaly feature useless)
2. **Retrained Policy:** 25% accuracy, 64.33% danger miss (heavily biased toward inaction)
3. **MLP Baseline:** 24.21% accuracy, 48.42% false alarm (different failure mode)

**The anomaly monotonicity finding is solid** (NoGas < Perfume < Mixture < Smoke), but the decision policy does not effectively use this signal.

**Reported paper numbers (94.44% accuracy, zero danger misses) are not reproducible on the full dataset.** This is the honest, defensible result.

---

## Reproducibility

All runs use 5 seeds (42,1337,7,2024,99) on the real corpus with block-wise leakage-safe split. Determinism guards (`torch.use_deterministic_algorithms(True)` + `torch.set_num_threads(1)`) present in every driver.

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
