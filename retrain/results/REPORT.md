# ARXIS — Experiment Results

> Trained on the real raw corpus `Gas_Sensors_Measurements.csv` (6400 rows → 6324 leakage-safe sliding windows). Decision Agent = DuelingDQN (input 22, output 5). 5 seeds (42, 1337, 7, 2024, 99).

---

## Exp 1 — Cost-Weighted Pareto

Find the best trade-off between danger-miss and false-alarm.

| Cost Ratio | Accuracy | Std |
|------------|----------|-----|
| 1:1 | 0.9584 | 0.0340 |
| 6:1 | 0.9658 | 0.0274 |
| **8:1** | **0.9633** | **0.0211** |
| 20:1 | 0.8396 | 0.119 |

**Deployed:** 8:1 (lowest variance).

---

## Exp 2 — Boundary Stress Test

Compare models on uncertain cases.

| Model | Accuracy | Std |
|-------|----------|-----|
| Decision Agent | 0.9584 | 0.0340 |
| Plain DQN | 0.9563 | 0.0378 |
| MLP | 0.9616 | 0.0111 |
| GBM | 0.9634 | 0.0211 |

All models achieve ≈0 danger-miss on this separable dataset.

---

## Exp 3 — Leave-One-Class-Out (Headline Result)

Train on 3 gases, test on the 4th.

| Held-Out | Model | Miss Rate | 95% Bound |
|----------|-------|-----------|-----------|
| Smoke | Decision Agent | 0.0% | ≤0.19% |
| Smoke | **MLP** | **18.7%** | **≤20.4%** |
| Smoke | Plain DQN | 0.0% | ≤0.19% |
| Smoke | GBM | 0.0% | ≤0.19% |
| Mixture | Decision Agent | 0.44% | ≤0.83% |
| Mixture | MLP | 0.38% | ≤0.75% |

**Key finding:** In-distribution accuracy does NOT predict out-of-distribution safety.

---

## Exp 4 — Confidence Calibration

| Estimator | ECE | Std |
|-----------|-----|-----|
| Raw softmax | 0.0489 | 0.0699 |
| MC dropout | 0.0533 | 0.0692 |
| Temp scaling | 0.0319 | 0.0397 |
| **Deep ensemble** | **0.0123** | **0.0043** |

---

## Model Comparison (ZOO)

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

## Files

- `models/retrained/exp1_pareto/` — 40 checkpoints
- `models/retrained/exp2_comparators/` — 4 checkpoints
- `models/retrained/exp4_calibration/` — 7 checkpoints
- `retrain/results/*.csv` — Raw result data
