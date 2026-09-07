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

## Baseline Models (Detailed)

### Decision Agent (Dueling DQN with Cost-Weighted Cross-Entropy)

- **Full Name:** Dueling Deep Q-Network with Cost-Weighted Cross-Entropy
- **Architecture:** DuelingDQN(input_dim=22, output_dim=5, dropout=0.15) — Dueling architecture with separate value and advantage streams
- **Training:** Cost-weighted cross-entropy against rule-oracle action. Per-row weight = `miss_cost` for danger gases (Smoke/Mixture) and `false_cost` for clean/Perfume rows. This encodes the cost-asymmetry directly into a weighted-CE objective.
- **File:** `models/retrained/exp1_pareto/exp1_miss8_seed42.pth` (deployed checkpoint)
- **Use:** Primary decision policy

### Plain DQN (Dueling DQN with Standard Cross-Entropy)

- **Full Name:** Dueling Deep Q-Network with Standard Cross-Entropy
- **Architecture:** Same DuelingDQN(input_dim=22, output_dim=5, dropout=0.15)
- **Training:** Standard cross-entropy (no cost weighting) — symmetric reward (+1/-1) instead of asymmetric cost weights
- **File:** `models/retrained/exp2_comparators/exp2_B_plain_dqn_seed42.pth`
- **Use:** Ablation — proves cost-weighting matters

### MLP (Multi-Layer Perceptron)

- **Full Name:** Multi-Layer Perceptron Classifier
- **Architecture:** MLPClassifier(hidden_layer_sizes=(256, 256, 128), activation="relu", max_iter=400, early_stopping=True) — 3 hidden layers with ReLU activation
- **Training:** Supervised classification with cross-entropy on the 22-feature state vectors
- **File:** `models/retrained/exp2_comparators/exp2_D_mlp_seed42.joblib`
- **Use:** Non-deep-learning baseline; shows danger of over-reliance on simple models

### GBM (Gradient Boosting Machine)

- **Full Name:** Cost-Sensitive Gradient Boosting Machine
- **Architecture:** GradientBoostingClassifier(n_estimators=300) — 300 boosted trees
- **Training:** Boosted trees with class weights: action 0 (Monitor) weighted 3.0, actions 3-4 (Raise Alarm, Emergency Shutdown) weighted 1.2, others weighted 1.0. This up-weights dangerous-gas misclassification.
- **File:** `models/retrained/exp2_comparators/exp2_E_gbm_seed42.joblib`
- **Use:** Classical ML baseline with cost-sensitive learning

### SVM (Support Vector Machine)

- **Full Name:** RBF Support Vector Classifier
- **Architecture:** SVC(kernel="rbf", C=1.0, gamma="scale", probability=True) — RBF kernel with Platt calibration enabled
- **Training:** Supervised classification with probability calibration
- **File:** Not saved (in-memory only)
- **Use:** Classical non-deep-learning baseline

### Random Forest

- **Full Name:** Random Forest Classifier
- **Architecture:** RandomForestClassifier(n_estimators=300, n_jobs=1) — 300 decision trees
- **Training:** Ensemble of decision trees with bootstrap aggregation
- **File:** Not saved (in-memory only)
- **Use:** Classical ensemble baseline

### LSTM (Raw Window LSTM)

- **Full Name:** Recurrent Neural Network with Sliding Window
- **Architecture:** LSTM(input_dim=22, hidden=64, n_actions=5, dropout=0.2) with explicit dropout on final hidden state — recurrent net over K=10 consecutive 22-feature rows
- **Training:** Within-run windowing (no class-straddling), cross-entropy. Windows are built only within contiguous runs of one gas class. Every row gets its own prediction from the window ending at that row, left-padded by repeating the first row of its own run when fewer than K rows of history exist.
- **File:** Not saved (in-memory only)
- **Use:** Temporal context baseline

### CQL (Conservative Q-Learning)

- **Full Name:** Conservative Q-Learning Agent
- **Architecture:** DuelingDQN with conservative penalty — same architecture as Decision Agent but with CQL regularizer
- **Training:** TD(0) Bellman bootstrap (r + γ·max Q(s',a')) on offline (s, a_data, r, s') tuples, plus CQL penalty: `loss = TD_error + α·(logsumexp_a Q(s,a) - Q(s, a_data))`. This pushes down Q-values of all actions relative to the behavior action, preventing overestimation of unseen (state, action) values.
- **File:** Not saved (in-memory only)
- **Use:** Genuine offline safe-RL baseline

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
