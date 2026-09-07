# ARXIS — Industrial Gas Safety Intelligence

> **Multimodal agentic safety system for gas monitoring and action selection.**
> Dueling Deep Q-Network · LSTM Autoencoder · YOLOv8 · Safety Guardrails · Explainability Layer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.62+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [The Problem](#the-problem)
- [System Architecture](#system-architecture)
- [Experiments](#experiments)
- [Baseline Models](#baseline-models)
- [Setup \& Installation](#setup--installation)
- [Running the System](#running-the-system)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Open Items](#open-items)
- [License](#license)

---

## Overview

ARXIS detects four gas conditions from MQ sensor arrays and selects one of five safety actions using a combination of time-series analysis, anomaly detection, reinforcement learning, thermal-image verification, and operator-facing explanations.

### Action Space

| Action ID | Action | Description |
|-----------|--------|-------------|
| 0 | Monitor | Continue normal observation |
| 1 | Increase Sampling | Collect more sensor data |
| 2 | Request Verification | Ask for human confirmation |
| 3 | Raise Alarm | Alert operators |
| 4 | Emergency Shutdown | Immediate shutdown |

### Canonical Gas Mapping

| Gas Class | gas_id | Hazard Level | Correct Action |
|-----------|--------|--------------|----------------|
| NoGas | 0 | Safe | Monitor (0) |
| Smoke | 1 | **Danger** | Raise Alarm (3) |
| Mixture | 2 | **Danger** | Emergency Shutdown (4) |
| Perfume | 3 | Low-risk VOC | Increase Sampling (1) or Request Verification (2) |

> **Note:** YOLO's internal class index order differs from this semantic mapping. The Vision Agent converts raw YOLO class index → text label → canonical gas_id through `GAS_MAP`.

---

## The Problem

Current gas monitoring uses fixed thresholds. This fails in three ways:

1. **No temporal reasoning** — a spike is not a trend. Thresholds false-alarm on harmless spikes and miss slow leaks.
2. **No generalization** — a model trained on known gases may miss 18.7% of danger cases on an unfamiliar gas, even at 96% in-distribution accuracy.
3. **No calibrated uncertainty** — saying "danger" without confidence prevents operators from triaging.

ARXIS addresses all three.

---

## System Architecture

```
MultimodalAgent (orchestrator)
├── AnomalyTool      — LSTM autoencoder reconstruction error
├── DecisionTool     — Dueling Deep Q-Network (22-dim state → 5 actions)
├── VisionTool       — YOLOv8 thermal-image classification
├── ExplanationTool  — Ollama Gemma 3 1B natural-language explanation
└── ShortTermMemory  — Output history
```

### State Vector (22 features)

```python
[anomaly, MQ2...MQ135, dMQ2...dMQ135, sMQ2...sMQ135]
# 1 + 7 current + 7 delta + 7 std
```

---

## Experiments

### Exp 1 — Cost-Weighted Pareto

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

### Exp 2 — Boundary Stress Test

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

### Exp 3 — Leave-One-Class-Out (Headline Result)

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

### Exp 4 — Confidence Calibration

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

### Model Comparison (ZOO)

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

**Files:** No saved weights (SVM, RF, LSTM, CQL trained in-memory only).

---

## Baseline Models

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

## Setup & Installation

### Requirements

- Python 3.10+ or 3.11+
- ~2 GB disk space
- Ollama (optional, for explanation features)

### Step 1 — Clone

```bash
git clone https://github.com/CryptoGuy1/ARXIS.git
cd ARXIS
```

### Step 2 — Create Virtual Environment

```bash
python -m venv .venv-retrain
source .venv-retrain/bin/activate  # Linux/macOS
# OR
.venv-retrain\Scripts\activate  # Windows
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download Dataset

Place `Gas_Sensors_Measurements.csv` in `data/`:
```
data/Gas_Sensors_Measurements.csv
```

### Step 5 — Install Ollama (Optional)

For explanation/critique features:
```bash
ollama pull gemma3:1b
```

---

## Running the System

### Dashboard (Recommended)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Five tabs:
- **Overview** — action distribution, confidence histogram, anomaly scatter, per-gas comparison
- **Case Details** — pipeline chart, Q-values, structured output, vision report, explanation, critique
- **Sensor Feed** — 7-channel sensor traces, anomaly evolution, per-sensor statistics
- **State Inspector** — anomaly gauge, 22-dim state vector, decision chain
- **Structured Table** — export-ready results

### CLI Runner

```bash
python main.py
```

### Run Experiments

```bash
python retrain/run_exp1_real.py        # Cost-Weighted Pareto
python retrain/run_exp2_real.py        # Boundary Stress Test
python retrain/run_exp3_loco.py        # LOCO
python retrain/run_exp4_calibration.py # Calibration
python retrain/run_expzoo.py           # All Baselines
```

---

## Results Summary

| Experiment | Key Finding |
|------------|-------------|
| **Cost-Weighted Pareto** | 8:1 ratio optimal (0.9633 accuracy, lowest variance) |
| **Boundary Stress** | All models ≈0 danger-miss on separable data |
| **LOCO** | MLP misses 18.7% of unseen Smoke; Decision Agent misses 0% |
| **Calibration** | Deep ensemble ECE = 0.0123 |
| **ZOO** | No baseline significantly better than Decision Agent |

---

## Project Structure

```
ARXIS/
├── app.py                  # Streamlit dashboard
├── main.py                 # CLI runner
├── requirements.txt        # Dependencies
├── README.md               # This file
├── .gitignore              # Security + exclusions
├── data/
│   └── Gas_Sensors_Measurements.csv
├── models/
│   ├── DeepQnet.pth
│   ├── lstm_autoencoder_weights.pth
│   ├── yolov8_gas_classifier.pt
│   ├── model_config.json
│   └── retrained/
│       ├── MANIFEST.json
│       ├── exp1_pareto/        # 40 checkpoints
│       ├── exp2_comparators/   # 4 checkpoints
│       └── exp4_calibration/   # 7 checkpoints
├── retrain/
│   ├── raw_pipeline.py         # Data pipeline
│   ├── run_exp1_real.py        # Exp 1 driver
│   ├── run_exp2_real.py        # Exp 2 driver
│   ├── run_exp3_loco.py        # Exp 3 driver
│   ├── run_exp4_calibration.py # Exp 4 driver
│   ├── run_expzoo.py           # ZOO driver
│   ├── zoo.py                  # ZOO model definitions
│   ├── comparators.py          # Comparator models
│   ├── metrics.py              # Evaluation metrics
│   ├── calibration.py          # Calibration methods
│   ├── rewards.py              # Reward functions
│   ├── agent_rl.py             # RL agent
│   ├── fetch_from_drive.py     # Google Drive dataset fetcher
│   ├── drive_access.py         # Google Drive direct access
│   └── results/
│       ├── REPORT.md           # Full results report
│       ├── *.csv               # Raw result data
│       └── *.png               # Result plots
└── src/
    ├── agent/
    │   ├── agent_core.py       # Main orchestrator
    │   ├── safety.py           # Safety override layer
    │   ├── critic.py           # CriticAgent
    │   ├── goal_manager.py     # GoalManager
    │   ├── memory.py           # ShortTermMemory
    │   ├── supervisor.py       # SupervisorAgent
    │   ├── trainer.py          # AgentTrainer
    │   ├── reward_system.py    # Reward functions
    │   └── metrics_logger.py   # Logging
    └── tools/
        ├── anomaly_tool.py     # LSTM-AE anomaly detection
        ├── decision_tool.py    # Dueling DQN decision
        ├── vision_tool.py      # YOLOv8 vision verification
        └── explanation_tool.py # Ollama explanation generation
```

---

## Open Items

1. **Exp 3** was run per-class (not single uninterrupted run) due to Cygwin limits. Run on Linux for single-run reproducibility.
2. **n=5 seeds** limits statistical power. Danger-miss reported as exact Clopper–Pearson bounds.
3. **Thermal images** not included — system runs sensor-only without them.

---

## License

MIT

---

**ARXIS** — Intelligence That Delivers.
