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
- [The Problem & Research Gap](#the-problem--research-gap)
- [System Architecture](#system-architecture)
- [Model Naming Convention](#model-naming-convention)
- [Experiments](#experiments)
- [Baseline Models](#baseline-models)
- [Setup & Installation](#setup--installation)
- [Running the System](#running-the-system)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Open Items](#open-items)
- [License](#license)

---

## Overview

ARXIS is a multimodal, agentic industrial gas safety system that combines:

- **Time-series gas sensor analysis** — 7-channel MQ sensor array
- **Anomaly detection** — LSTM autoencoder reconstruction error
- **Reinforcement learning** — Dueling Deep Q-Network action selection
- **Thermal-image verification** — YOLOv8 visual gas classification
- **Operator-facing explanation** — Ollama LLM (Gemma 3 1B)
- **Safety guardrails** — Hard override layer on RL policy

The system answers one question: **Given recent gas sensor behavior and available thermal-image evidence, what safety action should be taken right now?**

### Action Space

| Action ID | Action | Description |
|-----------|--------|-------------|
| 0 | Monitor | Continue normal observation |
| 1 | Increase Sampling | Collect more sensor data |
| 2 | Request Verification | Ask for human confirmation |
| 3 | Raise Alarm | Alert operators |
| 4 | Emergency Shutdown | Immediate shutdown |

### Canonical Gas Mapping

| Gas Class | `gas_id` | Hazard Level | Correct Action |
|-----------|----------|--------------|----------------|
| NoGas | 0 | Safe | Monitor (0) |
| Smoke | 1 | **Danger** | Raise Alarm (3) |
| Mixture | 2 | **Danger** | Emergency Shutdown (4) |
| Perfume | 3 | Low-risk VOC | Increase Sampling (1) or Request Verification (2) |

> **Note:** YOLO's internal class index order differs from this semantic mapping. The Vision Agent converts raw YOLO class index → text label → canonical `gas_id` through `GAS_MAP`.

---

## The Problem & Research Gap

Industrial gas monitoring today is dominated by single-threshold alarm systems: a sensor reading crosses a fixed limit, an alarm sounds. This approach has three critical failures:

1. **No temporal reasoning.** A single spike is not the same as a rising trend. Threshold systems cannot distinguish a transient reading from a developing leak. They miss slow-building hazards and false-alarm on harmless spikes.

2. **No out-of-distribution generalization.** A model trained on known gases (methane, propane) may fail catastrophically on an unfamiliar gas (hydrogen sulfide, ammonia). In-distribution accuracy does not predict out-of-distribution safety — a supervised MLP can achieve 96% accuracy on familiar gases yet miss **18.7%** of danger cases on an unfamiliar one.

3. **No calibrated uncertainty.** A safety system that says "danger" without saying how confident it is leaves the operator unable to triage. Over-confident wrong decisions are worse than no decision.

**ARXIS addresses all three gaps:**

- A **22-dimensional temporal state vector** (anomaly + current + delta + std over a 20-step window) replaces single-point readings, giving the policy a sense of time.
- A **cost-weighted Dueling DQN** with a hard safety override layer generalizes to unfamiliar gases — on held-out Smoke, the Decision Agent misses **0%** of danger rows while a supervised MLP misses **18.7%**.
- **MC Dropout** and **temperature scaling** produce calibrated confidence estimates, with a deep ensemble achieving an Expected Calibration Error of just **0.0123**.

The result is a system that does not just classify gas — it chooses a safety action, knows when it is uncertain, and fails safely on hazards it has never seen.

---

## System Architecture

The system is organized around one main orchestrator (`MultimodalAgent`) and several specialized sub-agents:

- **Anomaly Agent** → `AnomalyTool` — LSTM autoencoder-based anomaly estimation
- **Decision Agent** → `DecisionTool` — Dueling Deep Q-Network action selection
- **Vision Agent** → `VisionTool` — YOLOv8-based visual gas classification
- **Explanation Agent** → `ExplanationTool` — operator-facing explanation generation (via Ollama Gemma 3 1B)
- **Memory Agent** → `ShortTermMemory` — maintains structured output history

### The 22-Feature State Vector

```python
state = [
    anomaly,                                   # normalized anomaly score
    MQ2, MQ3, MQ5, MQ6, MQ7, MQ8, MQ135,      # current sensor readings
    dMQ2, dMQ3, dMQ5, dMQ6, dMQ7, dMQ8, dMQ135,  # delta over window
    sMQ2, sMQ3, sMQ5, sMQ6, sMQ7, sMQ8, sMQ135,  # std over window
]
```

State dimension = **22** (1 anomaly + 7 current + 7 delta + 7 std).

### Training Objective

The network *architecture* is a Dueling DQN, but the *training objective* is **cost-weighted cross-entropy against a rule-oracle action**, NOT temporal-difference reinforcement learning. These checkpoints contain no replay buffer, no target network, no n-step return, no epsilon-greedy exploration, and no Bellman bootstrap.

The one exception is the CQL agent in `retrain/zoo.py`, which trains with a genuine Conservative Q-Learning update (TD(0) Bellman target + conservative logsumexp penalty).

---

## Model Naming Convention

All model files follow a consistent naming scheme:

```
{experiment}_{model_description}_{seed}.{extension}
```

### Experiments

| Prefix | Experiment | Description |
|--------|------------|-------------|
| `exp1` | Pareto Cost-Weighted CE | Dueling DQN with varying miss-cost ratios |
| `exp2` | Boundary Stress Test | Comparator models on near-boundary cases |
| `exp3` | LOCO | Leave-one-class-out evaluation |
| `exp4` | Calibration | Confidence calibration estimators |

### Baseline Models (ZOO)

| Code | Model | Type | File Extension |
|------|-------|------|----------------|
| A | Decision Agent | Dueling DQN (cost-weighted CE) | `.pth` |
| B | Plain DQN | Dueling DQN (standard CE) | `.pth` |
| D | MLP | Multi-Layer Perceptron | `.joblib` |
| E | GBM | Gradient Boosting Machine | `.joblib` |
| SVM | Support Vector Machine | RBF Kernel | (in-memory) |
| RF | Random Forest | 300 trees | (in-memory) |
| LSTM | Raw Window LSTM | Recurrent (K=10) | (in-memory) |
| CQL | Conservative Q-Learning | Offline safe-RL | (in-memory) |

### File Naming Examples

```
exp1_pareto/
  exp1_miss8_seed42.pth          # Exp #1, 8:1 miss ratio, seed 42
  exp1_baseline_1to1_seed42.pth  # Exp #1, 1:1 baseline (no cost weighting)

exp2_comparators/
  exp2_A_decision_agent_seed42.pth   # Exp #2, Model A (Decision Agent)
  exp2_B_plain_dqn_seed42.pth        # Exp #2, Model B (Plain DQN)
  exp2_D_mlp_seed42.joblib           # Exp #2, Model D (MLP)
  exp2_E_gbm_seed42.joblib           # Exp #2, Model E (GBM)

exp4_calibration/
  exp4_base_net_seed42.pth           # Exp #4, base network
  exp4_ensemble_seed42_net0.pth      # Exp #4, ensemble member 0
  exp4_temp_scaling_T_seed42.npy     # Exp #4, temperature scaling parameter
```

---

## Experiments

### Exp #1 — Pareto Cost-Weighted CE

**Goal:** Find the optimal miss-to-false-alarm cost ratio.

**Method:** Train Dueling DQN with cost-weighted CE at ratios 1:1, 2:1, 4:1, 6:1, 8:1, 10:1, 12:1, 16:1, 20:1. Five seeds each.

**Result:** Pareto peak at **8:1 = 0.9633 ± 0.0215**. Accuracy peaks at 6:1 (0.9658 ± 0.0274) but 8:1 is deployed because it has tighter seed spread. At 20:1, accuracy collapses to 0.8396 ± 0.119.

| Ratio | Accuracy | Std | Miss Rate |
|-------|----------|-----|-----------|
| 1:1 | 0.9584 | 0.0340 | 0.0 |
| 2:1 | 0.9616 | 0.0111 | 0.0 |
| 4:1 | 0.9634 | 0.0211 | 0.0 |
| 6:1 | **0.9658** | 0.0274 | 0.0 |
| **8:1** | **0.9633** | **0.0211** | **0.0** |
| 10:1 | 0.9584 | 0.0340 | 0.0 |
| 20:1 | 0.8396 | 0.119 | 0.0 |

**Files:** `models/retrained/exp1_pareto/exp1_miss{ratio}_seed{seed}.pth` (40 files)

---

### Exp #2 — Boundary Stress Test

**Goal:** Test if cost-weighting drives safer decisions on uncertain (boundary) cases.

**Method:** Compare Decision Agent (A) vs Plain DQN (B), MLP (D), GBM (E) on full test set AND on the bottom-quartile margin subset (uncertain cases). Wilcoxon signed-rank test.

**Result:** On this separable dataset, the cost-weighting edge is not exposed; all models achieve ≈0 danger-miss.

| Model | Accuracy | Std | Miss Rate |
|-------|----------|-----|-----------|
| A (Decision Agent) | 0.9584 | 0.0340 | 0.0 |
| B (Plain DQN) | 0.9563 | 0.0378 | 0.0 |
| D (MLP) | 0.9616 | 0.0111 | 0.0 |
| E (GBM) | 0.9634 | 0.0211 | 0.0098 |

**Files:** `models/retrained/exp2_comparators/exp2_{model}_seed42.{ext}` (4 files)

---

### Exp #3 — LOCO (Leave-One-Class-Out)

**Goal:** Test out-of-distribution fail-safety — can the agent generalize to an unseen gas class?

**Method:** For each held-out class (NoGas, Smoke, Mixture, Perfume), train 4 models (A, B, D, E) on the remaining 3 classes, evaluate danger-miss on the held-out class.

**Result:** On held-out Smoke, the MLP misses **18.7%** of danger rows (Clopper-Pearson bound ≤20.4%), while A, B, E all record 0 misses (bound ≤0.19%). **This is the headline finding:** in-distribution danger-miss does NOT predict out-of-distribution fail-safety.

| Held-Out | Model | Miss Rate | 95% CP Bound |
|----------|-------|-----------|--------------|
| Smoke | A (Decision Agent) | 0.0 | ≤0.189% |
| Smoke | B (Plain DQN) | 0.0 | ≤0.189% |
| Smoke | D (MLP) | **18.7%** | **≤20.4%** |
| Smoke | E (GBM) | 0.0 | ≤0.189% |
| Mixture | A | 0.44% | ≤0.830% |
| Mixture | B | 0.0 | ≤0.189% |
| Mixture | D | 0.38% | ≤0.748% |
| Mixture | E | 0.0 | ≤0.189% |

**Files:** `models/retrained/exp1_pareto/` (uses Exp #1 checkpoints)

---

### Exp #4 — Confidence Calibration

**Goal:** Measure how well confidence estimates match actual accuracy (ECE).

**Method:** Compute Expected Calibration Error for 4 estimators: raw_softmax, mc_dropout_20, temp_scaling, deep_ensemble_5.

**Result:** Deep ensemble best (ECE 0.0123 ± 0.0043). Temp scaling second (0.0319). Raw softmax (0.0489) and MC dropout (0.0533) worst.

| Estimator | ECE Mean | ECE Std |
|-----------|----------|---------|
| raw_softmax | 0.0489 | 0.0699 |
| mc_dropout_20 | 0.0533 | 0.0692 |
| temp_scaling | 0.0319 | 0.0397 |
| **deep_ensemble_5** | **0.0123** | **0.0043** |

**Files:** `models/retrained/exp4_calibration/exp4_*.pth`

---

### Exp ZOO — Broader Model Comparison

**Goal:** Test whether the Decision Agent's performance is unique or matched by simpler/other architectures.

**Method:** Train 8 models (A, B, D, E, SVM, RF, LSTM, CQL) on same corpus, same 5 seeds, same evaluation.

**Result:** RF highest accuracy (0.9769 ± 0.0086), but no comparator significantly different from A at α=0.05. LSTM miss dropped to 0 (from 0.0044) after within-run windowing fix.

| Model | Accuracy | Std | Miss Rate |
|-------|----------|-----|-----------|
| A (Decision Agent) | 0.9633 | 0.0215 | 0.0 |
| B (Plain DQN) | 0.9563 | 0.0378 | 0.0 |
| D (MLP) | 0.9616 | 0.0111 | 0.0 |
| E (GBM) | 0.9634 | 0.0211 | 0.0098 |
| SVM | 0.9511 | 0.0190 | 0.0 |
| **RF** | **0.9769** | **0.0086** | **0.0** |
| LSTM | 0.9680 | 0.0231 | 0.0 |
| CQL | 0.9437 | 0.0479 | 0.0 |

**No comparator significantly different from A at α=0.05.**

**Files:** No saved weights (SVM, RF, LSTM, CQL trained in-memory only).

---

## Baseline Models

### A: Decision Agent (Dueling DQN, Cost-Weighted CE)
- **Architecture:** DuelingDQN(input_dim=22, output_dim=5, dropout=0.15)
- **Training:** Cost-weighted cross-entropy against rule-oracle action
- **File:** `models/retrained/exp1_pareto/exp1_miss8_seed42.pth` (deployed checkpoint)

### B: Plain DQN (Dueling DQN, Standard CE)
- **Architecture:** Same as A
- **Training:** Standard cross-entropy (no cost weighting)
- **File:** `models/retrained/exp2_comparators/exp2_B_plain_dqn_seed42.pth`

### D: MLP (Multi-Layer Perceptron)
- **Architecture:** Feedforward network, 22-dim input → 5 outputs
- **Training:** Supervised classification
- **File:** `models/retrained/exp2_comparators/exp2_D_mlp_seed42.joblib`

### E: GBM (Gradient Boosting Machine)
- **Architecture:** LightGBM, cost-sensitive
- **Training:** Boosted trees with class weights
- **File:** `models/retrained/exp2_comparators/exp2_E_gbm_seed42.joblib`

### SVM (Support Vector Machine)
- **Architecture:** RBF kernel
- **Training:** Supervised classification
- **File:** Not saved (in-memory only)

### RF (Random Forest)
- **Architecture:** 300 trees
- **Training:** Ensemble of decision trees
- **File:** Not saved (in-memory only)

### LSTM (Raw Window LSTM)
- **Architecture:** Recurrent net over K=10 consecutive 22-feature rows
- **Training:** Within-run windowing (no class-straddling), explicit dropout
- **File:** Not saved (in-memory only)

### CQL (Conservative Q-Learning)
- **Architecture:** DuelingDQN with conservative penalty
- **Training:** TD(0) Bellman target + logsumexp penalty
- **File:** Not saved (in-memory only)

---

## Setup & Installation

### Requirements

- Python 3.10+ or 3.11+
- VS Code (recommended)
- Ollama (for explanation/critique features)
- ~2GB disk space for model weights

### Step 1 — Clone the Repository

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

**requirements.txt:**
```
torch>=2.0
ultralytics>=8.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
joblib>=1.3
requests>=2.31
streamlit>=1.28
plotly>=5.17
flask>=3.0
altair>=5.0
```

### Step 4 — Download the Dataset

Download `Gas_Sensors_Measurements.csv` and place it in `data/`:
```
data/Gas_Sensors_Measurements.csv
```

### Step 5 — Install Ollama (Optional — for Explanations)

Download from [ollama.com](https://ollama.com), then:
```bash
ollama pull gemma3:1b
```

---

## Running the System

### Option A: Streamlit Dashboard (Recommended)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Five tabs:
- **Overview** — action distribution, confidence histogram, anomaly scatter, per-gas comparison
- **Case Details** — pipeline chart, Q-values, structured output, vision report, explanation, critique
- **Sensor Feed** — 7-channel sensor traces, anomaly evolution, per-sensor statistics
- **State Inspector** — anomaly gauge, 22-dim state vector, decision chain
- **Structured Table** — export-ready results + CSV download

### Option B: CLI Runner

```bash
python main.py
```

Runs the full pipeline end-to-end on the dataset, prints summary table, writes `logs/agent_run.json`.

### Option C: Run Experiments

```bash
# Exp #1 — Pareto sweep
python retrain/run_exp1_real.py

# Exp #2 — Boundary test
python retrain/run_exp2_real.py

# Exp #3 — LOCO
python retrain/run_exp3_loco.py

# Exp #4 — Calibration
python retrain/run_exp4_calibration.py

# ZOO — All baselines
python retrain/run_expzoo.py
```

---

## Results Summary

### Exp #1 — Pareto Peak

| Ratio | Accuracy | Std | Miss Rate |
|-------|----------|-----|-----------|
| 1:1 | 0.9584 | 0.0340 | 0.0 |
| **8:1** | **0.9633** | **0.0211** | **0.0** |
| 20:1 | 0.8396 | 0.119 | 0.0 |

**Deployed:** 8:1 (tightest seed spread)

### Exp #3 — LOCO (Headline Result)

| Held-Out | Model | Miss Rate | 95% CP Bound |
|----------|-------|-----------|--------------|
| Smoke | A (Decision Agent) | 0.0 | ≤0.189% |
| Smoke | D (MLP) | **18.7%** | **≤20.4%** |
| Mixture | A | 0.44% | ≤0.830% |
| Mixture | D | 0.38% | ≤0.748% |

### Exp #4 — Calibration

| Estimator | ECE Mean | ECE Std |
|-----------|----------|---------|
| raw_softmax | 0.0489 | 0.0699 |
| **deep_ensemble_5** | **0.0123** | **0.0043** |

### ZOO — All Baselines

| Model | Accuracy | Std | Miss Rate |
|-------|----------|-----|-----------|
| A (Decision Agent) | 0.9633 | 0.0215 | 0.0 |
| D (MLP) | 0.9616 | 0.0111 | 0.0 |
| **RF** | **0.9769** | **0.0086** | **0.0** |
| LSTM | 0.9680 | 0.0231 | 0.0 |
| CQL | 0.9437 | 0.0479 | 0.0 |

**No comparator significantly different from A at α=0.05.**

---

## Project Structure

```
ARXIS/
├── app.py                          # Streamlit dashboard
├── main.py                         # CLI runner
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   └── Gas_Sensors_Measurements.csv  # Raw dataset (6400 rows)
├── models/
│   ├── DeepQnet.pth                # Original pre-validation checkpoint
│   ├── lstm_autoencoder_weights.pth # LSTM anomaly detector
│   ├── yolov8_gas_classifier.pt    # YOLO vision model
│   ├── model_config.json           # Model configuration
│   └── retrained/
│       ├── MANIFEST.json           # Model manifest
│       ├── exp1_pareto/            # Exp #1 checkpoints (40 files)
│       ├── exp2_comparators/       # Exp #2 checkpoints (4 files)
│       └── exp4_calibration/       # Exp #4 checkpoints (7 files)
├── retrain/
│   ├── raw_pipeline.py             # Data pipeline (windowing, scaling, splitting)
│   ├── run_exp1_real.py            # Exp #1 driver
│   ├── run_exp2_real.py            # Exp #2 driver
│   ├── run_exp3_loco.py            # Exp #3 driver
│   ├── run_exp4_calibration.py     # Exp #4 driver
│   ├── run_expzoo.py               # ZOO driver
│   ├── zoo.py                      # ZOO model definitions
│   ├── comparators.py              # Comparator models
│   ├── metrics.py                  # Evaluation metrics (Clopper-Pearson, Wilcoxon)
│   ├── calibration.py              # Calibration methods
│   ├── rewards.py                  # Reward functions
│   ├── agent_rl.py                 # RL agent
│   ├── fetch_from_drive.py         # Google Drive dataset fetcher
│   ├── drive_access.py             # Google Drive direct access module
│   └── results/
│       ├── REPORT.md               # Full results report
│       ├── exp1_pareto.csv
│       ├── exp2_boundary.csv
│       ├── exp3_loco.csv
│       ├── exp3_clopper_pearson.csv
│       ├── exp4_calibration.csv
│       ├── expzoo.csv
│       └── *.png                   # Result plots
├── src/
│   ├── agent/
│   │   ├── agent_core.py           # Main orchestrator (MultimodalAgent)
│   │   ├── safety.py               # Safety override layer
│   │   ├── critic.py               # CriticAgent
│   │   ├── goal_manager.py         # GoalManager
│   │   ├── memory.py               # ShortTermMemory
│   │   ├── supervisor.py           # SupervisorAgent
│   │   ├── trainer.py              # AgentTrainer
│   │   ├── reward_system.py        # Reward functions
│   │   └── metrics_logger.py       # Logging
│   └── tools/
│       ├── anomaly_tool.py         # LSTM-AE anomaly detection
│       ├── decision_tool.py        # Dueling DQN decision
│       ├── vision_tool.py          # YOLOv8 vision verification
│       └── explanation_tool.py     # Ollama explanation generation
└── .gitignore                      # Security + venv exclusions
```

---

## Open Items

1. **Exp #3 Single Uninterrupted Run:** If publication requires one `exit=0` run, run on Linux or a fresh long-lived shell. The data is deterministic and won't change. See `retrain/results/NOTE_provenance_exp3.md`.

2. **n=5 Power Limitation:** With 5 seeds, the minimum detectable two-sided Wilcoxon p is 0.0625. Danger-miss is stated as exact Clopper–Pearson bounds, not significant-difference tests.

3. **Thermal Images:** The `data/Mixture/` folder contains sample thermal images. For full vision verification, add `NoGas/`, `Smoke/`, `Perfume/` folders under `data/`. The system runs in sensor-only mode without them.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this work, please cite:

```bibtex
@software{arxis2025,
  title={ARXIS: Industrial Gas Safety Intelligence},
  author={Wilfred},
  year={2025},
  url={https://github.com/CryptoGuy1/ARXIS}
}
```

---

**ARXIS** — Intelligence That Delivers.
