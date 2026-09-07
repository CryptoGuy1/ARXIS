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
- [Setup \& Installation](#setup--installation)
- [Running the System](#running-the-system)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Open Items](#open-items)

---

## Overview

ARXIS detects four gas conditions from MQ sensor arrays and selects one of five safety actions:

| Action | Description |
|--------|-------------|
| **Monitor** | Continue normal observation |
| **Increase Sampling** | Collect more sensor data |
| **Request Verification** | Ask for human confirmation |
| **Raise Alarm** | Alert operators |
| **Emergency Shutdown** | Immediate shutdown |

It combines time-series analysis, anomaly detection, reinforcement learning, thermal-image verification, and operator-facing explanations into one system.

### Gas Classes

| Gas | Hazard | Correct Action |
|-----|--------|----------------|
| NoGas | Safe | Monitor |
| Smoke | **Danger** | Raise Alarm |
| Mixture | **Danger** | Emergency Shutdown |
| Perfume | Low-risk VOC | Increase Sampling / Request Verification |

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

Find the best trade-off between danger-miss and false-alarm.

| Cost Ratio | Accuracy | Std |
|------------|----------|-----|
| 1:1 | 0.9584 | 0.0340 |
| 6:1 | 0.9658 | 0.0274 |
| **8:1** | **0.9633** | **0.0211** |
| 20:1 | 0.8396 | 0.119 |

**Deployed:** 8:1 (lowest variance).

### Exp 2 — Boundary Stress Test

Compare models on uncertain cases.

| Model | Accuracy | Std |
|-------|----------|-----|
| Decision Agent | 0.9584 | 0.0340 |
| Plain DQN | 0.9563 | 0.0378 |
| MLP | 0.9616 | 0.0111 |
| GBM | 0.9634 | 0.0211 |

All models achieve ≈0 danger-miss on this separable dataset.

### Exp 3 — Leave-One-Class-Out (Headline Result)

Train on 3 gases, test on the 4th. The MLP misses **18.7%** of held-out Smoke danger rows while the Decision Agent misses **0%**.

| Held-Out | Model | Miss Rate | 95% Bound |
|----------|-------|-----------|-----------|
| Smoke | Decision Agent | 0.0% | ≤0.19% |
| Smoke | **MLP** | **18.7%** | **≤20.4%** |
| Smoke | Plain DQN | 0.0% | ≤0.19% |
| Smoke | GBM | 0.0% | ≤0.19% |
| Mixture | Decision Agent | 0.44% | ≤0.83% |
| Mixture | MLP | 0.38% | ≤0.75% |

### Exp 4 — Confidence Calibration

| Estimator | ECE | Std |
|-----------|-----|-----|
| Raw softmax | 0.0489 | 0.0699 |
| MC dropout | 0.0533 | 0.0692 |
| Temp scaling | 0.0319 | 0.0397 |
| **Deep ensemble** | **0.0123** | **0.0043** |

### Model Comparison (ZOO)

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

## Setup \& Installation

### Requirements

- Python 3.10+
- ~2 GB disk space

### Install

```bash
git clone https://github.com/CryptoGuy1/ARXIS.git
cd ARXIS
python -m venv .venv-retrain
.venv-retrain\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Dataset

Place `Gas_Sensors_Measurements.csv` in `data/`.

### Ollama (Optional)

For explanation features:
```bash
ollama pull gemma3:1b
```

---

## Running the System

### Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Five tabs: Overview, Case Details, Sensor Feed, State Inspector, Table.

### CLI

```bash
python main.py
```

### Experiments

```bash
python retrain/run_exp1_real.py   # Pareto
python retrain/run_exp2_real.py   # Boundary
python retrain/run_exp3_loco.py   # LOCO
python retrain/run_exp4_calibration.py  # Calibration
python retrain/run_expzoo.py      # All baselines
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
├── requirements.txt
├── README.md
├── data/
│   └── Gas_Sensors_Measurements.csv
├── models/
│   ├── DeepQnet.pth
│   ├── lstm_autoencoder_weights.pth
│   ├── yolov8_gas_classifier.pt
│   └── retrained/
│       ├── MANIFEST.json
│       ├── exp1_pareto/        # 40 checkpoints
│       ├── exp2_comparators/   # 4 checkpoints
│       └── exp4_calibration/   # 7 checkpoints
├── retrain/
│   ├── raw_pipeline.py         # Data pipeline
│   ├── run_exp*.py             # Experiment drivers
│   ├── zoo.py, comparators.py, metrics.py, calibration.py
│   └── results/
│       ├── REPORT.md
│       ├── *.csv, *.png
│       └── NOTE_provenance_exp3.md
└── src/
    ├── agent/   # Core agentic layer
    └── tools/   # Anomaly, Decision, Vision, Explanation
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
