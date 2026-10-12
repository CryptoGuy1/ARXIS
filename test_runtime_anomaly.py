"""Check that the runtime anomaly tool reproduces the autoencoder notebook.

The shipped weights (models/lstm_autoencoder_weights.pth) were trained in the
Colab notebook Autoencoder_Model. src/tools/anomaly_tool.py must score the same
way the notebook did: MinMax scaling fitted on NoGas rows, a repeated-hidden
decoder, and mean squared error over each 20-step window. This test recomputes
the notebook's in-sample numbers and fails if they drift.

Expected values are the notebook's stored outputs: ROC-AUC 0.9634734, mean
reconstruction error on NoGas windows 0.0027229267, and the 95th-percentile NoGas
threshold 0.00700815. The notebook ran on a GPU and this check runs on a CPU, so
the test allows a relative difference of about 1e-3 rather than bit equality.

    python3 test_runtime_anomaly.py

This check does not touch the reported experiments. Those refit their own
autoencoder per training partition (retrain/raw_pipeline.py).
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src", "tools"))

from anomaly_tool import LSTMAutoencoder, SENSOR_COLS, WINDOW_SIZE  # noqa: E402

WEIGHTS = os.path.join(ROOT, "models", "lstm_autoencoder_weights.pth")
CSV = os.path.join(ROOT, "data", "Gas_Sensors_Measurements.csv")

EXPECTED_AUC = 0.9634733649789029
EXPECTED_MEAN_NORMAL = 0.0027229267
EXPECTED_THRESHOLD = 0.00700815
REL_TOL = 1e-3


def main():
    df = pd.read_csv(CSV)
    normal = df[df["Gas"] == "NoGas"][SENSOR_COLS].reset_index(drop=True).values
    scaler = MinMaxScaler().fit(normal)

    model = LSTMAutoencoder(input_size=7, hidden_size=32)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.eval()

    scaled = scaler.transform(df[SENSOR_COLS].values)
    windows = np.stack([scaled[i:i + WINDOW_SIZE] for i in range(len(scaled) - WINDOW_SIZE)])
    x = torch.tensor(windows, dtype=torch.float32)
    with torch.no_grad():
        recon = model(x)
    scores = torch.mean((x - recon) ** 2, dim=(1, 2)).numpy()

    labels = df["Gas"].iloc[WINDOW_SIZE:].values
    y = np.array([0 if g == "NoGas" else 1 for g in labels])

    auc = roc_auc_score(y, scores)
    mean_normal = scores[y == 0].mean()
    threshold = np.percentile(scores[y == 0], 95)

    print(f"ROC-AUC {auc:.7f} (notebook {EXPECTED_AUC:.7f})")
    print(f"mean normal {mean_normal:.10f} (notebook {EXPECTED_MEAN_NORMAL:.10f})")
    print(f"threshold {threshold:.8f} (notebook {EXPECTED_THRESHOLD:.8f})")

    def rel(a, b):
        return abs(a - b) / abs(b)

    assert rel(auc, EXPECTED_AUC) < REL_TOL, "ROC-AUC does not match the notebook"
    assert rel(mean_normal, EXPECTED_MEAN_NORMAL) < REL_TOL, "mean NoGas error does not match"
    assert rel(threshold, EXPECTED_THRESHOLD) < REL_TOL, "threshold does not match"
    print("runtime anomaly tool reproduces the autoencoder notebook: OK")


if __name__ == "__main__":
    main()
