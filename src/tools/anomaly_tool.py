"""Anomaly tool — aligned with the retrain/ validation harness (audit A1/A2/fix #2).

The live inference path (src/) previously fed a SINGLE unscaled 7-value reading
into the AE and returned its point reconstruction error. That produced values in
the 1e5–1e6 range with an INVERTED class ordering (Perfume most anomalous), which
diverged from the retrain/ feature the DQN was actually trained on. This rewrite
mirrors retrain/raw_pipeline.py exactly:

  - load the pretrained NoGas LSTM-AE weights (audit A1),
  - scale the 20-step window on the NoGas distribution (audit A4 logic),
  - reconstruction error = mean over the full window,
so the anomaly feature the agent sees matches the trained 22-feature state
(low~NoGas, high~Smoke; ordering NoGas < Perfume < Mixture < Smoke).
"""

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SENSOR_COLS = ["MQ2", "MQ3", "MQ5", "MQ6", "MQ7", "MQ8", "MQ135"]
WINDOW_SIZE = 20


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=7, hidden_size=32):
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        out = self.output_layer(decoded)
        return out


class AnomalyTool:
    def __init__(self, model_path, raw_csv=None):
        # build model + load pretrained NoGas weights (audit A1 — was missing in src/)
        self.model = LSTMAutoencoder()
        weights = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(weights)
        self.model.eval()

        # Fit the scaler on the NoGas distribution (same as retrain/raw_pipeline.build_anomaly_model)
        if raw_csv is None:
            raw_csv = os.path.join(ROOT, "data", "Gas_Sensors_Measurements.csv")
        self.scaler = self._fit_nogas_scaler(raw_csv)
        self.window = []  # rolling buffer of recent 7-value sensor readings

    @staticmethod
    def _fit_nogas_scaler(raw_csv):
        import pandas as pd
        df = pd.read_csv(raw_csv)
        normal = df[df["Gas"] == "NoGas"][SENSOR_COLS].values
        return StandardScaler().fit(normal)

    def update(self, sensor_array):
        """Push a new 7-value reading into the rolling 20-step window."""
        self.window.append(np.asarray(sensor_array, dtype=float))
        if len(self.window) > WINDOW_SIZE:
            self.window.pop(0)

    def compute_from_window(self, sensor_window: np.ndarray) -> float:
        """Compute anomaly from a provided full window (does NOT modify rolling window).

        sensor_window: np.ndarray shape (N, 7) where N >= 1.
        Returns mean reconstruction error over the window.
        """
        win = np.asarray(sensor_window, dtype=float)
        if win.ndim == 1:
            win = win.reshape(1, -1)
        x = torch.tensor(self.scaler.transform(win.reshape(-1, 7)).reshape(1, win.shape[0], 7),
                         dtype=torch.float32)
        with torch.no_grad():
            recon = self.model(x)
        return float(torch.mean((x - recon) ** 2).item())

    def compute(self, sensor_array=None):
        """Return the window-level reconstruction-error anomaly (matches retrain/).

        If `sensor_array` is given it is appended to the rolling window first.
        Returns float; requires at least WINDOW_SIZE readings (call update()/feed
        the agent's rolling buffer first). Falls back to a single-scaled reading
        error if the window is not yet full, so early steps never crash.
        """
        if sensor_array is not None:
            self.update(sensor_array)
        win = np.asarray(self.window, dtype=float)
        if win.shape[0] == 0:
            return 0.0
        if win.shape[0] < WINDOW_SIZE:
            # not enough history yet: scale the available reading(s) and score them
            x = torch.tensor(self.scaler.transform(win.reshape(-1, 7)).reshape(1, win.shape[0], 7),
                             dtype=torch.float32)
            with torch.no_grad():
                recon = self.model(x)
            return float(torch.mean((x - recon) ** 2).item())
        x = torch.tensor(self.scaler.transform(win.reshape(-1, 7)).reshape(1, WINDOW_SIZE, 7),
                         dtype=torch.float32)
        with torch.no_grad():
            recon = self.model(x)
        # mean reconstruction error over the full 20-step window (audit A2)
        return float(torch.mean((x - recon) ** 2).item())
