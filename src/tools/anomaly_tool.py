"""Runtime anomaly tool: the pretrained NoGas LSTM autoencoder shipped in models/.

The shipped weights were trained in the author's Colab notebook (Autoencoder_Model),
and this module reproduces that notebook's definition exactly:

  - input scaler: MinMaxScaler fitted on the NoGas sensor rows of the corpus;
  - network: an LSTM encoder whose final hidden state is repeated across the
    20 time steps and fed to an LSTM decoder, then a linear output layer;
  - score: mean squared reconstruction error over the full 20-step window.

With these definitions and the shipped weights, the notebook's in-sample NoGas
threshold and ROC-AUC are reproduced (see test_runtime_anomaly.py).

This is the runtime path only. The reported experiments do not use these weights:
retrain/raw_pipeline.py refits an autoencoder inside each training partition with
its own scaler and forward pass (Section S21 of the supporting information).
"""

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

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
        # Notebook definition: the encoder's final hidden state is repeated over
        # every time step and passed to the decoder (not the encoder's per-step outputs).
        _, (h, _) = self.encoder(x)
        rep = h.repeat(x.shape[1], 1, 1).permute(1, 0, 2)
        decoded, _ = self.decoder(rep)
        return self.output_layer(decoded)


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
        return MinMaxScaler().fit(normal)

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
