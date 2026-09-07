"""Raw-data pipeline: build the Decision Agent's 22-feature training set
from the REAL raw Gas_Sensors_Measurements.csv.

Each row = a 20-step window ending at index i, labelled by the gas at i+19
(the last reading). Features:
    [anomaly, current(7), delta(7), std(7)]   (22 dims)

Audit fixes applied (see retrain/results/REPORT.md "Audit Follow-up"):
  A1  anomaly AE now LOADS models/lstm_autoencoder_weights.pth (verified arch).
  A2  anomaly is the reconstruction error of the FULL 20-step window, not a
      single reshaped reading.
  A4  sensors are scaled AFTER the train/test split (fit on train only); the
      leaked all-rows scaler was removed.
  A5  windows that straddle a class boundary are dropped (no split-label mix).
  A8  the AE is constructed/loaded deterministically and the feature cache is
      versioned (rebuilt if the data or this code changes).

The block-wise leakage-safe holdout (per-class contiguous blocks, last 20%
test, gap of WINDOW_SIZE excluded) is preserved and now seeded per call.
"""

import os
import sys
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(ROOT, "data", "Gas_Sensors_Measurements.csv")
WINDOW_SIZE = 20
SENSOR_COLS = ["MQ2", "MQ3", "MQ5", "MQ6", "MQ7", "MQ8", "MQ135"]
DELTA_COLS = [f"d{c}" for c in SENSOR_COLS]
STD_COLS = [f"s{c}" for c in SENSOR_COLS]
ALL_FEAT_COLS = SENSOR_COLS + DELTA_COLS + STD_COLS
GAS_MAP = {"NoGas": 0, "Smoke": 1, "Mixture": 2, "Perfume": 3}
GAS_ORDER = ["NoGas", "Smoke", "Mixture", "Perfume"]

# Version stamp: bump this if the windowing/scaling logic changes so any cached
# real_features.csv is rebuilt instead of silently reused (audit A8).
CACHE_VERSION = "v3-rawpipeline-a1-a2-a4-a5"


# ----------------------- LSTM Autoencoder (NoGas-trained) -----------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=7, hidden_size=32):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return self.output_layer(decoded)


def _ae_weights_path():
    return os.path.join(ROOT, "models", "lstm_autoencoder_weights.pth")


def build_anomaly_model(seed=0):
    """Load the pretrained NoGas LSTM-AE (audit A1). Deterministic: a fixed
    seed is set before construction (the loaded weights overwrite init anyway)
    and the file path is resolved relative to the project root (no hardcoded
    Windows path).

    Returns (model, scaler) where `scaler` standardises SENSOR_COLS using the
    NoGas subset (the distribution the AE was trained on)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    df = pd.read_csv(RAW_CSV)
    normal = df[df["Gas"] == "NoGas"][SENSOR_COLS].reset_index(drop=True)
    scaler = StandardScaler().fit(normal.values)

    model = LSTMAutoencoder().eval()
    wpath = _ae_weights_path()
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            f"AE weights missing at {wpath}; cannot load pretrained anomaly model (A1).")
    sd = torch.load(wpath, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)  # A1: previously this line never existed
    return model, scaler


def _anomaly_of_window(model, scaler, win_raw):
    """Reconstruction error of the FULL 20-step window (audit A2).

    win_raw: np.ndarray shape (WINDOW, 7) of raw sensor readings.
    Returns a single float = mean squared reconstruction error over the window.
    """
    x = torch.tensor(scaler.transform(win_raw).reshape(1, WINDOW_SIZE, 7),
                     dtype=torch.float32)
    with torch.no_grad():
        recon = model(x)
        return float(torch.mean((x - recon) ** 2).item())


def _data_version():
    """Hash of (data file mtime/size) + this module source, for cache validity."""
    try:
        st = os.stat(RAW_CSV)
        data_tag = f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        data_tag = "missing"
    src = open(__file__, "rb").read()
    return hashlib.sha256((data_tag + src.hex() + CACHE_VERSION).encode()).hexdigest()[:16]


def build_dataset(anomaly_model, anomaly_scaler, cache_path=None):
    """Build the full 22-feature table from raw CSV (per-window anomaly, RAW
    current/delta/std — scaling happens later, on the train split only).

    Boundary-straddling windows (audit A5) are dropped: a window is kept only
    if every reading in it (plus its label row) shares the same gas class.
    """
    meta_path = cache_path + ".meta.json" if cache_path else None
    if cache_path and os.path.exists(cache_path) and meta_path and os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
            if meta.get("version") == _data_version():
                return pd.read_csv(cache_path)
        except Exception:
            pass  # rebuild on any meta/cache error

    df = pd.read_csv(RAW_CSV)
    sensors = df[SENSOR_COLS].values
    gas_arr = df["Gas"].values
    img_arr = df["Corresponding Image Name"].values

    # A5: keep windows whose readings all share one class (drop straddlers)
    keep_starts = [i for i in range(len(df) - WINDOW_SIZE + 1)
                   if np.all(gas_arr[i:i + WINDOW_SIZE] == gas_arr[i + WINDOW_SIZE - 1])]

    if keep_starts:
        wins = np.stack([sensors[i:i + WINDOW_SIZE] for i in keep_starts], 0)  # (K,20,7)
        # vectorised anomaly = mean reconstruction error over the full window (A2)
        x = torch.tensor(anomaly_scaler.transform(wins.reshape(-1, 7)).reshape(-1, WINDOW_SIZE, 7),
                         dtype=torch.float32)
        with torch.no_grad():
            recon = anomaly_model(x)
            anom = torch.mean((x - recon) ** 2, dim=(1, 2)).numpy()
    else:
        anom = np.array([])

    data = []
    for k, i in enumerate(keep_starts):
        lab = gas_arr[i + WINDOW_SIZE - 1]
        win = sensors[i:i + WINDOW_SIZE]
        current = win[-1]
        delta = win[-1] - win[0]
        std = win.std(axis=0)
        data.append([lab, GAS_MAP[lab], img_arr[i + WINDOW_SIZE - 1], float(anom[k])]
                    + current.tolist() + delta.tolist() + std.tolist())

    columns = ["label", "gas_id", "img_name", "anomaly"] + ALL_FEAT_COLS
    out = pd.DataFrame(data, columns=columns)
    if cache_path:
        out.to_csv(cache_path, index=False)
        if meta_path:
            json.dump({"version": _data_version(), "rows": len(out),
                       "cache_version": CACHE_VERSION}, open(meta_path, "w"))
    return out


def block_wise_holdout(df, test_frac=0.2, gap=WINDOW_SIZE, seed=None):
    """Block-wise leakage-safe holdout (audit A7: now seeded AND seed-varying).

    For each class, hold out a contiguous block of `test_frac` of its rows as
    test, separated from BOTH training segments by `gap` excluded rows (so no
    training window neighbours a test window). Class blocks stay pure (no class
    leak). The held-out block's *position* is chosen per seed, so different seeds
    produce genuinely different train/test partitions (the reported ± now reflects
    partition-variance, not just model-init variance).

    With seed=None the held-out block is the TAIL (deterministic default).
    """
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for label in GAS_ORDER:
        block = df.index[df["label"] == label].tolist()
        n = len(block)
        n_test = max(1, int(round(n * test_frac)))
        n_gap = min(gap, max(0, n - n_test))  # exclusion band, clamped
        # Choose the held-out block's START position. It must leave room for the
        # test block + gap band inside [0, n], so legal starts are in
        # [0, n - n_test - n_gap]. seed=None -> tail (start = n - n_test - n_gap).
        hi = max(0, n - n_test - n_gap)
        if seed is not None and hi > 0:
            start = int(rng.integers(0, hi + 1))
        else:
            start = hi
        # train = block before the gap band and after the gap band (two segments)
        train_idx.extend(block[:start])
        train_idx.extend(block[start + n_test + n_gap:])
        test_idx.extend(block[start + n_gap:start + n_gap + n_test])
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


def split_and_prepare(df, seed=None, test_frac=0.2, gap=WINDOW_SIZE):
    """Audit A7 helper: split (seed-varying) then scale on train only (A4).

    Combines `block_wise_holdout` + `prepare` so a driver can call this once
    inside its SEEDS loop, keeping the partition in lock-step with the seed.
    Returns (tr, te, Xtr, gtr, yact_tr, Xte, gte).
    """
    from retrain import rewards as RW  # local import to avoid circular deps
    train_df, test_df = block_wise_holdout(df, test_frac=test_frac, gap=gap, seed=seed)
    train_df, test_df, _, _ = prepare(train_df, test_df)
    Xtr, gtr = to_arrays(train_df)
    Xte, gte = to_arrays(test_df)
    yact_tr = np.array([RW.rule_oracle_action(int(g)) for g in gtr])
    return train_df, test_df, Xtr, gtr, yact_tr, Xte, gte


def prepare(train_df, test_df):
    """Scale features on the TRAIN split only (audit A4: no all-rows scaler).

    Returns (train_df, test_df, feat_scaler, (p1, p99)) with ALL_FEAT_COLS
    standardised and `anomaly` min-max clipped to [0,1] using train percentiles.
    """
    feat_scaler = StandardScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[ALL_FEAT_COLS] = feat_scaler.fit_transform(train_df[ALL_FEAT_COLS])
    test_df[ALL_FEAT_COLS] = feat_scaler.transform(test_df[ALL_FEAT_COLS])
    p1 = float(np.percentile(train_df["anomaly"], 1))
    p99 = float(np.percentile(train_df["anomaly"], 99))
    def norm(a):
        return ((a - p1) / (p99 - p1 + 1e-8)).clip(0, 1)
    train_df["anomaly"] = norm(train_df["anomaly"].values)
    test_df["anomaly"] = norm(test_df["anomaly"].values)
    return train_df, test_df, feat_scaler, (p1, p99)


def get_state(row):
    return np.array([row["anomaly"]] + [row[c] for c in SENSOR_COLS]
                    + [row[c] for c in DELTA_COLS] + [row[c] for c in STD_COLS],
                    dtype=np.float32)


def to_arrays(df):
    X = np.stack([get_state(r) for _, r in df.iterrows()], 0)
    y = df["gas_id"].astype(int).values
    return X, y


def anomaly_by_class(anomaly_model, anomaly_scaler):
    """Audit A3: empirical check of whether anomaly tracks hazard.

    Returns a dict gas -> mean reconstruction error over that class's windows,
    computed on the raw CSV. Use this to verify (or honestly report) that the
    anomaly feature is hazard-correlated, not anti-correlated.
    """
    df = pd.read_csv(RAW_CSV)
    sensors = df[SENSOR_COLS].values
    gas_arr = df["Gas"].values
    out = {}
    for g in GAS_ORDER:
        starts = [i for i in range(len(df) - WINDOW_SIZE + 1)
                  if gas_arr[i + WINDOW_SIZE - 1] == g
                  and np.all(gas_arr[i:i + WINDOW_SIZE] == g)]
        if not starts:
            out[g] = float("nan")
            continue
        wins = np.stack([sensors[i:i + WINDOW_SIZE] for i in starts], 0)
        x = torch.tensor(anomaly_scaler.transform(wins.reshape(-1, 7)).reshape(-1, WINDOW_SIZE, 7),
                         dtype=torch.float32)
        with torch.no_grad():
            recon = anomaly_model(x)
            errs = torch.mean((x - recon) ** 2, dim=(1, 2)).numpy()
        out[g] = float(np.mean(errs))
    return out
