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
CACHE_VERSION = "v4-rawpipeline-twosided-embargo-partition-anomaly"


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
    """DEPRECATED, and retained only so that the pre-correction result files can
    still be regenerated for the before/after comparison.

    This fits the autoencoder's input scaler on EVERY NoGas row in the corpus and
    loads a checkpoint that was itself trained on every NoGas row, both before any
    train/test partition exists. The anomaly feature it produces therefore carries
    test-period exposure (reviewer comment 4.2). Live experiments must instead use
    `partition_anomaly`, which fits the scaler and trains the autoencoder inside
    the training partition.

    Returns (model, scaler) where `scaler` standardises SENSOR_COLS using the
    NoGas subset (the distribution the checkpoint was trained on)."""
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


# --------------------- partition-local anomaly model (fix 4.2) ---------------------
# Raw windows, keyed by the `win_id` column of the feature table, so that the
# anomaly feature can be refitted inside whatever training partition a driver has
# just formed. Populated by build_dataset and cached on disk beside it.
_WIN_CACHE = {"windows": None, "version": None}

AE_EPOCHS = 30
AE_BATCH = 64
AE_LR = 1e-3


def _windows_path(cache_path):
    return (cache_path + ".windows.npy") if cache_path else None


def raw_windows():
    """(K, WINDOW_SIZE, 7) array of raw sensor windows, row-aligned with win_id."""
    if _WIN_CACHE["windows"] is None:
        raise RuntimeError("raw windows not loaded; call build_dataset() first")
    return _WIN_CACHE["windows"]


def train_anomaly_on(win_raw, seed=0, epochs=AE_EPOCHS):
    """Fit an input scaler and train an LSTM autoencoder on `win_raw` alone.

    win_raw: (m, WINDOW_SIZE, 7) raw NoGas windows drawn from ONE training
    partition. Nothing outside that partition is touched, which is the whole
    point of this function.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler().fit(win_raw.reshape(-1, 7))
    x = torch.tensor(scaler.transform(win_raw.reshape(-1, 7)).reshape(-1, WINDOW_SIZE, 7),
                     dtype=torch.float32)
    model = LSTMAutoencoder()
    opt = torch.optim.Adam(model.parameters(), lr=AE_LR)
    lossf = nn.MSELoss()
    n = x.shape[0]
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, AE_BATCH):
            b = x[perm[i:i + AE_BATCH]]
            opt.zero_grad()
            loss = lossf(model(b), b)
            loss.backward()
            opt.step()
    model.eval()
    return model, scaler


def score_anomaly(model, scaler, win_raw, batch=512):
    """Mean squared reconstruction error per window, under a fitted (model, scaler)."""
    out = np.empty(len(win_raw), dtype=np.float64)
    for i in range(0, len(win_raw), batch):
        chunk = win_raw[i:i + batch]
        x = torch.tensor(scaler.transform(chunk.reshape(-1, 7)).reshape(-1, WINDOW_SIZE, 7),
                         dtype=torch.float32)
        with torch.no_grad():
            recon = model(x)
            out[i:i + batch] = torch.mean((x - recon) ** 2, dim=(1, 2)).numpy()
    return out


_AE_MEMO = {}


def partition_anomaly(train_df, *other_dfs, seed=0, nominal="NoGas"):
    """Refit the anomaly path inside `train_df` and rescore every frame given.

    Fixes reviewer comment 4.2. The autoencoder's input scaler and the
    autoencoder itself are fitted on the nominal (NoGas) windows of the TRAINING
    partition only; the resulting model then scores the training frame and any
    other frames passed in (test, held-out class, perturbed copies). No frame
    outside the training partition contributes to fitting.

    Returns copies of (train_df, *other_dfs) with the `anomaly` column replaced
    by raw reconstruction error. Downstream percentile normalisation stays where
    it already is, in `prepare`, and is likewise fitted on training rows only.
    """
    wins = raw_windows()
    nom_ids = train_df.loc[train_df["label"] == nominal, "win_id"].astype(int).values
    if len(nom_ids) == 0:
        raise ValueError(f"no {nominal} windows in the training partition; "
                         "the anomaly model has nothing nominal to learn")
    key = (seed, hash(nom_ids.tobytes()))
    if key not in _AE_MEMO:
        _AE_MEMO[key] = train_anomaly_on(wins[nom_ids], seed=seed)
    model, scaler = _AE_MEMO[key]

    out = []
    for d in (train_df,) + other_dfs:
        d = d.copy()
        ids = d["win_id"].astype(int).values
        d["anomaly"] = score_anomaly(model, scaler, wins[ids])
        out.append(d)
    return out[0] if not other_dfs else tuple(out)


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
    wpath = _windows_path(cache_path)
    if (cache_path and os.path.exists(cache_path) and meta_path
            and os.path.exists(meta_path) and wpath and os.path.exists(wpath)):
        try:
            meta = json.load(open(meta_path))
            if meta.get("version") == _data_version():
                _WIN_CACHE["windows"] = np.load(wpath)
                _WIN_CACHE["version"] = meta.get("version")
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
        data.append([k, lab, GAS_MAP[lab], img_arr[i + WINDOW_SIZE - 1], float(anom[k])]
                    + current.tolist() + delta.tolist() + std.tolist())

    columns = ["win_id", "label", "gas_id", "img_name", "anomaly"] + ALL_FEAT_COLS
    out = pd.DataFrame(data, columns=columns)
    # Keep the raw windows beside the feature table so that the anomaly path can
    # be refitted inside a training partition (fix 4.2).
    _WIN_CACHE["windows"] = wins if keep_starts else np.zeros((0, WINDOW_SIZE, 7))
    _WIN_CACHE["version"] = _data_version()
    if cache_path:
        out.to_csv(cache_path, index=False)
        np.save(_windows_path(cache_path), _WIN_CACHE["windows"])
        if meta_path:
            json.dump({"version": _data_version(), "rows": len(out),
                       "cache_version": CACHE_VERSION}, open(meta_path, "w"))
    return out


def block_wise_holdout(df, test_frac=0.2, gap=WINDOW_SIZE, seed=None):
    """Block-wise leakage-safe holdout with a TWO-SIDED embargo.

    For each class, hold out a contiguous block of `test_frac` of its rows as
    test, separated from the training data by `gap` excluded rows on EACH side,
    so no training window shares a raw reading with any test window. Class blocks
    stay pure (no class leak). The held-out block's *position* is chosen per seed,
    so different seeds produce genuinely different train/test partitions (the
    reported +/- reflects partition variance, not just model-init variance).

    Layout within one class block of n rows, for a test block starting at `lo`:

        [0, lo-gap)                 train (leading segment)
        [lo-gap, lo)                embargo, discarded
        [lo, lo+n_test)             test
        [lo+n_test, lo+n_test+gap)  embargo, discarded
        [lo+n_test+gap, n)          train (trailing segment)

    With seed=None the held-out block sits at the TAIL (deterministic default),
    in which case only the leading embargo exists because there is nothing after
    the test block to separate from.

    FIX (reviewer comment 4.1): the previous implementation placed the embargo
    only before the test block and resumed training immediately after it, so the
    first post-test training window could share up to WINDOW_SIZE-1 raw readings
    with the last test window. Per-class training rows drop from
    n - n_test - gap = 1,245 to n - n_test - 2*gap = 1,225 as a result, except
    for a tail-anchored block, which is unchanged.
    """
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for label in GAS_ORDER:
        block = df.index[df["label"] == label].tolist()
        n = len(block)
        n_test = max(1, int(round(n * test_frac)))
        n_gap = min(gap, max(0, (n - n_test) // 2))  # per-side band, clamped
        # The test block start `lo` must leave a full embargo on each side that
        # has training data beside it. Legal starts run over [n_gap, n - n_test].
        hi = max(n_gap, n - n_test - n_gap)
        if seed is not None and hi > n_gap:
            lo = int(rng.integers(n_gap, hi + 1))
        else:
            lo = hi  # tail-anchored default
        train_idx.extend(block[:max(0, lo - n_gap)])
        train_idx.extend(block[lo + n_test + n_gap:])
        test_idx.extend(block[lo:lo + n_test])
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


def split_and_prepare(df, seed=None, test_frac=0.2, gap=WINDOW_SIZE):
    """Audit A7 helper: split (seed-varying) then scale on train only (A4).

    Combines `block_wise_holdout` + `prepare` so a driver can call this once
    inside its SEEDS loop, keeping the partition in lock-step with the seed.
    Returns (tr, te, Xtr, gtr, yact_tr, Xte, gte).
    """
    from retrain import rewards as RW  # local import to avoid circular deps
    train_df, test_df = block_wise_holdout(df, test_frac=test_frac, gap=gap, seed=seed)
    train_df, test_df, _, _ = prepare(train_df, test_df, anomaly_seed=(seed or 0))
    Xtr, gtr = to_arrays(train_df)
    Xte, gte = to_arrays(test_df)
    yact_tr = np.array([RW.rule_oracle_action(int(g)) for g in gtr])
    return train_df, test_df, Xtr, gtr, yact_tr, Xte, gte


def prepare(train_df, test_df, anomaly_seed=0, refit_anomaly=True):
    """Scale features on the TRAIN split only (audit A4: no all-rows scaler).

    Returns (train_df, test_df, feat_scaler, (p1, p99)) with ALL_FEAT_COLS
    standardised and `anomaly` min-max clipped to [0,1] using train percentiles.

    Fix 4.2: the anomaly column is also recomputed here, from an autoencoder and
    input scaler fitted on the training partition's nominal windows alone. Pass
    refit_anomaly=False only to reproduce the pre-correction numbers.
    """
    if refit_anomaly and "win_id" in train_df.columns:
        train_df, test_df = partition_anomaly(train_df, test_df, seed=anomaly_seed)
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
