# ARXIS: thermal classifier (YOLOv8n-cls) under two splits — Kaggle GPU run
import os, json, random, shutil, subprocess, sys
import glob as _glob
import numpy as np
import pandas as pd

# Settings. Environment-variable overrides exist only so the same code can be smoke-tested locally.
SPLIT = os.environ.get("SPLIT", "randomized")        # "blockwise" or "randomized"
assert SPLIT in ("blockwise", "randomized"), f"SPLIT must be 'blockwise' or 'randomized', got {SPLIT!r}"
SEED = int(os.environ.get("SEED", "42"))            # blockwise: position of the held-out block; randomized: which frames are held out
TEST_FRAC = 0.2                                     # held-out share of each class
GAP = 20                                            # blockwise only: frames discarded on each side of the test block (WINDOW_SIZE in the sensor pipeline)
GAS_ORDER = ["NoGas", "Smoke", "Mixture", "Perfume"]  # same order as retrain/raw_pipeline.py; the random draws depend on it
HAZARD = ["Smoke", "Mixture"]                       # hazardous surrogate classes, as the paper states; used only for the hazard-miss count
EPOCHS = int(os.environ.get("EPOCHS", "50"))        # S1: 50 epochs
IMGSZ = int(os.environ.get("IMGSZ", "224"))         # S1: image size 224
BATCH = int(os.environ.get("BATCH", "32"))          # S1: batch size 32
DEVICE = os.environ.get("DEVICE", "0")              # GPU 0 on Kaggle; "cpu" for local tests
RUN_NAME = f"thermal_{SPLIT}_seed{SEED}"

# --- Locate the inputs. Kaggle may rename folders (spaces become underscores), so search for them. ---
_INPUT = "/kaggle/input"

def _find_csv():
    cands = _glob.glob(f"{_INPUT}/**/Gas_Sensors_Measurements*.csv", recursive=True)
    if not cands:
        raise FileNotFoundError("Gas_Sensors_Measurements.csv not found under /kaggle/input")
    return cands[0]

def _find_thermal_root():
    matches = _glob.glob(f"{_INPUT}/**/Thermal*Camera*Images", recursive=True)
    if not matches:
        matches = _glob.glob(f"{_INPUT}/**/NoGas", recursive=True)
    if not matches:
        raise FileNotFoundError("Thermal Camera Images folder not found under /kaggle/input")
    return matches[0]

CSV_PATH = os.environ.get("SENSOR_CSV") or _find_csv()
DATA_ROOT = os.environ.get("THERMAL_ROOT") or _find_thermal_root()   # searched recursively for frames
WORK = os.environ.get("WORK_DIR", "/kaggle/working")
OUT_DIR = os.path.join(WORK, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
print("split:", SPLIT, "| seed:", SEED, "| epochs:", EPOCHS, "| imgsz:", IMGSZ)
print("resolved CSV:", CSV_PATH)
print("resolved DATA_ROOT:", DATA_ROOT)

# %% ---- cell ----

if os.environ.get("SKIP_INSTALL") != "1":
    try:
        import ultralytics
        if ultralytics.__version__ != "8.4.14":
            raise ImportError("wrong version")
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.14"])
import ultralytics
import torch
from ultralytics import YOLO
assert ultralytics.__version__ == "8.4.14", f"need ultralytics 8.4.14, found {ultralytics.__version__}"
DEV_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
random.seed(0); np.random.seed(0); torch.manual_seed(0)
print("ultralytics", ultralytics.__version__, "| torch", torch.__version__, "| device", DEV_NAME)

# %% ---- cell ----

df = pd.read_csv(CSV_PATH)
needed = {"Serial Number", "Gas", "Corresponding Image Name"}
assert needed <= set(df.columns), f"CSV is missing {needed - set(df.columns)}"
df = df.sort_values("Serial Number").reset_index(drop=True)
df["label"] = df["Gas"]
print(df["label"].value_counts().to_dict())

# Index every image under DATA_ROOT by file stem. Stems must be unique to align rows to frames.
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
index, dups = {}, []
for root, _, files in os.walk(DATA_ROOT):
    for f in files:
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMG_EXT:
            if stem in index:
                dups.append(stem)
            index[stem] = os.path.join(root, f)
assert not dups, f"duplicate frame names, cannot align uniquely: {dups[:5]}"

df["image_path"] = df["Corresponding Image Name"].map(index.get)
missing = df[df["image_path"].isna()]
assert missing.empty, (f"{len(missing)} rows have no frame under {DATA_ROOT}; "
                       f"first: {missing['Corresponding Image Name'].head(3).tolist()}")
mismatched = [n for n, l in zip(df["Corresponding Image Name"], df["label"]) if not str(n).endswith("_" + l)]
assert not mismatched, f"{len(mismatched)} frame names do not end in their class, e.g. {mismatched[:3]}"
print("matched", len(df), "rows to frames")

# %% ---- cell ----

def block_wise_holdout(frame, test_frac=TEST_FRAC, gap=GAP, seed=SEED):
    rng = np.random.default_rng(seed)
    split = pd.Series("", index=frame.index, dtype=object)
    starts = {}
    for label in GAS_ORDER:
        block = frame.index[frame["label"] == label].tolist()   # Serial Number order
        n = len(block)
        n_test = max(1, int(round(n * test_frac)))
        n_gap = min(gap, max(0, (n - n_test) // 2))
        hi = max(n_gap, n - n_test - n_gap)
        lo = int(rng.integers(n_gap, hi + 1)) if (seed is not None and hi > n_gap) else hi
        starts[label] = lo
        for pos, idx in enumerate(block):
            if lo <= pos < lo + n_test:
                split[idx] = "test"
            elif lo - n_gap <= pos < lo or lo + n_test <= pos < lo + n_test + n_gap:
                split[idx] = "embargo"
            else:
                split[idx] = "train"
    return split, starts

def stratified_random_holdout(frame, test_frac=TEST_FRAC, seed=SEED):
    rng = np.random.default_rng(seed)
    split = pd.Series("", index=frame.index, dtype=object)
    for label in GAS_ORDER:
        block = frame.index[frame["label"] == label].to_numpy()
        n_test = int(round(len(block) * test_frac))
        held = rng.choice(block, size=n_test, replace=False)
        split.loc[block] = "train"
        split.loc[held] = "test"
    return split, {}

if SPLIT == "blockwise":
    df["split"], starts = block_wise_holdout(df)
else:
    df["split"], starts = stratified_random_holdout(df)
table = df.groupby(["label", "split"]).size().unstack(fill_value=0)
print(table)
print("held-out block starts (row index within each class):", starts)

# Checks: every class is fully assigned; for the block-wise split the test blocks are contiguous.
assert set(df["split"]) <= {"train", "test", "embargo"}
assert (table.sum(axis=1) == df["label"].value_counts()[table.index]).all()
assert (table["test"] == table["test"].iloc[0]).all(), "held-out count differs between classes"
if SPLIT == "blockwise":
    for label in GAS_ORDER:
        within = [pos for pos, s_ in enumerate(df.loc[df["label"] == label, "split"]) if s_ == "test"]
        assert within == list(range(within[0], within[0] + len(within))), f"test block for {label} is not contiguous"

df[["Serial Number", "Corresponding Image Name", "label", "split"]].to_csv(os.path.join(OUT_DIR, "split_table.csv"), index=False)

# %% ---- cell ----

DATA_DIR = os.path.join(WORK, f"thermal_{SPLIT}")
if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR)
for _, r in df[df["split"].isin(["train", "test"])].iterrows():
    sub = "train" if r["split"] == "train" else "val"
    dst_dir = os.path.join(DATA_DIR, sub, r["label"])
    os.makedirs(dst_dir, exist_ok=True)
    os.symlink(os.path.abspath(r["image_path"]), os.path.join(dst_dir, os.path.basename(r["image_path"])))
for sub in ("train", "val"):
    print(sub, {c: len(os.listdir(os.path.join(DATA_DIR, sub, c))) for c in sorted(os.listdir(os.path.join(DATA_DIR, sub)))})

# %% ---- cell ----

model = YOLO(os.environ.get("MODEL_INIT", "yolov8n-cls.pt"))   # env override only for the local smoke test

opt_info = {}
def _record_optimizer(trainer):
    # Read the optimizer as it is when training starts, so the record shows what was actually used.
    o = trainer.optimizer
    g0 = o.param_groups[0]
    opt_info.update({"class": type(o).__name__,
                     "lr_at_train_start": g0["lr"],
                     "betas": list(g0["betas"]) if "betas" in g0 else None,
                     "momentum": g0.get("momentum"),
                     "weight_decay_values": sorted({g.get("weight_decay") for g in o.param_groups})})
model.add_callback("on_train_start", _record_optimizer)

model.train(
    data=DATA_DIR,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    seed=0,                 # the original run used seed 0
    deterministic=True,
    optimizer="auto",       # resolves to AdamW, lr 0.00125, beta1 0.9 here
    warmup_epochs=3,
    warmup_bias_lr=0.0,     # what optimizer="auto" selects; set explicitly so the run does not depend on it
    cos_lr=False,           # linear learning-rate decay
    fliplr=0.5,
    degrees=0.0,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    auto_augment="randaugment",
    erasing=0.4,
    project=os.path.join(WORK, "runs"),
    name=RUN_NAME,
    exist_ok=True,
    plots=False,            # avoids a font download; results.csv is still written
    workers=2,
    amp=True,
    verbose=True,
)
trainer = model.trainer
print("optimizer used:", opt_info)

# %% ---- cell ----

save_dir = str(trainer.save_dir)
last = YOLO(os.path.join(save_dir, "weights", "last.pt"))
best = YOLO(os.path.join(save_dir, "weights", "best.pt"))
names = last.names                                  # {index: class}

def predict_all(m, rows):
    paths = rows["image_path"].tolist()
    preds = []
    for i in range(0, len(paths), 64):
        out = m.predict(paths[i:i + 64], imgsz=IMGSZ, device=DEVICE, verbose=False)
        preds += [int(o.probs.top1) for o in out]
    return [names[p] for p in preds]

held = df[df["split"] == "test"].copy()
pred_last = predict_all(last, held)
pred_best = predict_all(best, held)
true = held["label"].tolist()
acc_last = float(np.mean([p == t for p, t in zip(pred_last, true)]))
acc_best = float(np.mean([p == t for p, t in zip(pred_best, true)]))
per_class = {c: float(np.mean([p == t for p, t in zip(pred_last, true) if t == c])) for c in GAS_ORDER}
confusion = pd.crosstab(pd.Series(true, name="true"), pd.Series(pred_last, name="pred")).to_dict(orient="index")   # true class -> predicted counts
hazard_frames = sum(1 for t in true if t in HAZARD)
hazard_called_nonhazard = sum(1 for t, p in zip(true, pred_last) if t in HAZARD and p not in HAZARD)

result = {
    "run": RUN_NAME,
    "split": SPLIT,
    "seed": SEED,
    "ultralytics": ultralytics.__version__,
    "torch": torch.__version__,
    "device": DEV_NAME,
    "config": {"epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH, "test_frac": TEST_FRAC,
               "embargo_frames": GAP if SPLIT == "blockwise" else 0, "gas_order": GAS_ORDER, "hazard": HAZARD},
    "held_out_block_start_per_class": starts,
    "counts": {"train": int((df["split"] == "train").sum()),
               "embargo": int((df["split"] == "embargo").sum()),
               "heldout": int(len(held))},
    "optimizer": opt_info,
    "primary_top1_final_epoch": acc_last,
    "top1_best_checkpoint_reference_only": acc_best,
    "per_class_final_epoch": per_class,
    "confusion_final_epoch": confusion,
    "hazard_frames": int(hazard_frames),
    "hazard_called_nonhazard_final_epoch": int(hazard_called_nonhazard),
}
with open(os.path.join(OUT_DIR, f"{RUN_NAME}.json"), "w") as fh:
    json.dump(result, fh, indent=2, default=str)
shutil.copy(os.path.join(save_dir, "weights", "last.pt"), os.path.join(OUT_DIR, f"{RUN_NAME}_last.pt"))
if os.path.exists(os.path.join(save_dir, "results.csv")):
    shutil.copy(os.path.join(save_dir, "results.csv"), os.path.join(OUT_DIR, f"{RUN_NAME}_results.csv"))

# %% ---- cell ----

print(f"SPLIT {SPLIT}, SEED {SEED}")
print(f"HELD-OUT, FINAL EPOCH: top-1 = {acc_last:.5f} ({int(round(acc_last * len(held)))}/{len(held)})")
print(f"per class: { {k: round(v, 4) for k, v in per_class.items()} }")
print(f"hazard frames {hazard_frames}, called non-hazard {hazard_called_nonhazard}")
print(f"reference, best checkpoint: {acc_best:.5f}")
print("files in", OUT_DIR, ":", sorted(os.listdir(OUT_DIR)))