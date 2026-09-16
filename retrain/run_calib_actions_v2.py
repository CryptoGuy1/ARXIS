"""Two experiments the earlier revision left open.

E7  Calibration on the corrected pipeline, with a proper calibration split for
    temperature scaling, equal-mass (adaptive) binning, a bin-count sweep,
    bootstrap intervals, class-conditional ECE on the hazardous classes, and
    multiclass Brier score.  The earlier ECE numbers came from a different data
    path and from a temperature fitted on training logits, so they are dropped
    rather than carried forward.

E8  Action-selection matrices in distribution: for each model, the share of each
    gas class assigned to each of the five actions.  This is what the earlier
    draft's Fig. 9b showed, rebuilt on the corrected pipeline for several models
    rather than one.
"""
import os, sys, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrain import raw_pipeline as rp
from retrain import rewards as RW
from retrain.comparators import SupervisedDQN, SupervisedMLP, CostSensitiveGBM
from retrain.comparators import cost_weighted_sample_weight
from retrain.zoo import SVMClassifier, RandomForest
from retrain.new_baselines import KNN, ThresholdRule
from retrain.safety_metrics import evaluate_safety

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "retrain", "results_v2")
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "retrain", "results", "real_features.csv")
SEEDS = [42, 1337, 7, 2024, 99]
EPOCHS = 80
DANGER = {1, 2}


# ---------------------------------------------------------------- calibration
def ece_equal_mass(conf, correct, n_bins=15):
    """Adaptive (equal-mass) binned ECE. Equal-width bins leave most bins empty
    when confidence piles up near 1, which is exactly this corpus."""
    conf = np.asarray(conf, float); correct = np.asarray(correct, bool)
    n = len(conf)
    if n == 0:
        return float("nan")
    order = np.argsort(conf)
    bins = np.array_split(order, n_bins)
    e = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        e += abs(conf[b].mean() - correct[b].mean()) * len(b)
    return float(e / n)


def brier_multiclass(probs, y_true_action, n_actions=5):
    p = np.asarray(probs, float)
    oh = np.zeros_like(p)
    oh[np.arange(len(p)), np.asarray(y_true_action, int)] = 1.0
    return float(np.mean(np.sum((p - oh) ** 2, axis=1)))


BOOT_BLOCK = 20          # = WINDOW_SIZE: the shortest block that keeps a window intact
BOOT_REPLICATES = 400


def _block_index(n, L, rng):
    """Moving-block index set for one bootstrap replicate of a length-n series."""
    if L >= n:
        return np.arange(n)
    k = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=k)
    return np.concatenate([np.arange(t, t + L) for t in starts])[:n]


def boot_ci(conf, correct, n_bins=15, n_boot=BOOT_REPLICATES, seed=0, block=BOOT_BLOCK):
    """Percentile interval for ECE under a MOVING-BLOCK bootstrap.

    Resampling individual windows independently would assume exactly the
    independence this corpus does not have: consecutive windows share 19 of 20
    raw rows, so an i.i.d. bootstrap understates the interval. Blocks of length
    `block` are drawn with replacement instead, which preserves dependence up to
    that lag. The windows are supplied in acquisition order by the caller.
    """
    rng = np.random.default_rng(seed)
    n = len(conf)
    vals = [ece_equal_mass(conf[i], correct[i], n_bins)
            for i in (_block_index(n, block, rng) for _ in range(n_boot))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits, y, grid=np.linspace(0.25, 6.0, 200)):
    """1-D search for T minimizing NLL on a HELD-OUT calibration split."""
    best, bestT = np.inf, 1.0
    for T in grid:
        p = softmax(logits / T)
        nll = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)))
        if nll < best:
            best, bestT = nll, T
    return float(bestT)


def dqn_logits(model, X):
    model.net.eval()
    with torch.no_grad():
        return model.net(torch.tensor(np.asarray(X, np.float32))).cpu().numpy()


def mc_dropout_probs(model, X, passes=20, seed=0):
    torch.manual_seed(seed)
    model.net.train()                      # keep dropout active
    acc = None
    with torch.no_grad():
        xt = torch.tensor(np.asarray(X, np.float32))
        for _ in range(passes):
            p = torch.softmax(model.net(xt), dim=1).cpu().numpy()   # average PROBS
            acc = p if acc is None else acc + p
    model.net.eval()
    return acc / passes


def exp_calibration(ds):
    print("\n=== E7  calibration, corrected pipeline ===", flush=True)
    rows, sweep_rows = [], []
    for s in SEEDS:
        tr_df, te_df = rp.block_wise_holdout(ds, seed=s)
        tr_df, te_df, _, _ = rp.prepare(tr_df, te_df, anomaly_seed=s)
        tr_df = tr_df.sort_values("win_id").reset_index(drop=True)
        te_df = te_df.sort_values("win_id").reset_index(drop=True)   # acquisition order
        Xtr_all, gtr_all = rp.to_arrays(tr_df)
        Xte, gte = rp.to_arrays(te_df)
        ytr_all = np.array([RW.rule_oracle_action(int(g)) for g in gtr_all])
        yte = np.array([RW.rule_oracle_action(int(g)) for g in gte])

        # Carve the calibration split out of TRAIN as a CONTIGUOUS band per class,
        # with the same embargo used for the outer holdout. A uniformly random
        # calibration subset over overlapping windows is the leakage this paper is
        # about: the earlier version used rng.permutation here, so the temperature
        # was fitted on windows sharing 19 of 20 raw rows with the fitting set.
        cal, fit = [], []
        for lab in rp.GAS_ORDER:
            pos = np.flatnonzero((tr_df["label"] == lab).values)
            if len(pos) == 0:
                continue
            ncal_c = max(1, int(round(0.2 * len(pos))))
            gap = min(rp.WINDOW_SIZE, max(0, len(pos) - ncal_c))
            cal.extend(pos[-ncal_c:])                       # tail band of the class
            fit.extend(pos[:max(0, len(pos) - ncal_c - gap)])  # embargo between them
        cal, fit = np.array(cal, int), np.array(fit, int)
        Xfit, yfit, gfit = Xtr_all[fit], ytr_all[fit], gtr_all[fit]
        Xcal, ycal = Xtr_all[cal], ytr_all[cal]

        sw = cost_weighted_sample_weight(gfit, 8, 1)
        base = SupervisedDQN(seed=s); base.fit(Xfit, yfit, sample_weight=sw, epochs=EPOCHS)
        lo_te = dqn_logits(base, Xte); lo_cal = dqn_logits(base, Xcal)

        ens_p = None                       # same class + objective as the single model
        for j in range(5):
            m = SupervisedDQN(seed=s * 100 + j)
            m.fit(Xfit, yfit, sample_weight=sw, epochs=EPOCHS)
            p = softmax(dqn_logits(m, Xte))
            ens_p = p if ens_p is None else ens_p + p
        ens_p /= 5.0

        T = fit_temperature(lo_cal, ycal)
        est = {"raw_softmax": softmax(lo_te),
               "mc_dropout_20": mc_dropout_probs(base, Xte, 20, seed=s),
               "temp_scaling": softmax(lo_te / T),
               "deep_ensemble_5": ens_p}

        for name, p in est.items():
            pred = p.argmax(1); conf = p.max(1)
            corr = (pred == yte)
            dmask = np.isin(gte, list(DANGER))
            lo, hi = boot_ci(conf, corr, 15, BOOT_REPLICATES, seed=s)
            rows.append(dict(seed=s, estimator=name, T=(T if name == "temp_scaling" else np.nan),
                             ece=ece_equal_mass(conf, corr, 15), ece_lo=lo, ece_hi=hi,
                             ece_danger=ece_equal_mass(conf[dmask], corr[dmask], 10),
                             brier=brier_multiclass(p, yte),
                             nll=float(-np.mean(np.log(np.clip(p[np.arange(len(yte)), yte], 1e-12, None)))),
                             acc=float(corr.mean())))
            for nb in (5, 10, 15, 20, 30):
                sweep_rows.append(dict(seed=s, estimator=name, n_bins=nb,
                                       ece=ece_equal_mass(conf, corr, nb)))
        print(f"  seed {s} done (T={T:.2f})", flush=True)

    df = pd.DataFrame(rows)
    agg = df.groupby("estimator").agg(
        ece_mean=("ece", "mean"), ece_std=("ece", "std"),
        ece_lo=("ece_lo", "mean"), ece_hi=("ece_hi", "mean"),
        ece_danger_mean=("ece_danger", "mean"), ece_danger_std=("ece_danger", "std"),
        brier_mean=("brier", "mean"), brier_std=("brier", "std"),
        nll_mean=("nll", "mean"), acc_mean=("acc", "mean")).reset_index()
    df.to_csv(os.path.join(OUT, "v2_calibration_perseed.csv"), index=False)
    agg.to_csv(os.path.join(OUT, "v2_calibration.csv"), index=False)
    pd.DataFrame(sweep_rows).to_csv(os.path.join(OUT, "v2_calibration_binsweep.csv"), index=False)
    print(agg.round(4).to_string(index=False), flush=True)
    return agg


# ------------------------------------------------------- action distributions
ACTION_MODELS = ["A_cost_weighted", "D_mlp", "E_gbm", "RF", "SVM", "KNN", "ThresholdRule"]


def exp_action_matrix(ds):
    print("\n=== E8  action-selection matrices, in distribution ===", flush=True)
    makers = {
        "A_cost_weighted": lambda s, g: (SupervisedDQN(seed=s),
                                         dict(sample_weight=cost_weighted_sample_weight(g, 8, 1), epochs=EPOCHS)),
        "D_mlp": lambda s, g: (SupervisedMLP(seed=s), {}),
        "E_gbm": lambda s, g: (CostSensitiveGBM(seed=s), {}),
        "RF": lambda s, g: (RandomForest(seed=s), {}),
        "SVM": lambda s, g: (SVMClassifier(seed=s), {}),
        "KNN": lambda s, g: (KNN(seed=s), {}),
        "ThresholdRule": lambda s, g: (ThresholdRule(seed=s), {}),
    }
    rows = []
    for name in ACTION_MODELS:
        mats = []
        for s in SEEDS:
            _, _, Xtr, gtr, ytr, Xte, gte = rp.split_and_prepare(ds, seed=s)
            m, kw = makers[name](s, gtr)
            m.fit(Xtr, ytr, **kw)
            acts = np.asarray(m.predict(Xte), int)
            M = np.zeros((4, 5))
            for g in range(4):
                sel = gte == g
                if sel.sum():
                    for a in range(5):
                        M[g, a] = float((acts[sel] == a).mean())
            mats.append(M)
        M = np.mean(mats, axis=0)
        for g, gname in enumerate(rp.GAS_ORDER):
            for a in range(5):
                rows.append(dict(model=name, gas=gname, action=a, share=float(M[g, a])))
        print(f"  {name:16s} done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "v2_action_matrix.csv"), index=False)


# --------------------------------------------- anomaly ROC curves for plotting
def exp_roc(ds):
    print("\n=== ROC curves for the anomaly detector ===", flush=True)
    from sklearn.metrics import roc_curve
    out = {"held_out": [], "in_sample": {}}
    for s in SEEDS:
        tr, te = rp.block_wise_holdout(ds, seed=s)
        tr, te = rp.partition_anomaly(tr, te, seed=s)   # fix 4.2
        nor = te.loc[te.label == "NoGas", "anomaly"].values
        ano = te.loc[te.label != "NoGas", "anomaly"].values
        y = np.r_[np.zeros(len(nor)), np.ones(len(ano))]
        fpr, tpr, _ = roc_curve(y, np.r_[nor, ano])
        k = max(1, len(fpr) // 400)
        out["held_out"].append(dict(seed=s, fpr=fpr[::k].tolist(), tpr=tpr[::k].tolist()))
    nor = ds.loc[ds.label == "NoGas", "anomaly"].values
    ano = ds.loc[ds.label != "NoGas", "anomaly"].values
    y = np.r_[np.zeros(len(nor)), np.ones(len(ano))]
    fpr, tpr, _ = roc_curve(y, np.r_[nor, ano])
    k = max(1, len(fpr) // 400)
    out["in_sample"] = dict(fpr=fpr[::k].tolist(), tpr=tpr[::k].tolist())
    out["per_class_scores"] = {c: ds.loc[ds.label == c, "anomaly"].values.tolist()
                               for c in rp.GAS_ORDER}
    json.dump(out, open(os.path.join(OUT, "v2_roc.json"), "w"))
    print("  wrote v2_roc.json", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    which = sys.argv[1:] or ["roc", "actions", "calibration"]
    if "roc" in which:         exp_roc(ds)
    if "actions" in which:     exp_action_matrix(ds)
    if "calibration" in which: exp_calibration(ds)
    print("\ndone ->", OUT)
