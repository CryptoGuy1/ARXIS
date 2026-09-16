"""Re-run of the full experiment programme with the safety-relevant metric set.

Produces, into retrain/results_v2/:
  v2_zoo.csv            model comparison, 10 models, escalation + FA + CP bounds
  v2_zoo_sig.json       paired Wilcoxon vs the cost-weighted policy
  v2_loco.csv           leave-one-class-out, all models, escalation column
  v2_costsweep.csv      cost ratio 1:1..20:1 with escalation
  v2_perturb.csv        graded perturbation suite WITH escalation + FA
  v2_ablation.csv       decision-state feature ablation
  v2_anomaly.json       out-of-sample anomaly evaluation + per-class scores
"""
import os, sys, json, time, inspect, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrain import raw_pipeline as rp
from retrain import rewards as RW
from retrain.comparators import SupervisedDQN, SupervisedMLP, CostSensitiveGBM
from retrain.comparators import cost_weighted_sample_weight
from retrain.zoo import SVMClassifier, RandomForest, RawWindowLSTM, CQLAgent
from retrain.new_baselines import KNN, ThresholdRule, CusumDetector
from retrain.safety_metrics import evaluate_safety, agg, AGG_KEYS, cp_upper
from retrain.metrics import paired_significance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "retrain", "results_v2")
CACHE = os.path.join(ROOT, "retrain", "results", "real_features.csv")
SEEDS = [42, 1337, 7, 2024, 99]
EPOCHS = 80


def log(*a):
    print(*a, flush=True)


def get_ds():
    am, asc = rp.build_anomaly_model()
    return rp.build_dataset(am, asc, cache_path=CACHE), am, asc


# --------------------------------------------------------------------------
# model registry.  fit_kw may depend on (seed, gtr, sw)
# --------------------------------------------------------------------------
def registry(sw):
    return {
        "A_cost_weighted": (lambda s: SupervisedDQN(seed=s),
                            lambda s, g: dict(sample_weight=cost_weighted_sample_weight(g, 8, 1), epochs=EPOCHS)),
        "B_unweighted":    (lambda s: SupervisedDQN(seed=s), lambda s, g: dict(sample_weight=None, epochs=EPOCHS)),
        "D_mlp":           (lambda s: SupervisedMLP(seed=s), lambda s, g: {}),
        "E_gbm":           (lambda s: CostSensitiveGBM(seed=s), lambda s, g: {}),
        "SVM":             (lambda s: SVMClassifier(seed=s), lambda s, g: {}),
        "RF":              (lambda s: RandomForest(seed=s), lambda s, g: {}),
        "LSTM":            (lambda s: RawWindowLSTM(seed=s, K=10), lambda s, g: dict(epochs=40)),
        "CQL":             (lambda s: CQLAgent(seed=s, alpha=1.0), lambda s, g: dict(epochs=EPOCHS, g_class=g)),
        "KNN":             (lambda s: KNN(seed=s), lambda s, g: {}),
        "ThresholdRule":   (lambda s: ThresholdRule(seed=s), lambda s, g: {}),
        "CUSUM":           (lambda s: CusumDetector(seed=s), lambda s, g: dict(g_class=g)),
    }


def fit_predict(m, Xtr, ytr, gtr, Xte, gte, kw):
    kw = dict(kw)
    if "groups" in inspect.signature(m.fit).parameters:
        kw.setdefault("groups", gtr)
    m.fit(Xtr, ytr, **kw)
    if "groups" in inspect.signature(m.predict).parameters:
        return m.predict(Xte, groups=gte), m
    return m.predict(Xte), m


# --------------------------------------------------------------------------
# E1  model comparison
# --------------------------------------------------------------------------
def exp_zoo(ds):
    log("\n=== E1  model comparison (10 models x 5 seeds) ===")
    rows, per_seed = [], {}
    for name, (mk, kwf) in registry(None).items():
        t0 = time.time()
        evs = []
        for s in SEEDS:
            _, _, Xtr, gtr, ytr, Xte, gte = rp.split_and_prepare(ds, seed=s)
            acts, _ = fit_predict(mk(s), Xtr, ytr, gtr, Xte, gte, kwf(s, gtr))
            evs.append(evaluate_safety(gte, acts))
        a = agg(evs, AGG_KEYS)
        a["model"] = name
        rows.append(a)
        per_seed[name] = {k: [e[k] for e in evs] for k in AGG_KEYS}
        log(f"  {name:16s} acc={a['decision_acc_mean']:.4f}±{a['decision_acc_std']:.4f} "
            f"miss={a['miss_rate_mean']:.4f} esc={a['escalation_mean']:.4f} "
            f"fa={a['fa_per_clean_mean']:.4f}  [{time.time()-t0:.0f}s]")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v2_zoo.csv"), index=False)

    ref = per_seed["A_cost_weighted"]
    sig = {}
    for name, v in per_seed.items():
        if name == "A_cost_weighted":
            continue
        sig[name] = {k: paired_significance(ref[k], v[k], alternative="two-sided")
                     for k in ("decision_acc", "miss_rate", "escalation")}
    json.dump({"per_seed": per_seed, "wilcoxon_vs_A": sig,
               "min_attainable_two_sided_p": 2 ** (1 - len(SEEDS))},
              open(os.path.join(OUT, "v2_zoo_sig.json"), "w"), indent=2)
    return df


# --------------------------------------------------------------------------
# E2  leave-one-class-out, with escalation
# --------------------------------------------------------------------------
LOCO_MODELS = ["A_cost_weighted", "B_unweighted", "D_mlp", "E_gbm", "SVM", "RF", "KNN", "ThresholdRule"]


def exp_loco(ds):
    log("\n=== E2  leave-one-class-out (escalation reported) ===")
    reg = registry(None)
    rows = []
    for held in ["Smoke", "Mixture"]:
        hid = rp.GAS_MAP[held]
        for name in LOCO_MODELS:
            mk, kwf = reg[name]
            evs = []
            for s in SEEDS:
                _, _, Xtr, gtr, ytr, Xte, gte = rp.split_and_prepare(ds, seed=s)
                # train on the three other classes; evaluate on ALL windows of
                # the held-out class (train + test), as in the original study
                keep = gtr != hid
                Xtr2, gtr2, ytr2 = Xtr[keep], gtr[keep], ytr[keep]
                allmask = ds["gas_id"].values == hid
                # rebuild the held-out class rows under the same scaling
                tr_df, te_df = rp.block_wise_holdout(ds, seed=s)
                tr_df, te_df, fs, (p1, p99) = rp.prepare(tr_df, te_df)
                full = ds[allmask].copy()
                full[rp.ALL_FEAT_COLS] = fs.transform(full[rp.ALL_FEAT_COLS])
                full["anomaly"] = ((full["anomaly"] - p1) / (p99 - p1 + 1e-8)).clip(0, 1)
                Xh, gh = rp.to_arrays(full)
                acts, _ = fit_predict(mk(s), Xtr2, ytr2, gtr2, Xh, gh, kwf(s, gtr2))
                evs.append(evaluate_safety(gh, acts))
            a = agg(evs, AGG_KEYS)
            a["held_out"], a["model"] = held, name
            rows.append(a)
            log(f"  {held:8s} {name:16s} miss={a['miss_rate_mean']:.4f} "
                f"(<={100*a['miss_upper95_pooled']:.3f}% pooled)  esc={a['escalation_mean']:.4f}±{a['escalation_std']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v2_loco.csv"), index=False)
    return df


# --------------------------------------------------------------------------
# E3  cost-asymmetry sweep, with escalation
# --------------------------------------------------------------------------
RATIOS = [1, 2, 4, 6, 8, 10, 12, 16, 20]


def exp_costsweep(ds):
    log("\n=== E3  cost-asymmetry sweep ===")
    rows = []
    for C in RATIOS:
        evs = []
        for s in SEEDS:
            _, _, Xtr, gtr, ytr, Xte, gte = rp.split_and_prepare(ds, seed=s)
            sw = None if C == 1 else cost_weighted_sample_weight(gtr, C, 1)
            m = SupervisedDQN(seed=s)
            m.fit(Xtr, ytr, sample_weight=sw, epochs=EPOCHS)
            evs.append(evaluate_safety(gte, m.predict(Xte)))
        a = agg(evs, AGG_KEYS)
        a["cost_ratio"] = f"{C}:1"
        rows.append(a)
        log(f"  C={C:2d}:1  acc={a['decision_acc_mean']:.4f}±{a['decision_acc_std']:.4f} "
            f"miss={a['miss_rate_mean']:.4f} esc={a['escalation_mean']:.4f}")
    # label-function reference (identity, reported as a ceiling not a baseline)
    evs = []
    for s in SEEDS:
        _, _, _, _, _, Xte, gte = rp.split_and_prepare(ds, seed=s)
        evs.append(evaluate_safety(gte, [RW.rule_oracle_action(int(g)) for g in gte]))
    a = agg(evs, AGG_KEYS); a["cost_ratio"] = "label_function"
    rows.append(a)
    log(f"  label fn  acc={a['decision_acc_mean']:.4f}  esc={a['escalation_mean']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v2_costsweep.csv"), index=False)
    return df


# --------------------------------------------------------------------------
# E4  graded perturbation suite WITH escalation + false alarm
# --------------------------------------------------------------------------
PERTURB_MODELS = ["A_cost_weighted", "B_unweighted", "D_mlp", "E_gbm", "RF", "ThresholdRule"]


def perturb(X, kind, level, rng):
    X = np.array(X, dtype=float, copy=True)
    cur = slice(1, 8)
    if kind == "noise":
        X[:, 1:] += rng.normal(0.0, level, size=X[:, 1:].shape)
    elif kind == "drift":
        # multiplicative gain + additive baseline shift on the sensor channels
        gain = 1.0 + rng.choice([-1.0, 1.0], size=7) * level
        X[:, cur] = X[:, cur] * gain + rng.choice([-1.0, 1.0], size=7) * level
    elif kind == "dropout":
        k = int(level)
        idx = rng.choice(7, size=k, replace=False)
        X[:, 1 + idx] = 0.0
        X[:, 8 + idx] = 0.0
        X[:, 15 + idx] = 0.0
    return X


def exp_perturb(ds):
    log("\n=== E4  graded perturbation suite (escalation + FA reported) ===")
    reg = registry(None)
    conds = ([("noise", l) for l in (0.1, 0.2, 0.3, 0.4, 0.5)]
             + [("drift", l) for l in (0.1, 0.2, 0.3, 0.4, 0.5)]
             + [("dropout", k) for k in (1, 2, 3, 4, 5, 6, 7)])
    rows = []
    for name in PERTURB_MODELS:
        mk, kwf = reg[name]
        t0 = time.time()
        # train once per seed, then evaluate on every perturbed test set
        fitted = []
        for s in SEEDS:
            _, _, Xtr, gtr, ytr, Xte, gte = rp.split_and_prepare(ds, seed=s)
            m = mk(s)
            kw = dict(kwf(s, gtr))
            if "groups" in inspect.signature(m.fit).parameters:
                kw.setdefault("groups", gtr)
            m.fit(Xtr, ytr, **kw)
            fitted.append((s, m, Xte, gte))
        for kind, lvl in [("clean", 0.0)] + conds:
            evs = []
            for s, m, Xte, gte in fitted:
                rng = np.random.default_rng(s)
                Xp = Xte if kind == "clean" else perturb(Xte, kind, lvl, rng)
                if "groups" in inspect.signature(m.predict).parameters:
                    acts = m.predict(Xp, groups=gte)
                else:
                    acts = m.predict(Xp)
                evs.append(evaluate_safety(gte, acts))
            a = agg(evs, AGG_KEYS)
            a["model"], a["perturbation"], a["level"] = name, kind, lvl
            rows.append(a)
        log(f"  {name:16s} done [{time.time()-t0:.0f}s]")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v2_perturb.csv"), index=False)
    return df


# --------------------------------------------------------------------------
# E5  decision-state ablation  (the claim with no experiment behind it)
# --------------------------------------------------------------------------
ABLATIONS = {
    "full_22":            list(range(22)),
    "no_anomaly_21":      list(range(1, 22)),
    "no_delta_std_8":     [0] + list(range(1, 8)),
    "no_std_15":          list(range(0, 15)),
    "no_delta_15":        [0] + list(range(1, 8)) + list(range(15, 22)),
    "anomaly_only_1":     [0],
    "current_only_7":     list(range(1, 8)),
}


def exp_ablation(ds):
    log("\n=== E5  decision-state ablation ===")
    rows = []
    for name, cols in ABLATIONS.items():
        evs = []
        for s in SEEDS:
            _, _, Xtr, gtr, ytr, Xte, gte = rp.split_and_prepare(ds, seed=s)
            m = SupervisedMLP(seed=s)          # architecture-agnostic probe
            m.fit(Xtr[:, cols], ytr)
            evs.append(evaluate_safety(gte, m.predict(Xte[:, cols])))
        a = agg(evs, AGG_KEYS)
        a["state"], a["n_features"] = name, len(cols)
        rows.append(a)
        log(f"  {name:18s} d={len(cols):2d} acc={a['decision_acc_mean']:.4f}±{a['decision_acc_std']:.4f} "
            f"miss={a['miss_rate_mean']:.4f} esc={a['escalation_mean']:.4f}")
    df = pd.DataFrame(rows)
    base = df.loc[df.state == "full_22", "decision_acc_mean"].iloc[0]
    df["delta_pp_vs_full"] = (df["decision_acc_mean"] - base) * 100.0
    df.to_csv(os.path.join(OUT, "v2_ablation.csv"), index=False)
    return df


# --------------------------------------------------------------------------
# E6  anomaly detector, evaluated OUT OF SAMPLE
# --------------------------------------------------------------------------
def exp_anomaly(ds, am, asc):
    log("\n=== E6  anomaly detector, out-of-sample ===")
    from sklearn.metrics import roc_auc_score
    out = {"per_class_mean_recon": rp.anomaly_by_class(am, asc)}
    per_seed = []
    for s in SEEDS:
        tr, te = rp.block_wise_holdout(ds, seed=s)
        tr, te = rp.partition_anomaly(tr, te, seed=s)   # fix 4.2
        tau = float(np.percentile(tr.loc[tr.label == "NoGas", "anomaly"], 95))
        te_norm = te.loc[te.label == "NoGas", "anomaly"].values
        te_anom = te.loc[te.label != "NoGas", "anomaly"].values
        y = np.r_[np.zeros(len(te_norm)), np.ones(len(te_anom))]
        sc = np.r_[te_norm, te_anom]
        fp = int((te_norm > tau).sum()); tp = int((te_anom > tau).sum())
        per_seed.append(dict(seed=s, tau=tau,
                             auc=float(roc_auc_score(y, sc)),
                             n_norm=len(te_norm), n_anom=len(te_anom),
                             fpr=fp / len(te_norm), tpr=tp / len(te_anom)))
        log(f"  seed {s:5d} tau={tau:.5f} AUC={per_seed[-1]['auc']:.4f} "
            f"TPR={per_seed[-1]['tpr']:.4f} FPR={per_seed[-1]['fpr']:.4f}")
    out["held_out_per_seed"] = per_seed
    for k in ("auc", "tpr", "fpr"):
        v = np.array([p[k] for p in per_seed])
        out[f"{k}_mean"], out[f"{k}_std"] = float(v.mean()), float(v.std(ddof=1))
    # in-sample reproduction of the published operating point, for comparison
    tau_all = float(np.percentile(ds.loc[ds.label == "NoGas", "anomaly"], 95))
    nrm = ds.loc[ds.label == "NoGas", "anomaly"].values
    anm = ds.loc[ds.label != "NoGas", "anomaly"].values
    out["in_sample"] = dict(tau=tau_all,
                            auc=float(roc_auc_score(np.r_[np.zeros(len(nrm)), np.ones(len(anm))],
                                                    np.r_[nrm, anm])),
                            fpr=float((nrm > tau_all).mean()),
                            tpr=float((anm > tau_all).mean()),
                            n_norm=len(nrm), n_anom=len(anm))
    log(f"  IN-SAMPLE  tau={tau_all:.5f} AUC={out['in_sample']['auc']:.4f} "
        f"TPR={out['in_sample']['tpr']:.4f} FPR={out['in_sample']['fpr']:.4f}")
    json.dump(out, open(os.path.join(OUT, "v2_anomaly.json"), "w"), indent=2)
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ds, am, asc = get_ds()
    log(f"dataset: {len(ds)} windows, {ds['label'].value_counts().to_dict()}")
    which = sys.argv[1:] or ["anomaly", "ablation", "zoo", "costsweep", "loco", "perturb"]
    if "anomaly" in which:   exp_anomaly(ds, am, asc)
    if "ablation" in which:  exp_ablation(ds)
    if "zoo" in which:       exp_zoo(ds)
    if "costsweep" in which: exp_costsweep(ds)
    if "loco" in which:      exp_loco(ds)
    if "perturb" in which:   exp_perturb(ds)
    log("\nAll done ->", OUT)
