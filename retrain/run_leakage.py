"""E15. How much of a reported accuracy on this corpus is window overlap?

The block-wise holdout is defended in Section 3.1 on the grounds that a random
split over 20-step sliding windows places near-duplicates in both partitions.
That argument has been made in the literature (Dennler et al. 2022) but not, as
far as we know, measured on this corpus. Here it is measured directly: the same
models, the same data, four partitioning protocols.

  random          uniformly random 80/20 over all windows
  blocked_noembargo   contiguous 20% per class, no separation band
  blocked_embargo     contiguous 20% per class, 20-window embargo (this paper)
  leave_one_block     train on the first 60% of each class block, test on the
                      last 20%, with the middle 20% discarded entirely

Also runs a blocked validation split, because Section 4.4's cost-ratio
selection needs a validation partition that is not itself leaky.
"""
import os, sys, json, warnings, inspect
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrain import raw_pipeline as rp
from retrain import rewards as RW
from retrain.comparators import SupervisedDQN, SupervisedMLP, CostSensitiveGBM
from retrain.comparators import cost_weighted_sample_weight
from retrain.zoo import RandomForest
from retrain.new_baselines import KNN, ThresholdRule
from retrain.safety_metrics import evaluate_safety, agg, AGG_KEYS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "retrain", "results_v2")
CACHE = os.path.join(ROOT, "retrain", "results", "real_features.csv")
SEEDS = [42, 1337, 7, 2024, 99]
EPOCHS = 80


def log(*a):
    print(*a, flush=True)


def split_indices(df, protocol, seed):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for label in rp.GAS_ORDER:
        blk = df.index[df["label"] == label].tolist()
        n = len(blk); n_te = int(round(0.2 * n))
        if protocol == "random":
            idx = rng.permutation(blk)
            te += list(idx[:n_te]); tr += list(idx[n_te:])
        elif protocol == "blocked_noembargo":
            hi = max(0, n - n_te); start = int(rng.integers(0, hi + 1))
            te += blk[start:start + n_te]
            tr += blk[:start] + blk[start + n_te:]
        elif protocol == "blocked_embargo":
            gap = rp.WINDOW_SIZE
            n_gap = min(gap, max(0, n - n_te))
            hi = max(0, n - n_te - n_gap); start = int(rng.integers(0, hi + 1))
            tr += blk[:start] + blk[start + n_te + n_gap:]
            te += blk[start + n_gap:start + n_gap + n_te]
        elif protocol == "leave_one_block":
            tr += blk[:int(0.6 * n)]
            te += blk[int(0.8 * n):]
        else:
            raise KeyError(protocol)
    return tr, te


def prep(df, tr_idx, te_idx):
    tr, te = df.loc[tr_idx].reset_index(drop=True), df.loc[te_idx].reset_index(drop=True)
    tr, te, _, _ = rp.prepare(tr, te)
    Xtr, gtr = rp.to_arrays(tr); Xte, gte = rp.to_arrays(te)
    ytr = np.array([RW.rule_oracle_action(int(g)) for g in gtr])
    return Xtr, gtr, ytr, Xte, gte


MODELS = {
    "A_cost_weighted": lambda s, g: (SupervisedDQN(seed=s),
                                     dict(sample_weight=cost_weighted_sample_weight(g, 8, 1), epochs=EPOCHS)),
    "D_mlp":  lambda s, g: (SupervisedMLP(seed=s), {}),
    "E_gbm":  lambda s, g: (CostSensitiveGBM(seed=s), {}),
    "RF":     lambda s, g: (RandomForest(seed=s), {}),
    "KNN":    lambda s, g: (KNN(seed=s), {}),
    "ThresholdRule": lambda s, g: (ThresholdRule(seed=s), {}),
}
PROTOCOLS = ["random", "blocked_noembargo", "blocked_embargo", "leave_one_block"]


def exp_leakage(ds):
    log("\n=== E15  leakage sensitivity: same models, four partitioning protocols ===")
    rows = []
    for proto in PROTOCOLS:
        for name, mk in MODELS.items():
            evs = []
            for s in SEEDS:
                tr_idx, te_idx = split_indices(ds, proto, s)
                Xtr, gtr, ytr, Xte, gte = prep(ds, tr_idx, te_idx)
                m, kw = mk(s, gtr)
                m.fit(Xtr, ytr, **kw)
                evs.append(evaluate_safety(gte, m.predict(Xte)))
            a = agg(evs, AGG_KEYS); a["protocol"], a["model"] = proto, name
            rows.append(a)
        sub = [r for r in rows if r["protocol"] == proto]
        log(f"  {proto:20s} mean acc across models = "
            f"{np.mean([r['decision_acc_mean'] for r in sub]):.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v3_leakage.csv"), index=False)
    piv = df.pivot_table(index="model", columns="protocol", values="decision_acc_mean")
    log("\n" + piv.round(4).to_string())
    gap = piv["random"] - piv["blocked_embargo"]
    log("\n  random minus blocked+embargo, per model:")
    for m, v in gap.items():
        log(f"    {m:16s} {100*v:+.2f} points")
    return df


def exp_cost_blocked_validation(ds):
    """Cost-ratio selection on a BLOCKED validation split carved from training."""
    log("\n=== E13b  cost ratio on a blocked validation split ===")
    rows = []
    for C in [1, 2, 4, 6, 8, 10, 12, 16, 20]:
        va, te = [], []
        for s in SEEDS:
            tr_df, te_df = rp.block_wise_holdout(ds, seed=s)
            # carve a contiguous validation band from the END of each class's
            # training rows, so validation windows do not overlap training ones
            fit_idx, val_idx = [], []
            for label in rp.GAS_ORDER:
                blk = tr_df.index[tr_df["label"] == label].tolist()
                nv = int(round(0.2 * len(blk)))
                gap = rp.WINDOW_SIZE
                fit_idx += blk[:len(blk) - nv - gap]
                val_idx += blk[len(blk) - nv:]
            fit_df = tr_df.loc[fit_idx].reset_index(drop=True)
            val_df = tr_df.loc[val_idx].reset_index(drop=True)
            f2, v2, sc, (p1, p99) = rp.prepare(fit_df, val_df)
            Xf, gf = rp.to_arrays(f2); Xv, gv = rp.to_arrays(v2)
            yf = np.array([RW.rule_oracle_action(int(g)) for g in gf])
            m = SupervisedDQN(seed=s)
            sw = None if C == 1 else cost_weighted_sample_weight(gf, C, 1)
            m.fit(Xf, yf, sample_weight=sw, epochs=EPOCHS)
            va.append(evaluate_safety(gv, m.predict(Xv)))
            te_scaled = te_df.copy()
            te_scaled[rp.ALL_FEAT_COLS] = sc.transform(te_scaled[rp.ALL_FEAT_COLS])
            te_scaled["anomaly"] = ((te_scaled["anomaly"] - p1) / (p99 - p1 + 1e-8)).clip(0, 1)
            Xt, gt = rp.to_arrays(te_scaled)
            te.append(evaluate_safety(gt, m.predict(Xt)))
        av, at = agg(va, AGG_KEYS), agg(te, AGG_KEYS)
        rows.append(dict(cost_ratio=f"{C}:1",
                         val_acc_mean=av["decision_acc_mean"], val_acc_std=av["decision_acc_std"],
                         test_acc_mean=at["decision_acc_mean"], test_acc_std=at["decision_acc_std"],
                         val_esc=av["escalation_mean"], test_esc=at["escalation_mean"],
                         val_miss=av["miss_rate_mean"], test_miss=at["miss_rate_mean"]))
        log(f"  C={C:2d}:1 blocked-val={av['decision_acc_mean']:.4f} test={at['decision_acc_mean']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v3_cost_blocked_validation.csv"), index=False)
    best = df.loc[df.val_acc_mean.idxmax()]
    log(f"  selected on blocked validation: {best.cost_ratio} (test {best.test_acc_mean:.4f})")
    return df


if __name__ == "__main__":
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    which = sys.argv[1:] or ["leakage", "costblocked"]
    if "leakage" in which:     exp_leakage(ds)
    if "costblocked" in which: exp_cost_blocked_validation(ds)
    log("\ndone")
