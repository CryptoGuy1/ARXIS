"""Is the random-versus-blocked gap overlap contamination, or temporal position?

The manuscript reports that a random split over overlapping sliding windows adds
3.23 to 5.57 accuracy points for every model that reads temporal features. Every
review made the same objection to reading that causally: a random split changes
two things at once. Near-duplicate windows can land on both sides of it, which is
contamination, and it also draws training data from the whole recording rather
than from a contiguous early band, which is coverage. The reported gap is their
sum, and the manuscript had to narrow its claim to "the partitioning protocol
matters" because nothing here separated them.

This driver separates them, with a two-by-two that holds everything else fixed.

                      overlap possible?   temporal coverage
  a  stride-1, random        yes              whole block
  b  stride-20, random       no               whole block
  c  stride-1, blocked       no               contiguous band
  d  stride-20, blocked      no               contiguous band

A stride-20 window set shares no raw reading between any two windows, so
contamination is impossible in b and d by construction. Coverage is identical
between a and b, and between c and d.

  a - b   the effect of overlap contamination, coverage held constant
  b - d   the effect of temporal position, overlap absent from both

Training-set size is the obvious confound, since a stride-20 set is a twentieth
the size. Every arm is therefore subsampled to the SAME number of training
windows, the budget the stride-20 arm can afford, so the four numbers differ in
the two factors under study and in nothing else. Accuracy is lower in all four
arms than in the manuscript's main tables for that reason; the comparison between
arms is the result, not the level.

Writes results_v2/v3_overlap_isolation.csv.

    python3 -m retrain.run_overlap_isolation
"""
import os, sys, time, inspect, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrain import raw_pipeline as rp
from retrain import rewards as RW
from retrain.run_all_v2 import registry, SEEDS, OUT, CACHE

# The claim under test is about models that read temporal features, so those are
# the models tested. The shallow tree reads only the seven current channels and
# is carried as a control the protocol should barely touch.
MODELS = ["A_cost_weighted", "B_unweighted", "D_mlp", "E_gbm", "RF", "ThresholdRule"]
STRIDE = rp.WINDOW_SIZE          # 20: no two windows share a raw reading
TEST_FRAC = 0.2


def thin(df, stride):
    """Every `stride`-th window inside each class block, so no two share a row."""
    out = []
    for cls, g in df.groupby("gas_id"):
        g = g.sort_values("win_id")
        out.append(g.iloc[::stride])
    return pd.concat(out).sort_values("win_id").reset_index(drop=True)


def random_split(df, seed, test_frac=TEST_FRAC):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_te = int(round(test_frac * len(df)))
    return df.iloc[idx[n_te:]].copy(), df.iloc[idx[:n_te]].copy()


def blocked_split(df, seed, test_frac=TEST_FRAC):
    """Contiguous per-class band with a two-sided embargo, in window units of the
    set being split: one window of the thinned set spans 20 raw rows, so a
    one-window band there is already a full embargo."""
    gap = 1 if len(df) < 1000 else rp.WINDOW_SIZE
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for cls, g in df.groupby("gas_id"):
        g = g.sort_values("win_id")
        n = len(g)
        n_te = int(round(test_frac * n))
        n_gap = min(gap, max(0, (n - n_te) // 2))
        hi = max(n_gap, n - n_te - n_gap)
        lo = int(rng.integers(n_gap, hi + 1)) if hi > n_gap else hi
        tr.append(g.iloc[:max(0, lo - n_gap)])
        tr.append(g.iloc[lo + n_te + n_gap:])
        te.append(g.iloc[lo:lo + n_te])
    return (pd.concat(tr).reset_index(drop=True),
            pd.concat(te).reset_index(drop=True))


def budget(df, n_target, seed):
    """Subsample to n_target rows, stratified by class, so every arm trains on
    the same number of windows."""
    if len(df) <= n_target:
        return df
    rng = np.random.default_rng(seed + 7919)
    take = []
    per = max(1, n_target // df["gas_id"].nunique())
    for cls, g in df.groupby("gas_id"):
        k = min(per, len(g))
        take.append(g.iloc[rng.choice(len(g), size=k, replace=False)])
    return pd.concat(take).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def accuracy(name, tr, te, seed, reg):
    tr, te, _, _ = rp.prepare(tr.copy(), te.copy(), anomaly_seed=seed)
    Xtr, gtr = rp.to_arrays(tr)
    Xte, gte = rp.to_arrays(te)
    ytr = np.array([RW.rule_oracle_action(int(g)) for g in gtr])
    mk, kwf = reg[name]
    m = mk(seed)
    kw = dict(kwf(seed, gtr))
    if "groups" in inspect.signature(m.fit).parameters:
        kw.setdefault("groups", gtr)
    m.fit(Xtr, ytr, **kw)
    acts = (m.predict(Xte, groups=gte)
            if "groups" in inspect.signature(m.predict).parameters else m.predict(Xte))
    yte = np.array([RW.rule_oracle_action(int(g)) for g in gte])
    from retrain.safety_metrics import CORRECT_ACTIONS
    return float(np.mean([int(a) in CORRECT_ACTIONS[int(g)] for a, g in zip(acts, gte)]))


def main():
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    thin_ds = thin(ds, STRIDE)
    n_budget = int(round((1 - TEST_FRAC) * len(thin_ds)))
    print(f"full set {len(ds)} windows, stride-{STRIDE} set {len(thin_ds)} windows")
    print(f"training budget for every arm: {n_budget} windows\n")

    ARMS = [("a_overlap_random", ds, random_split),
            ("b_disjoint_random", thin_ds, random_split),
            ("c_overlap_blocked", ds, blocked_split),
            ("d_disjoint_blocked", thin_ds, blocked_split)]

    rows = []
    for name in MODELS:
        t0 = time.time()
        reg = registry(None)
        line = {}
        for arm, source, splitter in ARMS:
            accs = []
            for s in SEEDS:
                tr, te = splitter(source, s)
                tr = budget(tr, n_budget, s)
                accs.append(accuracy(name, tr, te, s, reg))
            line[arm] = (float(np.mean(accs)), float(np.std(accs)))
            rows.append(dict(model=name, arm=arm, n_train_budget=n_budget,
                             decision_acc_mean=line[arm][0], decision_acc_std=line[arm][1]))
        overlap_effect = 100 * (line["a_overlap_random"][0] - line["b_disjoint_random"][0])
        position_effect = 100 * (line["b_disjoint_random"][0] - line["d_disjoint_blocked"][0])
        rows.append(dict(model=name, arm="effect_overlap_pp", n_train_budget=n_budget,
                         decision_acc_mean=overlap_effect, decision_acc_std=np.nan))
        rows.append(dict(model=name, arm="effect_position_pp", n_train_budget=n_budget,
                         decision_acc_mean=position_effect, decision_acc_std=np.nan))
        print(f"  {name:16s} a={line['a_overlap_random'][0]:.4f} b={line['b_disjoint_random'][0]:.4f} "
              f"c={line['c_overlap_blocked'][0]:.4f} d={line['d_disjoint_blocked'][0]:.4f}  "
              f"overlap={overlap_effect:+.2f}pp position={position_effect:+.2f}pp "
              f"[{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v3_overlap_isolation.csv"), index=False)
    eff = df[df.arm.isin(["effect_overlap_pp", "effect_position_pp"])]
    print("\nmean across models:")
    for arm, g in eff.groupby("arm"):
        print(f"  {arm:22s} {g.decision_acc_mean.mean():+.2f} pp "
              f"(range {g.decision_acc_mean.min():+.2f} to {g.decision_acc_mean.max():+.2f})")
    print("\nwrote v3_overlap_isolation.csv")


if __name__ == "__main__":
    main()
