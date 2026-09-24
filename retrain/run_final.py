"""Final experiment programme. Closes the open items that are closable.

E10  Powered comparison: 30 randomized runs varying seed, split position and a
     hyperparameter draw, with a Bayesian correlated t-test over a region of
     practical equivalence (Benavoli et al. 2017). Replaces "not statistically
     distinguishable at n = 5" with a positive statement about equivalence.
E11  Leave-one-class-out extended to every multiclass comparator.
E12  Ordinal cost-matrix objective: under- and over-escalation priced
     differently along the action ladder.
E13  Cost ratio re-selected on a validation split rather than on test dispersion.
E14  Reliability curves for the calibration estimators.
"""
import os, sys, json, time, inspect, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy.stats import t as student_t
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrain import raw_pipeline as rp
from retrain import rewards as RW
from retrain.comparators import SupervisedDQN, SupervisedMLP, CostSensitiveGBM
from retrain.comparators import cost_weighted_sample_weight
from retrain.zoo import SVMClassifier, RandomForest, RawWindowLSTM, CQLAgent
from retrain.new_baselines import KNN, ThresholdRule
from retrain.safety_metrics import evaluate_safety, agg, AGG_KEYS
from retrain.agent_rl import DuelingDQN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "retrain", "results_v2")
CACHE = os.path.join(ROOT, "retrain", "results", "real_features.csv")
EPOCHS = 80
N_RUNS = 30
RUN_SEEDS = list(range(1000, 1000 + N_RUNS))
SEEDS5 = [42, 1337, 7, 2024, 99]


def log(*a):
    print(*a, flush=True)


# =====================================================================
# E12  ordinal cost matrix
# =====================================================================
# Rows: gas class (0 NoGas, 1 Smoke, 2 Mixture, 3 Perfume). Columns: action 0..4.
# Costs are stated in units of "operational consequence" and are deliberately
# coarse. The structure, not the exact numbers, is what the experiment tests:
# for a hazardous class, silence costs more than a sub-alarm response, which
# costs more than notifying at the wrong tier; for clean air, a high-severity
# alarm costs more than a nuisance action.
COST = np.array([
    #  a=0   a=1   a=2   a=3   a=4
    [  0.0,  1.0,  1.0,  4.0,  4.0],   # NoGas
    [ 10.0,  5.0,  5.0,  0.0,  1.0],   # Smoke   (target 3)
    [ 10.0,  5.0,  5.0,  1.0,  0.0],   # Mixture (target 4)
    [  2.0,  0.0,  0.0,  3.0,  3.0],   # Perfume (target {1,2})
], dtype=np.float32)


class OrdinalCostNet:
    """Same dueling topology, trained to minimize expected cost under COST."""

    def __init__(self, seed=42, cost=None):
        torch.manual_seed(seed)
        self.net = DuelingDQN()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=2e-4)
        self.C = torch.tensor(COST if cost is None else cost, dtype=torch.float32)

    def fit(self, X, y_action, g_class=None, epochs=EPOCHS, batch=256):
        Xt = torch.from_numpy(np.asarray(X, np.float32))
        g = torch.from_numpy(np.asarray(g_class, np.int64))
        n = len(Xt)
        self.net.train()
        for _ in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                p = torch.softmax(self.net(Xt[idx]), dim=1)      # (b, 5)
                cost_rows = self.C[g[idx]]                        # (b, 5)
                loss = (p * cost_rows).sum(1).mean()              # expected cost
                self.opt.zero_grad(); loss.backward(); self.opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        """The network is trained so that its output distribution places mass on
        low-cost actions, so the arg-max is the minimum-expected-cost choice."""
        self.net.eval()
        p = torch.softmax(self.net(torch.from_numpy(np.asarray(X, np.float32))), dim=1).numpy()
        return p.argmax(1)


def expected_cost(gas, action):
    g = np.asarray(gas, int); a = np.asarray(action, int)
    return float(COST[g, a].mean())


# =====================================================================
# E10  powered comparison
# =====================================================================
def hp_draw(rng, name):
    """A small hyperparameter perturbation per run, so the comparison varies
    seed, split AND hyperparameter draw (Bouthillier et al. 2021)."""
    if name in ("A_cost_weighted", "B_unweighted", "Ordinal"):
        return dict(epochs=int(rng.integers(60, 101)))
    return {}


def make(name, seed, gtr, C=8):
    if name == "A_cost_weighted":
        return SupervisedDQN(seed=seed), dict(sample_weight=cost_weighted_sample_weight(gtr, C, 1))
    if name == "B_unweighted":  return SupervisedDQN(seed=seed), dict(sample_weight=None)
    if name == "Ordinal":       return OrdinalCostNet(seed=seed), dict(g_class=gtr)
    if name == "D_mlp":         return SupervisedMLP(seed=seed), {}
    if name == "E_gbm":         return CostSensitiveGBM(seed=seed), {}
    if name == "SVM":           return SVMClassifier(seed=seed), {}
    if name == "RF":            return RandomForest(seed=seed), {}
    if name == "LSTM":          return RawWindowLSTM(seed=seed, K=10), dict(epochs=40)
    if name == "CQL":           return CQLAgent(seed=seed, alpha=1.0), dict(epochs=EPOCHS, g_class=gtr)
    if name == "KNN":           return KNN(seed=seed), {}
    if name == "ThresholdRule": return ThresholdRule(seed=seed), {}
    raise KeyError(name)


POWERED = ["A_cost_weighted", "Ordinal", "B_unweighted", "D_mlp", "E_gbm",
           "SVM", "RF", "LSTM", "CQL", "KNN", "ThresholdRule"]


def fit_pred(m, Xtr, ytr, gtr, Xte, gte, kw):
    kw = dict(kw)
    if "groups" in inspect.signature(m.fit).parameters:
        kw.setdefault("groups", gtr)
    m.fit(Xtr, ytr, **kw)
    if "groups" in inspect.signature(m.predict).parameters:
        return m.predict(Xte, groups=gte)
    return m.predict(Xte)


def bayes_correlated_t(diff, rho, rope=0.01):
    """Benavoli et al. (2017) correlated Bayesian t-test.
    Returns P(left), P(rope), P(right) for `diff` = A - B."""
    d = np.asarray(diff, float); n = len(d)
    m = d.mean(); v = d.var(ddof=1)
    if v <= 0:
        return (0.0, 1.0, 0.0) if abs(m) < rope else ((1.0, 0.0, 0.0) if m < 0 else (0.0, 0.0, 1.0))
    scale = np.sqrt(v * (1.0 / n + rho / (1.0 - rho)))
    df = n - 1
    lo = student_t.cdf((-rope - m) / scale, df)
    hi = 1.0 - student_t.cdf((rope - m) / scale, df)
    return float(lo), float(1.0 - lo - hi), float(hi)


def exp_powered(ds):
    log(f"\n=== E10  powered comparison, {N_RUNS} randomized runs ===")
    per_run = {n: [] for n in POWERED}
    for r, seed in enumerate(RUN_SEEDS):
        rng = np.random.default_rng(seed)
        _, _, Xtr, gtr, ytr, Xte, gte = rp.split_and_prepare(ds, seed=seed)
        for name in POWERED:
            m, kw = make(name, seed, gtr)
            kw.update(hp_draw(rng, name))
            acts = fit_pred(m, Xtr, ytr, gtr, Xte, gte, kw)
            ev = evaluate_safety(gte, acts)
            ev["expected_cost"] = expected_cost(gte, acts)
            per_run[name].append(ev)
        if (r + 1) % 5 == 0:
            log(f"  run {r+1}/{N_RUNS} done")
    rows = []
    for name in POWERED:
        a = agg(per_run[name], AGG_KEYS + ["expected_cost"])
        a["model"] = name
        rows.append(a)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v3_powered.csv"), index=False)

    # Bayesian equivalence, reference = cost-weighted policy.
    # Plan item A5: rho comes from the partition this run actually produced, not
    # from a constant. The previous hard-coded pair (1264, 4980) disagreed with
    # the 4,900 training windows the pipeline builds and the manuscript reports.
    _tr0, _te0 = rp.block_wise_holdout(ds, seed=RUN_SEEDS[0])
    n_tr, n_te = len(_tr0), len(_te0)
    rho = n_te / (n_tr + n_te)
    log(f"  rho = {n_te}/({n_tr}+{n_te}) = {rho:.17g}")
    ref = np.array([e["decision_acc"] for e in per_run["A_cost_weighted"]])
    sig = {}
    for name in POWERED:
        if name == "A_cost_weighted":
            continue
        other = np.array([e["decision_acc"] for e in per_run[name]])
        l, m, rgt = bayes_correlated_t(ref - other, rho, rope=0.01)
        sig[name] = dict(p_other_better=l, p_practically_equivalent=m, p_ref_better=rgt,
                         mean_diff=float((ref - other).mean()))
    json.dump(dict(rho=rho, rope=0.01, n_runs=N_RUNS, vs_cost_weighted=sig,
                   per_run_acc={k: [e["decision_acc"] for e in v] for k, v in per_run.items()}),
              open(os.path.join(OUT, "v3_powered_bayes.json"), "w"), indent=2)
    log(df[["model", "decision_acc_mean", "decision_acc_std", "miss_rate_mean",
            "escalation_mean", "expected_cost_mean"]].round(4).to_string(index=False))
    for k, v in sig.items():
        log(f"  A vs {k:16s} P(equiv)={v['p_practically_equivalent']:.3f} "
            f"P(A better)={v['p_ref_better']:.3f} P(other better)={v['p_other_better']:.3f}")
    return df


# =====================================================================
# E11  LOCO, all multiclass comparators
# =====================================================================
LOCO_ALL = ["A_cost_weighted", "Ordinal", "B_unweighted", "D_mlp", "E_gbm",
            "SVM", "RF", "LSTM", "CQL", "KNN", "ThresholdRule"]


def scale_with(df, sc, p1, p99):
    d = df.copy()
    d[rp.ALL_FEAT_COLS] = sc.transform(d[rp.ALL_FEAT_COLS])
    d["anomaly"] = ((d["anomaly"] - p1) / (p99 - p1 + 1e-8)).clip(0, 1)
    return d


def exp_loco_all(ds):
    log("\n=== E11  leave-one-class-out, every comparator ===")
    # Pre-submission plan item A1. The bound column of Table 10 is a per-partition
    # quantity, so it has to be computed from the integer miss count each partition
    # actually produced. Averaging the rate across seeds and rounding the product
    # back to a count is not the same thing, and where the seeds disagree it is
    # optimistic. Per-seed counts are written alongside the aggregate so the bound
    # can be recomputed without rerunning anything.
    rows, per_seed = [], []
    for held in ["Smoke", "Mixture"]:
        hid = rp.GAS_MAP[held]
        for name in LOCO_ALL:
            evs, t0 = [], time.time()
            for s in SEEDS5:
                tr_df, _ = rp.block_wise_holdout(ds, seed=s)
                tr3 = tr_df[tr_df["gas_id"] != hid].reset_index(drop=True)
                held_raw = ds[ds["gas_id"] == hid].reset_index(drop=True)
                tr3, held_raw = rp.partition_anomaly(tr3, held_raw, seed=s)   # fix 4.2
                sc = StandardScaler().fit(tr3[rp.ALL_FEAT_COLS])
                p1 = float(np.percentile(tr3["anomaly"], 1)); p99 = float(np.percentile(tr3["anomaly"], 99))
                tr3s = scale_with(tr3, sc, p1, p99)
                held_s = scale_with(held_raw, sc, p1, p99)
                Xtr, gtr = rp.to_arrays(tr3s)
                ytr = np.array([RW.rule_oracle_action(int(g)) for g in gtr])
                Xh, gh = rp.to_arrays(held_s)
                m, kw = make(name, s, gtr)
                acts = fit_pred(m, Xtr, ytr, gtr, Xh, gh, kw)
                e = evaluate_safety(gh, acts)
                evs.append(e)
                per_seed.append(dict(held_out=held, model=name, seed=s,
                                     n_danger=int(e["n_danger"]),
                                     miss_k=int(e["miss_k"]),
                                     esc_k=int(e["esc_k"]),
                                     under_esc_k=int(e["under_esc_k"]),
                                     miss_rate=float(e["miss_rate"]),
                                     escalation=float(e["escalation"]),
                                     under_escalation=float(e["under_escalation"])))
            a = agg(evs, AGG_KEYS); a["held_out"], a["model"] = held, name
            rows.append(a)
            log(f"  {held:8s}{name:16s} miss={a['miss_rate_mean']:.4f} "
                f"(<={100*a['miss_upper95_pooled']:.3f}%) esc={a['escalation_mean']:.4f}"
                f"±{a['escalation_std']:.4f} [{time.time()-t0:.0f}s]")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "v3_loco_all.csv"), index=False)
    pd.DataFrame(per_seed).to_csv(os.path.join(OUT, "v3_loco_perseed.csv"), index=False)


# =====================================================================
# E13  cost ratio selected on a validation split
# =====================================================================
RATIOS = [1, 2, 4, 6, 8, 10, 12, 16, 20]


def exp_cost_validation(ds):
    log("\n=== E13  cost ratio selected on validation, not test ===")
    rows = []
    for C in RATIOS:
        va, te = [], []
        for s in SEEDS5:
            tr_df, te_df = rp.block_wise_holdout(ds, seed=s)
            tr_df, te_df, _, _ = rp.prepare(tr_df, te_df)
            Xall, gall = rp.to_arrays(tr_df)
            yall = np.array([RW.rule_oracle_action(int(g)) for g in gall])
            rng = np.random.default_rng(s); idx = rng.permutation(len(Xall))
            nv = int(0.2 * len(idx)); vi, fi = idx[:nv], idx[nv:]
            m = SupervisedDQN(seed=s)
            sw = None if C == 1 else cost_weighted_sample_weight(gall[fi], C, 1)
            m.fit(Xall[fi], yall[fi], sample_weight=sw, epochs=EPOCHS)
            va.append(evaluate_safety(gall[vi], m.predict(Xall[vi])))
            Xte, gte = rp.to_arrays(te_df)
            te.append(evaluate_safety(gte, m.predict(Xte)))
        av = agg(va, AGG_KEYS); at = agg(te, AGG_KEYS)
        rows.append(dict(cost_ratio=f"{C}:1",
                         val_acc_mean=av["decision_acc_mean"], val_acc_std=av["decision_acc_std"],
                         val_miss=av["miss_rate_mean"], val_esc=av["escalation_mean"],
                         test_acc_mean=at["decision_acc_mean"], test_acc_std=at["decision_acc_std"],
                         test_miss=at["miss_rate_mean"], test_esc=at["escalation_mean"]))
        log(f"  C={C:2d}:1 val={av['decision_acc_mean']:.4f} test={at['decision_acc_mean']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v3_cost_validation.csv"), index=False)
    best = df.loc[df.val_acc_mean.idxmax()]
    log(f"  selected on validation: {best.cost_ratio}  (test acc {best.test_acc_mean:.4f})")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    log(f"dataset: {len(ds)} windows")
    which = sys.argv[1:] or ["costval", "loco", "powered"]
    if "costval" in which: exp_cost_validation(ds)
    if "loco" in which:    exp_loco_all(ds)
    if "powered" in which: exp_powered(ds)
    log("\ndone ->", OUT)
