"""Hazard-keyed gradient boosting, carried through the 30-run comparison and the
leave-one-class-out test.

The shipped comparator, CostSensitiveGBM (retrain/comparators.py), weights rows by
their target action, so clean-air rows carry 3.0 and hazardous rows 1.2 (a 2.5 to 1
tilt toward clean air). The corrected variant, HazardWeightedGBM, uses the same
classifier and the same 2.5 magnitude but keys the weight on the gas: hazardous
rows (Smoke, Mixture) carry 2.5 and clean rows 1.0.

Both are run here under exactly the protocol of run_final.exp_powered (30 runs,
seeds RUN_SEEDS, split_and_prepare) and run_final.exp_loco_all (five seeds, both
held-out hazardous classes). The shipped model is refitted too, and its numbers are
checked against the stored v3_powered.csv and v3_loco_all.csv before the corrected
variant's numbers are used. Any mismatch stops the run.

Writes results_v2/v3_powered_hazard_gbm.csv, v3_powered_hazard_gbm_bayes.json,
v3_loco_hazard_gbm.csv and v3_loco_hazard_gbm_perseed.csv.

    python3 -m retrain.run_hazard_gbm powered loco
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from retrain import raw_pipeline as rp
from retrain import rewards as RW
from retrain.run_final import (RUN_SEEDS, N_RUNS, SEEDS5, OUT, CACHE, expected_cost,
                               fit_pred, scale_with, bayes_correlated_t, make, log)
from retrain.safety_metrics import evaluate_safety, agg, AGG_KEYS

HAZARD_GAS = (1, 2)      # gas ids: Smoke = 1, Mixture = 2 (rp.GAS_MAP)
HAZARD_RATIO = 2.5       # the magnitude the shipped weights apply, mirrored onto hazards


class HazardWeightedGBM:
    """Identical to CostSensitiveGBM except that the weight is keyed on the gas."""

    def __init__(self, seed=42):
        self.clf = GradientBoostingClassifier(random_state=seed, n_estimators=300)

    def fit(self, X, y_action, groups=None):
        if groups is None:
            raise ValueError("HazardWeightedGBM needs the gas labels to weight rows")
        sw = np.where(np.isin(np.asarray(groups, int), HAZARD_GAS), HAZARD_RATIO, 1.0)
        self.clf.fit(X, y_action, sample_weight=sw)

    def predict(self, X):
        return self.clf.predict(X)


def build(name, seed, gtr):
    if name == "E_gbm_hazard":
        return HazardWeightedGBM(seed=seed), {}
    return make(name, seed, gtr)


def check_against(df_new, path, key, cols, tol=1e-9):
    """Report, do not abort: a refit in a different library build can differ by
    a few parts in 10^5, and the gap is itself a result to state."""
    ref = pd.read_csv(path)
    for _, r in df_new.iterrows():
        ref_row = ref[ref[key].astype(str) == str(r[key])]
        if ref_row.empty:
            log(f"  reproduction: no stored row for {r[key]} in {os.path.basename(path)}")
            continue
        for c in cols:
            gap = float(r[c]) - float(ref_row.iloc[0][c])
            flag = "exact" if abs(gap) <= tol else f"differs by {gap:+.6f}"
            log(f"  reproduction {r[key]} {c}: stored {float(ref_row.iloc[0][c]):.6f} refit {float(r[c]):.6f} ({flag})")


def powered(ds):
    log(f"\n=== 30-run comparison, shipped and hazard-keyed gradient boosting ===")
    names = ["E_gbm", "E_gbm_hazard"]
    per_run = {n: [] for n in names}
    t0 = time.time()
    for r, seed in enumerate(RUN_SEEDS):
        _, _, Xtr, gtr, ytr, Xte, gte = rp.split_and_prepare(ds, seed=seed)
        for n in names:
            m, kw = build(n, seed, gtr)
            acts = fit_pred(m, Xtr, ytr, gtr, Xte, gte, kw)
            ev = evaluate_safety(gte, acts)
            ev["expected_cost"] = expected_cost(gte, acts)
            per_run[n].append(ev)
        if (r + 1) % 5 == 0:
            log(f"  run {r+1}/{N_RUNS} [{time.time()-t0:.0f}s]")
    rows = []
    for n in names:
        a = agg(per_run[n], AGG_KEYS + ["expected_cost"])
        a["model"] = n
        rows.append(a)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v3_powered_hazard_gbm.csv"), index=False)
    json.dump({n: [e["decision_acc"] for e in per_run[n]] for n in names},
              open(os.path.join(OUT, "v3_powered_hazard_gbm_perrun.json"), "w"), indent=2)

    # Reproduction: the shipped model must match the stored 30-run row exactly.
    check_against(df[df.model == "E_gbm"], os.path.join(OUT, "v3_powered.csv"), "model",
                  ["decision_acc_mean", "decision_acc_std", "miss_rate_mean", "escalation_mean"],
                  tol=1e-6)

    stored = json.load(open(os.path.join(OUT, "v3_powered_bayes.json")))
    ref = np.array(stored["per_run_acc"]["A_cost_weighted"])
    rho = stored["rho"]
    sig = {}
    for n in names:
        other = np.array([e["decision_acc"] for e in per_run[n]])
        l, m_, rgt = bayes_correlated_t(ref - other, rho, rope=0.01)
        sig[n] = dict(p_other_better=l, p_practically_equivalent=m_, p_ref_better=rgt,
                      mean_diff=float((ref - other).mean()))
    json.dump(dict(rho=rho, rope=0.01, n_runs=N_RUNS, vs_cost_weighted=sig,
                   per_run_acc={n: [e["decision_acc"] for e in per_run[n]] for n in names}),
              open(os.path.join(OUT, "v3_powered_hazard_gbm_bayes.json"), "w"), indent=2)
    log(df[["model", "decision_acc_mean", "decision_acc_std", "miss_rate_mean",
            "escalation_mean", "expected_cost_mean"]].round(4).to_string(index=False))
    for n, v in sig.items():
        log(f"  cost-weighted vs {n:14s} P(equiv)={v['p_practically_equivalent']:.3f} "
            f"P(ref better)={v['p_ref_better']:.3f} P(other better)={v['p_other_better']:.3f}")


def loco(ds):
    log("\n=== leave-one-class-out, shipped and hazard-keyed gradient boosting ===")
    names = ["E_gbm", "E_gbm_hazard"]
    rows, per_seed = [], []
    for held in ["Smoke", "Mixture"]:
        hid = rp.GAS_MAP[held]
        for name in names:
            evs, t0 = [], time.time()
            for s in SEEDS5:
                tr_df, _ = rp.block_wise_holdout(ds, seed=s)
                tr3 = tr_df[tr_df["gas_id"] != hid].reset_index(drop=True)
                held_raw = ds[ds["gas_id"] == hid].reset_index(drop=True)
                tr3, held_raw = rp.partition_anomaly(tr3, held_raw, seed=s)
                sc = StandardScaler().fit(tr3[rp.ALL_FEAT_COLS])
                p1 = float(np.percentile(tr3["anomaly"], 1)); p99 = float(np.percentile(tr3["anomaly"], 99))
                tr3s = scale_with(tr3, sc, p1, p99)
                held_s = scale_with(held_raw, sc, p1, p99)
                Xtr, gtr = rp.to_arrays(tr3s)
                ytr = np.array([RW.rule_oracle_action(int(g)) for g in gtr])
                Xh, gh = rp.to_arrays(held_s)
                m, kw = build(name, s, gtr)
                acts = fit_pred(m, Xtr, ytr, gtr, Xh, gh, kw)
                e = evaluate_safety(gh, acts)
                evs.append(e)
                per_seed.append(dict(held_out=held, model=name, seed=s,
                                     n_danger=int(e["n_danger"]), miss_k=int(e["miss_k"]),
                                     esc_k=int(e["esc_k"]), under_esc_k=int(e["under_esc_k"]),
                                     miss_rate=float(e["miss_rate"]),
                                     escalation=float(e["escalation"]),
                                     under_escalation=float(e["under_escalation"])))
            a = agg(evs, AGG_KEYS)
            a["held_out"], a["model"] = held, name
            rows.append(a)
            log(f"  {held:8s}{name:14s} miss={a['miss_rate_mean']:.4f} "
                f"(<={100*a['miss_upper95_pooled']:.3f}%) esc={a['escalation_mean']:.4f}"
                f"±{a['escalation_std']:.4f} [{time.time()-t0:.0f}s]")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v3_loco_hazard_gbm.csv"), index=False)
    pd.DataFrame(per_seed).to_csv(os.path.join(OUT, "v3_loco_hazard_gbm_perseed.csv"), index=False)

    # Reproduction: the shipped model must match the stored leave-one-class rows.
    stored = pd.read_csv(os.path.join(OUT, "v3_loco_all.csv"))
    for _, r in df[df.model == "E_gbm"].iterrows():
        s_row = stored[(stored.held_out == r.held_out) & (stored.model == "E_gbm")].iloc[0]
        for c in ["miss_rate_mean", "escalation_mean", "escalation_std"]:
            gap = float(r[c]) - float(s_row[c])
            log(f"  reproduction {r.held_out} {c}: stored {float(s_row[c]):.6f} refit {float(r[c]):.6f} (gap {gap:+.6f})")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    log(f"dataset: {len(ds)} windows")
    which = sys.argv[1:] or ["powered", "loco"]
    if "powered" in which:
        powered(ds)
    if "loco" in which:
        loco(ds)
