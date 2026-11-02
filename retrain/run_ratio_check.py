"""Do the class-exclusion and degradation results for the cost-weighted policy
hold at the validation-selected ratio, 6:1, as well as at the reference 8:1?

Review comments 144 and 215. The reference ratio C = 8:1 was chosen on
test-partition dispersion; re-selecting it on an embargoed validation band gives
6:1 (v3_cost_blocked_validation.csv). A later validation analysis does not undo
the earlier test-informed choice, so instead of arguing that the choice does not
matter, this driver measures it: the cost-weighted policy is refitted at both
ratios, in the same run and with the same seeds and code as the main
experiments, and put through

  * class exclusion (each hazardous class withheld; Table 6 protocol),
  * the feature-space perturbation suite (Experiment 6 protocol), and
  * the raw-signal faults (Experiment 8 protocol).

The 8:1 arm is refitted here rather than read from the main result files so the
two arms differ in nothing but C.

Writes results_v2/v3_ratio_check.csv (one row per ratio x experiment x
condition, five-seed mean and SD) and v3_ratio_check_perseed.csv.

    python3 -m retrain.run_ratio_check
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrain import raw_pipeline as rp
from retrain import rewards as RW
from retrain.comparators import SupervisedDQN, cost_weighted_sample_weight
from retrain.run_all_v2 import SEEDS, OUT, CACHE, EPOCHS, perturb
from retrain.run_final import make, fit_pred, scale_with, SEEDS5
from retrain.run_raw_faults import FAULTS, apply_fault, features_from
from retrain.safety_metrics import evaluate_safety

RATIOS = (8, 6)
PERTURB_CONDS = ([("clean", 0.0)]
                 + [("noise", l) for l in (0.1, 0.2, 0.3, 0.4, 0.5)]
                 + [("drift", l) for l in (0.1, 0.2, 0.3, 0.4, 0.5)]
                 + [("dropout", k) for k in (1, 2, 3, 4, 5, 6, 7)])
KEYS = ("decision_acc", "miss_rate", "escalation", "under_escalation", "fa_per_clean")


def log(*a):
    print(*a, flush=True)


def summarize(evs):
    out = {}
    for k in KEYS:
        v = np.array([e[k] for e in evs], float)
        out[k + "_mean"] = float(np.nanmean(v))
        out[k + "_std"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
    for k in ("miss_k", "esc_k", "under_esc_k", "fa_k", "n_danger", "n_clean"):
        if k in evs[0]:
            out[k + "_pooled"] = int(sum(int(e[k]) for e in evs))
    return out


def class_exclusion(ds, C, rows, per_seed):
    """Table 6 protocol (run_final.exp_loco_all) for the cost-weighted policy."""
    for held in ("Smoke", "Mixture"):
        hid = rp.GAS_MAP[held]
        evs = []
        for s in SEEDS5:
            tr_df, te_df = rp.block_wise_holdout(ds, seed=s)
            tr3 = tr_df[tr_df["gas_id"] != hid].reset_index(drop=True)
            held_raw = ds[ds["gas_id"] == hid].reset_index(drop=True)
            te3 = te_df[te_df["gas_id"] != hid].reset_index(drop=True)
            tr3, held_raw, te3 = rp.partition_anomaly(tr3, held_raw, te3, seed=s)
            sc = StandardScaler().fit(tr3[rp.ALL_FEAT_COLS])
            p1 = float(np.percentile(tr3["anomaly"], 1))
            p99 = float(np.percentile(tr3["anomaly"], 99))
            tr3s = scale_with(tr3, sc, p1, p99)
            Xtr, gtr = rp.to_arrays(tr3s)
            ytr = np.array([RW.rule_oracle_action(int(g)) for g in gtr])
            Xh, gh = rp.to_arrays(scale_with(held_raw, sc, p1, p99))
            Xr, gr = rp.to_arrays(scale_with(te3, sc, p1, p99))
            m, kw = make("A_cost_weighted", s, gtr, C=C)
            acts = fit_pred(m, Xtr, ytr, gtr, Xh, gh, kw, seg=rp.segment_ids(tr3s))
            e = evaluate_safety(gh, acts)
            acts_r = np.asarray(m.predict(Xr)).astype(int)
            nog = gr == rp.GAS_MAP["NoGas"]
            e["nogas_alarm_rate"] = float((acts_r[nog] >= 3).mean()) if nog.any() else np.nan
            evs.append(e)
            per_seed.append(dict(ratio=f"{C}:1", experiment="class_exclusion",
                                 condition=held, level=0.0, seed=s,
                                 miss_k=int(e["miss_k"]), esc_k=int(e["esc_k"]),
                                 under_esc_k=int(e["under_esc_k"]),
                                 n_danger=int(e["n_danger"]),
                                 nogas_alarm_rate=e["nogas_alarm_rate"]))
        row = dict(ratio=f"{C}:1", experiment="class_exclusion", condition=held, level=0.0)
        row.update(summarize(evs))
        row["nogas_alarm_rate_mean"] = float(np.nanmean([e["nogas_alarm_rate"] for e in evs]))
        rows.append(row)
        log(f"  {C}:1 class exclusion {held:8s} miss={row['miss_rate_mean']:.4f} "
            f"esc={row['escalation_mean']:.4f} under={row['under_escalation_mean']:.4f} "
            f"NoGas alarms={row['nogas_alarm_rate_mean']:.4f}")


def feature_perturbation(ds, C, rows, per_seed):
    """Experiment 6 protocol (run_all_v2.exp_perturb) for the cost-weighted policy."""
    fitted = []
    for s in SEEDS:
        _, _, Xtr, gtr, ytr, Xte, gte = rp.split_and_prepare(ds, seed=s)
        m = SupervisedDQN(seed=s)
        m.fit(Xtr, ytr, sample_weight=cost_weighted_sample_weight(gtr, C, 1), epochs=EPOCHS)
        fitted.append((s, m, Xte, gte))
    for kind, lvl in PERTURB_CONDS:
        evs = []
        for s, m, Xte, gte in fitted:
            rng = np.random.default_rng(s)
            Xp = Xte if kind == "clean" else perturb(Xte, kind, lvl, rng)
            e = evaluate_safety(gte, m.predict(Xp))
            evs.append(e)
            per_seed.append(dict(ratio=f"{C}:1", experiment="feature_perturbation",
                                 condition=kind, level=float(lvl), seed=s,
                                 miss_k=int(e["miss_k"]), esc_k=int(e["esc_k"]),
                                 under_esc_k=int(e["under_esc_k"]),
                                 n_danger=int(e["n_danger"]), fa_k=int(e["fa_k"]),
                                 n_clean=int(e["n_clean"])))
        row = dict(ratio=f"{C}:1", experiment="feature_perturbation", condition=kind,
                   level=float(lvl))
        row.update(summarize(evs))
        rows.append(row)
    log(f"  {C}:1 feature perturbation done")


def raw_faults(ds, wins, C, rows, per_seed):
    """Experiment 8 protocol (run_raw_faults) for the cost-weighted policy."""
    fitted = []
    for s in SEEDS:
        tr, te = rp.block_wise_holdout(ds, seed=s)
        nom_ids = tr.loc[tr["label"] == "NoGas", "win_id"].astype(int).values
        ae, ae_sc = rp.train_anomaly_on(wins[nom_ids], seed=s)
        tr_p, te_p, feat_sc, (p1, p99) = rp.prepare(tr, te, anomaly_seed=s)
        Xtr, gtr = rp.to_arrays(tr_p)
        ytr = np.array([RW.rule_oracle_action(int(g)) for g in gtr])
        m = SupervisedDQN(seed=s)
        m.fit(Xtr, ytr, sample_weight=cost_weighted_sample_weight(gtr, C, 1), epochs=EPOCHS)
        tr_ids = tr["win_id"].astype(int).values
        fitted.append(dict(seed=s, model=m, te_ids=te["win_id"].astype(int).values,
                           gte=te["gas_id"].values, ae=ae, ae_sc=ae_sc, feat_sc=feat_sc,
                           p1=p1, p99=p99,
                           ch_max=wins[tr_ids].reshape(-1, 7).max(axis=0),
                           ch_sd=wins[tr_ids].reshape(-1, 7).std(axis=0)))
    for kind, level in FAULTS:
        evs = []
        for f in fitted:
            rng = np.random.default_rng(f["seed"])
            w = apply_fault(wins[f["te_ids"]], kind, level, rng, f["ch_max"], f["ch_sd"])
            anom = rp.score_anomaly(f["ae"], f["ae_sc"], w)
            anom_n = np.clip((anom - f["p1"]) / (f["p99"] - f["p1"] + 1e-8), 0, 1)
            X = np.column_stack([anom_n, f["feat_sc"].transform(features_from(w))])
            e = evaluate_safety(f["gte"], f["model"].predict(X))
            evs.append(e)
            per_seed.append(dict(ratio=f"{C}:1", experiment="raw_faults", condition=kind,
                                 level=float(level), seed=f["seed"],
                                 miss_k=int(e["miss_k"]), esc_k=int(e["esc_k"]),
                                 under_esc_k=int(e["under_esc_k"]),
                                 n_danger=int(e["n_danger"]), fa_k=int(e["fa_k"]),
                                 n_clean=int(e["n_clean"])))
        row = dict(ratio=f"{C}:1", experiment="raw_faults", condition=kind, level=float(level))
        row.update(summarize(evs))
        rows.append(row)
    log(f"  {C}:1 raw faults done")


def main():
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    wins = rp.raw_windows()
    rows, per_seed = [], []
    for C in RATIOS:
        t0 = time.time()
        class_exclusion(ds, C, rows, per_seed)
        feature_perturbation(ds, C, rows, per_seed)
        raw_faults(ds, wins, C, rows, per_seed)
        log(f"  {C}:1 finished [{time.time() - t0:.0f}s]")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "v3_ratio_check.csv"), index=False)
    pd.DataFrame(per_seed).to_csv(os.path.join(OUT, "v3_ratio_check_perseed.csv"), index=False)
    log("\nwrote v3_ratio_check.csv and v3_ratio_check_perseed.csv")


if __name__ == "__main__":
    main()
