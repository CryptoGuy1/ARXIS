"""Escalation adequacy at each alarm boundary on the held-out Mixture class.

Reviewer-facing check for the sentence "moving the alarm boundary down one
rung collapses that spread to 0.751 to 1.000". The leave-one-class-out driver
(run_final.py, exp_loco_all) stores escalation only at the alarm boundary
a >= 3. This driver re-runs the same held-out Mixture protocol for the ten
comparators (same seeds, same scaler, same partitioning) and records, per
seed, the share of hazardous windows whose action reaches

    a >= 3  (alarm grade, Eq. 6 as written)
    a >= 2  (one rung below the alarm grade: Request verification)
    a >= 1  (any action above Monitor)

so the spread at each boundary can be read from a stored file rather than
recalled. The a >= 3 column must reproduce v3_loco_all.csv; that is the check
on this driver itself.

Writes results_v2/v3_rung_check.csv (per seed) and v3_rung_check_summary.csv.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrain import raw_pipeline as rp
from retrain import rewards as RW
from retrain.run_final import make, fit_pred, scale_with, SEEDS5, OUT, LOCO_ALL

COMPARATORS = [m for m in LOCO_ALL if m != "Ordinal"]
HELD = "Mixture"
BOUNDARIES = {3: "a>=3 (alarm grade, Eq. 6)", 2: "a>=2 (one rung down)", 1: "a>=1 (any action above Monitor)"}


def main():
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "real_features.csv"))
    hid = rp.GAS_MAP[HELD]
    rows = []
    for name in COMPARATORS:
        t0 = time.time()
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
            m, kw = make(name, s, gtr)
            acts = np.asarray(fit_pred(m, Xtr, ytr, gtr, Xh, gh, kw)).astype(int)
            n = len(acts)
            rec = dict(model=name, seed=s, n_hazard=n)
            for k in BOUNDARIES:
                rec[f"esc_ge{k}"] = float((acts >= k).mean())
            rec["miss_rate"] = float((acts == 0).mean())
            rows.append(rec)
        print(f"  {name:16s} [{time.time()-t0:.0f}s]", flush=True)

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(OUT, "v3_rung_check.csv"), index=False)

    summ = []
    for name, g in per_seed.groupby("model", sort=False):
        r = dict(model=name)
        for k in BOUNDARIES:
            r[f"esc_ge{k}_mean"] = g[f"esc_ge{k}"].mean()
            r[f"esc_ge{k}_std"] = g[f"esc_ge{k}"].std(ddof=1)
        r["miss_rate_mean"] = g["miss_rate"].mean()
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(OUT, "v3_rung_check_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print(summ.round(4).to_string(index=False))
    for k in BOUNDARIES:
        col = summ[f"esc_ge{k}_mean"]
        print(f"spread at {BOUNDARIES[k]}: {col.min():.4f} to {col.max():.4f}")


if __name__ == "__main__":
    main()
