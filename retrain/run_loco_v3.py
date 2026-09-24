"""LOCO, corrected: the feature scaler is fitted on the THREE training classes
only. The first pass fitted it on all four, so the held-out class's feature
statistics leaked into the scaling - the same class of error this paper is
about, so it cannot stand in this paper's own experiment.
"""
import os, sys, json, inspect, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrain import raw_pipeline as rp
from retrain import rewards as RW
from retrain.run_all_v2 import registry, fit_predict, SEEDS, OUT, CACHE, LOCO_MODELS
from retrain.safety_metrics import evaluate_safety, agg, AGG_KEYS


def scale_with(df, scaler, p1, p99):
    d = df.copy()
    d[rp.ALL_FEAT_COLS] = scaler.transform(d[rp.ALL_FEAT_COLS])
    d["anomaly"] = ((d["anomaly"] - p1) / (p99 - p1 + 1e-8)).clip(0, 1)
    return d


def main():
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    reg = registry(None)
    rows = []
    for held in ["Smoke", "Mixture"]:
        hid = rp.GAS_MAP[held]
        for name in LOCO_MODELS:
            mk, kwf = reg[name]
            evs, t0 = [], time.time()
            for s in SEEDS:
                tr_df, _ = rp.block_wise_holdout(ds, seed=s)
                tr3 = tr_df[tr_df["gas_id"] != hid].reset_index(drop=True)
                held_raw = ds[ds["gas_id"] == hid].reset_index(drop=True)
                tr3, held_raw = rp.partition_anomaly(tr3, held_raw, seed=s)   # fix 4.2
                sc = StandardScaler().fit(tr3[rp.ALL_FEAT_COLS])
                p1 = float(np.percentile(tr3["anomaly"], 1))
                p99 = float(np.percentile(tr3["anomaly"], 99))
                tr3s = scale_with(tr3, sc, p1, p99)
                held_s = scale_with(held_raw, sc, p1, p99)
                Xtr, gtr = rp.to_arrays(tr3s)
                ytr = np.array([RW.rule_oracle_action(int(g)) for g in gtr])
                Xh, gh = rp.to_arrays(held_s)
                acts, _ = fit_predict(mk(s), Xtr, ytr, gtr, Xh, gh, kwf(s, gtr))
                evs.append(evaluate_safety(gh, acts))
            a = agg(evs, AGG_KEYS)
            a["held_out"], a["model"] = held, name
            rows.append(a)
            print(f"  {held:8s} {name:16s} miss={a['miss_rate_mean']:.4f}±{a['miss_rate_std']:.4f} "
                  f"(<={100*a['miss_upper95_pooled']:.3f}% pooled)  "
                  f"esc={a['escalation_mean']:.4f}±{a['escalation_std']:.4f} "
                  f"under={a['under_escalation_mean']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "v2_loco.csv"), index=False)
    print("wrote v2_loco.csv (scaler fitted on 3 training classes only)")


if __name__ == "__main__":
    main()
