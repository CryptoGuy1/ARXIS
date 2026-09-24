"""Sensor faults injected in raw sensor space, not in standardized feature space.

Experiment 6 perturbs the standardized decision state: it adds noise to already
scaled features, and it "drops" a channel by setting its standardized value to
zero, which under a training-partition standard scaler is the training mean. The
manuscript is explicit that this is a dimensionless stress test rather than a
physical fault model, and every review agreed while pointing out that the
stronger version of the experiment is available: perturb the raw seven-channel
window, then recompute everything downstream of it.

That is what this driver does. A fault is applied to the raw 20-by-7 window, and
then the current readings, the per-channel deltas, the per-channel standard
deviations AND the autoencoder reconstruction error are all recomputed from the
faulted window. Nothing is patched in feature space. A frozen channel really does
produce a zero delta and a zero standard deviation; an open circuit really does
present the autoencoder with a trace it has never seen.

Four fault modes, chosen because a fixed detector in service meets all four:

    stuck        the channel holds its first reading for the whole window, which
                 is what a frozen ADC or a failed sample-and-hold looks like
    zero         the channel reads 0 in raw units, an open circuit or a dead
                 element. This is the case Experiment 6 says it does not test.
    saturate     the channel pins at the largest value seen in training, an
                 out-of-range or rail-limited reading
    rawnoise     additive Gaussian noise at a fraction of each channel's own
                 training standard deviation, in raw units, so the level means
                 something about the signal rather than about the scaler

Severity for the three hard faults is the number of affected channels, k = 1, 4
and 7 of 7, matching the dropout sweep so the two experiments can be compared
row by row. For rawnoise the level is the fraction of the channel's training SD.

The anomaly model is the partition-local one: fitted on nominal training windows
only, then used to score the faulted windows. That matters here more than
anywhere else in the paper, because the whole point is that a raw fault reaches
the anomaly path and a standardized one does not.

Writes results_v2/v3_raw_faults.csv.

    python3 -m retrain.run_raw_faults
"""
import os, sys, time, inspect, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch

torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrain import raw_pipeline as rp
from retrain import rewards as RW
from retrain.run_all_v2 import registry, SEEDS, OUT, CACHE, PERTURB_MODELS
from retrain.safety_metrics import evaluate_safety

FAULTS = ([("clean", 0.0)]
          + [(f, k) for f in ("stuck", "zero", "saturate") for k in (1.0, 4.0, 7.0)]
          + [("rawnoise", lvl) for lvl in (0.5, 1.0, 2.0)])


def apply_fault(win, kind, level, rng, ch_max, ch_sd):
    """Fault a copy of the raw windows. win: (n, 20, 7) in raw sensor units."""
    w = np.array(win, dtype=float, copy=True)
    if kind == "clean":
        return w
    if kind == "rawnoise":
        return w + rng.normal(0.0, level * ch_sd, size=w.shape)
    k = int(level)
    idx = rng.choice(7, size=k, replace=False)
    if kind == "stuck":
        w[:, :, idx] = w[:, :1, idx]                 # hold the first reading
    elif kind == "zero":
        w[:, :, idx] = 0.0                           # open circuit, raw zero
    elif kind == "saturate":
        w[:, :, idx] = ch_max[idx]                   # pinned at the training max
    return w


def features_from(win):
    """The 21 sensor features the decision state carries, recomputed from raw."""
    current = win[:, -1, :]
    delta = win[:, -1, :] - win[:, 0, :]
    std = win.std(axis=1)
    return np.concatenate([current, delta, std], axis=1)


def main():
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    wins = rp.raw_windows()
    reg = registry(None)
    rows = []

    for name in PERTURB_MODELS:
        mk, kwf = reg[name]
        t0 = time.time()
        fitted = []
        for s in SEEDS:
            tr, te = rp.block_wise_holdout(ds, seed=s)
            # the partition-local anomaly model, kept so faulted windows can be
            # rescored by the same object the clean ones were scored by
            nom_ids = tr.loc[tr["label"] == "NoGas", "win_id"].astype(int).values
            ae, ae_sc = rp.train_anomaly_on(wins[nom_ids], seed=s)
            tr_p, te_p, feat_sc, (p1, p99) = rp.prepare(tr, te, anomaly_seed=s)
            Xtr, gtr = rp.to_arrays(tr_p)
            ytr = np.array([RW.rule_oracle_action(int(g)) for g in gtr])
            m = mk(s)
            kw = dict(kwf(s, gtr))
            if "groups" in inspect.signature(m.fit).parameters:
                kw.setdefault("groups", gtr)
            m.fit(Xtr, ytr, **kw)

            tr_ids = tr["win_id"].astype(int).values
            te_ids = te["win_id"].astype(int).values
            ch_max = wins[tr_ids].reshape(-1, 7).max(axis=0)
            ch_sd = wins[tr_ids].reshape(-1, 7).std(axis=0)
            fitted.append(dict(seed=s, model=m, te_ids=te_ids, gte=te["gas_id"].values,
                               ae=ae, ae_sc=ae_sc, feat_sc=feat_sc, p1=p1, p99=p99,
                               ch_max=ch_max, ch_sd=ch_sd))

        for kind, level in FAULTS:
            evs = []
            for f in fitted:
                rng = np.random.default_rng(f["seed"])
                w = apply_fault(wins[f["te_ids"]], kind, level, rng, f["ch_max"], f["ch_sd"])
                feats = features_from(w)
                anom = rp.score_anomaly(f["ae"], f["ae_sc"], w)
                anom_n = np.clip((anom - f["p1"]) / (f["p99"] - f["p1"] + 1e-8), 0, 1)
                X = np.column_stack([anom_n, f["feat_sc"].transform(feats)])
                gte = f["gte"]
                acts = (f["model"].predict(X, groups=gte)
                        if "groups" in inspect.signature(f["model"].predict).parameters
                        else f["model"].predict(X))
                e = evaluate_safety(gte, acts)
                e["anomaly_mean"] = float(anom_n.mean())
                evs.append(e)
            keys = ["decision_acc", "miss_rate", "escalation", "under_escalation",
                    "fa_per_clean", "anomaly_mean"]
            row = dict(model=name, fault=kind, level=level)
            for k in keys:
                v = np.array([e[k] for e in evs], float)
                row[k + "_mean"] = float(v.mean())
                row[k + "_std"] = float(v.std())
            rows.append(row)
            print(f"  {name:16s} {kind:9s}{level:>4.1f}  acc={row['decision_acc_mean']:.4f} "
                  f"miss={row['miss_rate_mean']:.4f} esc={row['escalation_mean']:.4f} "
                  f"fa={row['fa_per_clean_mean']:.4f} anom={row['anomaly_mean_mean']:.3f}",
                  flush=True)
        print(f"  {name:16s} done [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v3_raw_faults.csv"), index=False)
    print(f"\nwrote v3_raw_faults.csv ({len(df)} rows)")

    # the comparison the experiment exists for: a raw open circuit against the
    # standardized mean-imputation that stands in for it in Experiment 6
    try:
        pt = pd.read_csv(os.path.join(OUT, "v2_perturb.csv"))
        print("\nraw open circuit against standardized mean imputation, k = 1:")
        for name in PERTURB_MODELS:
            a = df[(df.model == name) & (df.fault == "zero") & (df.level == 1.0)]
            b = pt[(pt.model == name) & (pt.perturbation == "dropout") & (pt.level == 1.0)]
            if len(a) and len(b):
                print(f"  {name:16s} miss raw={a.miss_rate_mean.iloc[0]:.4f} "
                      f"standardized={b.miss_rate_mean.iloc[0]:.4f}   "
                      f"fa raw={a.fa_per_clean_mean.iloc[0]:.4f} "
                      f"standardized={b.fa_per_clean_mean.iloc[0]:.4f}")
    except Exception as exc:
        print(f"(comparison against v2_perturb.csv unavailable: {exc})")


if __name__ == "__main__":
    main()
