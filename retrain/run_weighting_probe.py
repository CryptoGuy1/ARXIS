"""Does the cost-sensitive GBM weight the thing it says it weights, and does the
ordinal objective's zero-miss result mean anything?

Two review findings are answered here with measurements rather than argument.

The first. `CostSensitiveGBM` in retrain/comparators.py applies

    ACTION_WEIGHT = {0: 3.0, 1: 1.0, 2: 1.0, 3: 1.2, 4: 1.2}

to training rows keyed on the *target action*. The target action for a clean-air
window is 0 and for a hazardous window is 3 or 4, so the weights up-weight clean
air by 3.0 / 1.2 = 2.5 against hazards. The comment above them argues the
opposite ("danger-miss is action 0 on danger gas"), but no training row for a
hazardous window ever carries target 0, so that row never exists. The model
described in the paper as cost-sensitive toward hazards is, in the code, 2.5
times more sensitive to clean air.

This driver refits the same classifier under the weighting the name implies,
hazardous rows at 2.5 against clean, keyed on the gas rather than the target,
and measures both under the clean condition and under the raw zero fault that
produced the silent-failure result. If the silent failure is an artifact of the
sign, it goes away here.

The second. The ordinal objective reports a zero missed-hazard rate and a
decision accuracy pinned at exactly 0.7500 in 23 of 30 runs. Three of four
classes correct and one entirely wrong reads like a model that never emits one
action. A missed hazard is defined as a hazardous window assigned action 0, so a
model that never assigns action 0 cannot record one, whatever it does with the
hazard. This driver records the full action histogram per class so the question
is settled by counting rather than by inference from an accuracy.

Writes results_v2/v3_weighting_probe.csv and results_v2/v3_ordinal_actions.csv.

    python3 -m retrain.run_weighting_probe
"""
import inspect
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier

torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrain import raw_pipeline as rp
from retrain import rewards as RW
from retrain.run_all_v2 import registry, SEEDS, OUT, CACHE
from retrain.run_raw_faults import apply_fault, features_from
from retrain.safety_metrics import evaluate_safety

HAZARD_GAS = (1, 2)          # Smoke, Mixture
HAZARD_RATIO = 2.5           # the magnitude the shipped weights apply, mirrored

FAULTS = ([("clean", 0.0)]
          + [("zero", k) for k in (1.0, 4.0, 7.0)]
          + [("saturate", k) for k in (1.0, 4.0, 7.0)])

# The anomaly-input probe is where the gradient-boosted model separates from
# every other comparator: forcing the normalized anomaly feature to a constant
# moves its missed-hazard rate from 0.0092 to 0.4203 while no other model moves
# at all. If that dependence is a consequence of the weighting sign rather than
# of the model family, it changes under the corrected weights.
ANOM_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


class HazardWeightedGBM:
    """The same classifier, weighted by the cost of getting that row wrong.

    Identical to CostSensitiveGBM except that the weight is keyed on the gas of
    the row rather than on its target action, so a hazardous window is the one
    carrying the larger weight.
    """

    def __init__(self, seed=42):
        self.clf = GradientBoostingClassifier(random_state=seed, n_estimators=300)

    def fit(self, X, y_action, groups=None):
        if groups is None:
            raise ValueError("HazardWeightedGBM needs the gas labels to weight rows")
        sw = np.where(np.isin(np.asarray(groups, int), HAZARD_GAS), HAZARD_RATIO, 1.0)
        self.clf.fit(X, y_action, sample_weight=sw)

    def predict(self, X):
        return self.clf.predict(X)


def main():
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    wins = rp.raw_windows()
    reg = registry(None)

    arms = {
        "E_gbm_as_shipped": reg["E_gbm"],
        "E_gbm_hazard_weighted": (lambda s: HazardWeightedGBM(s), lambda s, g: {}),
    }

    rows, act_rows = [], []
    for name, (mk, kwf) in arms.items():
        t0 = time.time()
        fitted = []
        for s in SEEDS:
            tr, te = rp.block_wise_holdout(ds, seed=s)
            nom = tr.loc[tr["label"] == "NoGas", "win_id"].astype(int).values
            ae, ae_sc = rp.train_anomaly_on(wins[nom], seed=s)
            tr_p, te_p, feat_sc, (p1, p99) = rp.prepare(tr, te, anomaly_seed=s)
            Xtr, gtr = rp.to_arrays(tr_p)
            ytr = np.array([RW.rule_oracle_action(int(g)) for g in gtr])
            m = mk(s)
            kw = dict(kwf(s, gtr))
            if "groups" in inspect.signature(m.fit).parameters:
                kw.setdefault("groups", gtr)
            m.fit(Xtr, ytr, **kw)
            te_ids = te["win_id"].astype(int).values
            tr_ids = tr["win_id"].astype(int).values
            fitted.append(dict(seed=s, model=m, te_ids=te_ids, gte=te["gas_id"].values,
                               ae=ae, ae_sc=ae_sc, feat_sc=feat_sc, p1=p1, p99=p99,
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
                acts = f["model"].predict(X)
                e = evaluate_safety(f["gte"], acts)
                e["anomaly_mean"] = float(anom_n.mean())
                evs.append(e)
            row = dict(model=name, fault=kind, level=level)
            for k in ("decision_acc", "miss_rate", "escalation",
                      "under_escalation", "fa_per_clean", "anomaly_mean"):
                v = np.array([e[k] for e in evs], float)
                row[k + "_mean"], row[k + "_std"] = float(v.mean()), float(v.std())
            rows.append(row)
            print(f"  {name:24s} {kind:6s}{level:>4.1f}  acc={row['decision_acc_mean']:.4f} "
                  f"miss={row['miss_rate_mean']:.4f} fa={row['fa_per_clean_mean']:.4f}",
                  flush=True)
        # the anomaly-input probe, on the clean windows, under this weighting
        for lvl in ANOM_LEVELS:
            evs = []
            for f in fitted:
                w = wins[f["te_ids"]]
                anom = rp.score_anomaly(f["ae"], f["ae_sc"], w)
                anom_n = np.clip((anom - f["p1"]) / (f["p99"] - f["p1"] + 1e-8), 0, 1)
                forced = np.full_like(anom_n, lvl)
                X = np.column_stack([forced, f["feat_sc"].transform(features_from(w))])
                e = evaluate_safety(f["gte"], f["model"].predict(X))
                e["anomaly_mean"] = float(lvl)
                evs.append(e)
            row = dict(model=name, fault="anomaly_forced", level=lvl)
            for k in ("decision_acc", "miss_rate", "escalation",
                      "under_escalation", "fa_per_clean", "anomaly_mean"):
                v = np.array([e[k] for e in evs], float)
                row[k + "_mean"], row[k + "_std"] = float(v.mean()), float(v.std())
            rows.append(row)
            print(f"  {name:24s} anom  {lvl:>4.2f}  acc={row['decision_acc_mean']:.4f} "
                  f"miss={row['miss_rate_mean']:.4f} esc={row['escalation_mean']:.4f}",
                  flush=True)
        print(f"  {name:24s} done [{time.time() - t0:.0f}s]", flush=True)

    # --- the ordinal objective's action histogram, per class, clean condition ---
    # OrdinalCostNet lives in run_final rather than in the shared registry, and
    # it takes the gas labels as g_class rather than as groups.
    from retrain.run_final import OrdinalCostNet
    for s in SEEDS:
        tr, te = rp.block_wise_holdout(ds, seed=s)
        tr_p, te_p, feat_sc, _ = rp.prepare(tr, te, anomaly_seed=s)
        Xtr, gtr = rp.to_arrays(tr_p)
        Xte, gte = rp.to_arrays(te_p)
        ytr = np.array([RW.rule_oracle_action(int(g)) for g in gtr])
        m = OrdinalCostNet(seed=s)
        m.fit(Xtr, ytr, g_class=gtr)
        acts = np.asarray(m.predict(Xte), int)
        for gid, gname in enumerate(["NoGas", "Smoke", "Mixture", "Perfume"]):
            sel = np.asarray(gte, int) == gid
            if not sel.any():
                continue
            for a in range(5):
                act_rows.append(dict(seed=s, gas=gname, action=a,
                                     count=int((acts[sel] == a).sum()),
                                     n=int(sel.sum())))
        print(f"  Ordinal seed {s}: actions ever emitted = {sorted(set(acts.tolist()))}",
              flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "v3_weighting_probe.csv"), index=False)
    pd.DataFrame(act_rows).to_csv(os.path.join(OUT, "v3_ordinal_actions.csv"), index=False)
    print("\nwrote v3_weighting_probe.csv and v3_ordinal_actions.csv")


if __name__ == "__main__":
    main()
