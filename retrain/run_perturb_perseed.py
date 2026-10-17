"""Per-partition counts for the perturbation claims in Experiments 6 and Table 7.

run_all_v2.py pools the five seeds into v2_perturb.csv, so a statement about one
partition cannot be read from that file. This driver refits the models the claims
name, under the same seeds, splits and perturbation function, and writes every
partition's counts to retrain/results_v2/v2_perturb_perseed.csv. Its pooled totals
reproduce v2_perturb.csv exactly; verify_v15.py checks that they do.

Models: the cost-weighted policy, the unweighted network, the shallow decision
tree (ThresholdRule) at one and seven lost channels, and gradient boosting at seven.
"""
import os, sys, inspect, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from retrain import run_all_v2 as R
from retrain import raw_pipeline as rp
from retrain.safety_metrics import evaluate_safety

WANT = {"A_cost_weighted": [1.0, 7.0], "B_unweighted": [1.0, 7.0],
        "ThresholdRule": [1.0, 7.0], "E_gbm": [7.0]}


def main():
    ds, _, _ = R.get_ds()
    reg = R.registry(None)
    rows = []
    for name, levels in WANT.items():
        mk, kwf = reg[name]
        for s in R.SEEDS:
            _, _, Xtr, gtr, ytr, Xte, gte = rp.split_and_prepare(ds, seed=s)
            m = mk(s)
            kw = dict(kwf(s, gtr))
            if "groups" in inspect.signature(m.fit).parameters:
                kw.setdefault("groups", gtr)
            m.fit(Xtr, ytr, **kw)
            for lvl in levels:
                rng = np.random.default_rng(s)
                Xp = R.perturb(Xte, "dropout", lvl, rng)
                acts = (m.predict(Xp, groups=gte)
                        if "groups" in inspect.signature(m.predict).parameters else m.predict(Xp))
                e = evaluate_safety(gte, acts)
                rows.append(dict(model=name, perturbation="dropout", level=int(lvl), seed=s,
                                 n_clean=int(e["n_clean"]), fa_k=int(e["fa_k"]),
                                 n_danger=int(e["n_danger"]), miss_k=int(e["miss_k"])))
            print(name, s, "done", flush=True)
    out = os.path.join(ROOT, "retrain", "results_v2", "v2_perturb_perseed.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
