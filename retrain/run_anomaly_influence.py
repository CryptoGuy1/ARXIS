"""E9. Does the anomaly score actually influence the selected action?

The ablation in Section 4.5 removes the anomaly feature and retrains, which
answers "is the feature necessary". This asks the sharper counterfactual
question: holding a trained model and every other feature fixed, does moving
the anomaly input across its whole range change what the model does?

Feature 0 of the 22-dimensional state is overridden across [0, 1] on every test
window, for every model, and we record how often the selected action changes
and how the safety metrics move. A model whose actions never move is one whose
anomaly channel is inert at inference time, whatever its training-time
contribution.
"""
import os, sys, json, warnings, inspect
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrain import raw_pipeline as rp
from retrain.comparators import SupervisedDQN, SupervisedMLP, CostSensitiveGBM
from retrain.comparators import cost_weighted_sample_weight
from retrain.zoo import SVMClassifier, RandomForest
from retrain.new_baselines import KNN, ThresholdRule
from retrain.safety_metrics import evaluate_safety

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "retrain", "results_v2")
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "retrain", "results", "real_features.csv")
SEEDS = [42, 1337, 7, 2024, 99]
EPOCHS = 80
LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

MAKERS = {
    "A_cost_weighted": lambda s, g: (SupervisedDQN(seed=s),
                                     dict(sample_weight=cost_weighted_sample_weight(g, 8, 1), epochs=EPOCHS)),
    "D_mlp":         lambda s, g: (SupervisedMLP(seed=s), {}),
    "E_gbm":         lambda s, g: (CostSensitiveGBM(seed=s), {}),
    "RF":            lambda s, g: (RandomForest(seed=s), {}),
    "SVM":           lambda s, g: (SVMClassifier(seed=s), {}),
    "KNN":           lambda s, g: (KNN(seed=s), {}),
    "ThresholdRule": lambda s, g: (ThresholdRule(seed=s), {}),
}


def main():
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    rows = []
    for name, mk in MAKERS.items():
        per_seed_changed, per_seed_rows = [], []
        for s in SEEDS:
            _, _, Xtr, gtr, ytr, Xte, gte = rp.split_and_prepare(ds, seed=s)
            m, kw = mk(s, gtr)
            m.fit(Xtr, ytr, **kw)
            base = np.asarray(m.predict(Xte), int)
            acts_by_level = {}
            for lv in LEVELS:
                Xp = np.array(Xte, dtype=float, copy=True)
                Xp[:, 0] = lv                       # feature 0 is the normalized anomaly score
                a = np.asarray(m.predict(Xp), int)
                acts_by_level[lv] = a
                ev = evaluate_safety(gte, a)
                per_seed_rows.append(dict(model=name, seed=s, level=lv, **{
                    k: ev[k] for k in ("decision_acc", "miss_rate", "escalation",
                                       "under_escalation", "fa_per_clean")}))
            stack = np.stack([acts_by_level[lv] for lv in LEVELS], 0)
            changed = (stack.max(0) != stack.min(0))       # action differs across the sweep
            per_seed_changed.append(float(changed.mean()))
        df = pd.DataFrame(per_seed_rows).drop(columns=["model"])
        agg = df.groupby("level").agg(["mean", "std"])
        for lv in LEVELS:
            rows.append(dict(model=name, level=lv,
                             decision_acc_mean=float(agg.loc[lv, ("decision_acc", "mean")]),
                             decision_acc_std=float(agg.loc[lv, ("decision_acc", "std")]),
                             miss_rate_mean=float(agg.loc[lv, ("miss_rate", "mean")]),
                             escalation_mean=float(agg.loc[lv, ("escalation", "mean")]),
                             fa_per_clean_mean=float(agg.loc[lv, ("fa_per_clean", "mean")]),
                             frac_windows_action_changed=float(np.mean(per_seed_changed)),
                             frac_changed_std=float(np.std(per_seed_changed, ddof=1))))
        print(f"  {name:16s} windows whose action moves anywhere in [0,1]: "
              f"{100*np.mean(per_seed_changed):.2f}% ± {100*np.std(per_seed_changed, ddof=1):.2f}%",
              flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "v2_anomaly_influence.csv"), index=False)
    print("wrote v2_anomaly_influence.csv")


if __name__ == "__main__":
    print("=== E9  anomaly-score influence on the selected action ===", flush=True)
    main()
