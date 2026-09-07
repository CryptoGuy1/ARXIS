"""Exp ZOO — broader model comparison ("is this just a fancy MLP?").

Evaluates SVM, RandomForest, RawWindow-LSTM, and offline CQL against the
already-built comparators (A=Decision Agent [asymmetric cost-weighted DQN],
B=plain DQN, D=MLP, E=cost-sensitive GBM) on the REAL raw corpus. Same 5-seed
protocol + metrics + paired significance as Exp #2.

Outputs: retrain/results/expzoo.csv + retrain/results/expzoo_significance.json
"""
import os, sys, json, inspect
import numpy as np
import torch
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import raw_pipeline as rp
import rewards as RW
from retrain.comparators import SupervisedDQN, SupervisedMLP, CostSensitiveGBM
from retrain.comparators import cost_weighted_sample_weight
from retrain.zoo import SVMClassifier, RandomForest, RawWindowLSTM, CQLAgent
from retrain.metrics import evaluate_actions, mean_std, paired_significance

ROOT = r"C:\Users\HP\Downloads\Agentic-AI-for-Pipeline-Leak-Detection-main"
OUTDIR = os.path.join(ROOT, "retrain", "results")
CACHE = os.path.join(OUTDIR, "real_features.csv")
SEEDS = [42, 1337, 7, 2024, 99]
DEVICE = "cpu"
EPOCHS = 80


def get_data():
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    return ds


def evaluate_model(make, fit_kwargs, ds, name):
    acc, miss, fa = [], [], []
    for s in SEEDS:
        # audit A7: split varies per seed
        _, _, Xtr, gtr, yact_tr, Xte, gte = rp.split_and_prepare(ds, seed=s)
        m = make(seed=s)
        kw = dict(fit_kwargs(s))
        # Audit C3: the temporal LSTM must know which rows belong to the same
        # gas run, or its K-row windows straddle class boundaries. Models that
        # do not accept `groups` are unaffected.
        if "groups" in inspect.signature(m.fit).parameters:
            kw.setdefault("groups", gtr)
        m.fit(Xtr, yact_tr, **kw)
        if "groups" in inspect.signature(m.predict).parameters:
            acts = m.predict(Xte, groups=gte)
        else:
            acts = m.predict(Xte)
        ev = evaluate_actions(gte, acts)
        acc.append(ev["decision_acc"]); miss.append(ev["miss_rate"]); fa.append(ev["false_alarm_rate"])
    am2, as_ = mean_std(acc); mm, ms = mean_std(miss); fm, fs = mean_std(fa)
    print(f"  {name:18s} acc={am2:.4f}±{as_:.4f}  miss={mm:.4f}  fa={fm:.4f}")
    return dict(model=name, acc_mean=am2, acc_std=as_, miss_mean=mm, miss_std=ms, fa_mean=fm, fa_std=fs), acc, miss, fa


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ds = get_data()
    # A uses asymmetric cost weights; B uses none (plain). CQL trains 80 epochs.
    # gtr is needed only to build the fixed cost-weight vector for model A.
    _, _, _, gtr, _, _, _ = rp.split_and_prepare(ds, seed=SEEDS[0])
    sw = cost_weighted_sample_weight(gtr, 8, 1)  # mid ratio; A is the asymmetric agent

    store, seed = {}, {}
    store["A_decision_agent"], a_acc, a_miss, a_fa = evaluate_model(
        lambda seed: SupervisedDQN(device=DEVICE, seed=seed),
        lambda s: dict(sample_weight=sw, epochs=EPOCHS), ds, "A_decision_agent")
    store["B_plain_dqn"], b_acc, b_miss, b_fa = evaluate_model(
        lambda seed: SupervisedDQN(device=DEVICE, seed=seed),
        lambda s: dict(sample_weight=None, epochs=EPOCHS), ds, "B_plain_dqn")
    store["D_mlp"], d_acc, d_miss, d_fa = evaluate_model(
        lambda seed: SupervisedMLP(seed=seed), lambda s: dict(), ds, "D_mlp")
    store["E_gbm"], e_acc, e_miss, e_fa = evaluate_model(
        lambda seed: CostSensitiveGBM(seed=seed), lambda s: dict(), ds, "E_gbm")
    store["SVM"], s_acc, s_miss, s_fa = evaluate_model(
        lambda seed: SVMClassifier(seed=seed), lambda s: dict(), ds, "SVM")
    store["RF"], rf_acc, rf_miss, rf_fa = evaluate_model(
        lambda seed: RandomForest(seed=seed), lambda s: dict(), ds, "RF")
    store["LSTM"], l_acc, l_miss, l_fa = evaluate_model(
        lambda seed: RawWindowLSTM(device=DEVICE, seed=seed, K=10), lambda s: dict(epochs=40), ds, "LSTM")
    store["CQL"], c_acc, c_miss, c_fa = evaluate_model(
        lambda seed: CQLAgent(device=DEVICE, seed=seed, alpha=1.0), lambda s: dict(epochs=EPOCHS, g_class=gtr), ds, "CQL")

    rows = list(store.values())

    # significance vs Decision Agent (A) on seed-level arrays
    sig = {}
    for other, (o_acc, o_miss) in {"B_plain_dqn":(b_acc,b_miss),"D_mlp":(d_acc,d_miss),
                                     "E_gbm":(e_acc,e_miss),"SVM":(s_acc,s_miss),"RF":(rf_acc,rf_miss),
                                     "LSTM":(l_acc,l_miss),"CQL":(c_acc,c_miss)}.items():
        p_acc = paired_significance(a_acc, o_acc, alternative="two-sided")
        p_miss = paired_significance(a_miss, o_miss, alternative="two-sided")
        sig[other] = dict(acc_p=p_acc, miss_p=p_miss)
        print(f"  A vs {other}: acc_p={p_acc}, miss_p={p_miss}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTDIR, "expzoo.csv"), index=False)
    json.dump(sig, open(os.path.join(OUTDIR, "expzoo_significance.json"), "w"), indent=2)
    print("\nSaved expzoo.csv + expzoo_significance.json")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
