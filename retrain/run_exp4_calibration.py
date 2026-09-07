"""Exp #4 — Confidence-Calibrated Escalation.

Report Expected Calibration Error (ECE) + decision accuracy for four
confidence estimators, each across 5 seeds:
  - raw_softmax      : uncalibrated argmax-Q, softmax(Q) confidence (BASELINE)
  - mc_dropout_20     : stochastic forward passes on the trained net
  - temp_scaling      : post-hoc temperature on the raw logits (cheap fix)
  - deep_ensemble_5   : 5 independently trained nets, mean logits

"correct" for ECE = (predicted action in CORRECT_ACTIONS[gas_id]).
Significance: paired Wilcoxon (deep_ensemble vs raw) on ECE across seeds.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from retrain import rewards as RW
from retrain.agent_rl import RLAgent, DuelingDQN
from retrain.comparators import SupervisedDQN, cost_weighted_sample_weight
from retrain.calibration import DeepEnsemble, TemperatureScaling, softmax
from retrain.metrics import evaluate_actions, expected_calibration_error, paired_significance, mean_std

CSV = os.path.join(ROOT, "data", "Gas_Sensors_Measurements.csv")  # A9: single real corpus
OUT = os.path.join(ROOT, "retrain", "results", "exp4_calibration.csv")
CACHE = os.path.join(ROOT, "retrain", "results", "real_features.csv")  # reuse the shared features
SEEDS = [42, 1337, 7, 2024, 99]
EPOCHS = 80          # Decision Agent training epochs (matches Exp #1/#2)
CALIB_FRAC = 0.20   # share of TRAIN held out to fit the temperature (audit C7)
DEVICE = "cpu"

CORRECT = {0: [0], 1: [3], 2: [4], 3: [1, 2]}


def correct_flag(ygas, yact):
    return np.array([int(a) in CORRECT.get(int(g), []) for g, a in zip(ygas, yact)])


@torch.no_grad()
def mc_logits(net, X, n=20):
    net.train()
    t = torch.from_numpy(X).float().to(DEVICE)
    qs = torch.stack([net(t) for _ in range(n)], 0)
    net.eval()
    return qs.mean(0).cpu().numpy()


def run():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # A9: use the SAME real-corpus pipeline as Exp #1/#2/#3/ZOO
    from retrain import raw_pipeline as rp
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)

    raw_ece, raw_acc = [], []
    mc_ece, mc_acc = [], []
    ts_ece, ts_acc = [], []
    ens_ece, ens_acc = [], []

    for s in SEEDS:
        # audit A7: split varies per seed (in lock-step with the Decision Agent init)
        train_df, test_df, Xtr, gtr, yact_tr, Xte, gte = rp.split_and_prepare(ds, seed=s)
        yte_correct = correct_flag(gte, np.array([RW.rule_oracle_action(g) for g in gte]))
        # Decision Agent = DuelingDQN trained with cost-weighted CE (same method
        # as Exp #1/A). This is fast and consistent with the other experiments;
        # the full RL loop (RLAgent, 200 episodes) is the paper's RL path and is
        # not required for a valid ECE calibration comparison.
        # Audit C7: temperature scaling must be fitted on data the network did
        # NOT train on. Carve a stratified calibration split out of TRAIN (the
        # test split stays untouched), fit the net on the remainder, and fit T
        # on the held-out calibration logits.
        rng = np.random.default_rng(s)
        calib_mask = np.zeros(len(Xtr), dtype=bool)
        for g in np.unique(gtr):                      # stratify by gas class
            idx = np.flatnonzero(gtr == g)
            n_cal = max(1, int(round(CALIB_FRAC * len(idx))))
            calib_mask[rng.choice(idx, size=n_cal, replace=False)] = True
        Xfit, yfit, gfit = Xtr[~calib_mask], yact_tr[~calib_mask], gtr[~calib_mask]
        Xcal, ycal = Xtr[calib_mask], yact_tr[calib_mask]

        sw = cost_weighted_sample_weight(gfit, miss_cost=10.0, false_cost=1.0)
        sdq = SupervisedDQN(device=DEVICE, seed=s)
        sdq.fit(Xfit, yfit, sample_weight=sw, epochs=EPOCHS)
        net = sdq.net
        q_raw = sdq.predict_q(Xte)
        q_mc = mc_logits(net, Xte, n=20)

        # raw
        p_raw = softmax(q_raw)
        raw_ece.append(expected_calibration_error(p_raw, yte_correct))
        raw_acc.append(evaluate_actions(gte, q_raw.argmax(1))["decision_acc"])

        # mc dropout
        p_mc = softmax(q_mc)
        mc_ece.append(expected_calibration_error(p_mc, yte_correct))
        mc_acc.append(evaluate_actions(gte, q_mc.argmax(1))["decision_acc"])

        # temperature scaling: T fitted on the HELD-OUT calibration split (C7)
        q_cal_raw = sdq.predict_q(Xcal)
        ts = TemperatureScaling(device=DEVICE)
        ts.fit(q_cal_raw, ycal)
        q_ts = ts.scale(q_raw)
        p_ts = softmax(q_ts)
        ts_ece.append(expected_calibration_error(p_ts, yte_correct))
        ts_acc.append(evaluate_actions(gte, q_ts.argmax(1))["decision_acc"])

        # deep ensemble (5 independent supervised nets)
        # Audit C5/C6: same fit split AND same epoch budget as the single model,
        # so "ensemble vs single" measures ensembling and not a budget gap.
        ens = DeepEnsemble(n=5, device=DEVICE, base_seed=s)
        ens.fit(Xfit, yfit, epochs=EPOCHS)
        q_ens = ens.predict_q(Xte)
        p_ens = softmax(q_ens)
        ens_ece.append(expected_calibration_error(p_ens, yte_correct))
        ens_acc.append(evaluate_actions(gte, q_ens.argmax(1))["decision_acc"])

        print(f"seed {s}: raw_ece={raw_ece[-1]:.3f} mc={mc_ece[-1]:.3f} ts={ts_ece[-1]:.3f} ens={ens_ece[-1]:.3f}")

    def row(name, ece, acc):
        em, es = mean_std(ece); am, as_ = mean_std(acc)
        return dict(estimator=name, ece_mean=em, ece_std=es, acc_mean=am, acc_std=as_)

    rows = [row("raw_softmax", raw_ece, raw_acc),
            row("mc_dropout_20", mc_ece, mc_acc),
            row("temp_scaling", ts_ece, ts_acc),
            row("deep_ensemble_5", ens_ece, ens_acc)]

    # Significance on ECE (lower is better, so 'greater' tests raw > other).
    # The headline comparison is raw vs temperature scaling; the ensemble
    # comparison is reported alongside it rather than instead of it.
    sig = {}
    for label, other in (("temp_scaling", ts_ece), ("deep_ensemble_5", ens_ece),
                         ("mc_dropout_20", mc_ece)):
        st, pv = paired_significance(raw_ece, other, alternative="greater")
        wins = int(sum(o < r for r, o in zip(raw_ece, other)))
        sig[label] = dict(W=st, p=pv, better_than_raw_on_seeds=f"{wins}/{len(SEEDS)}")
        print(f"  raw vs {label:16s}: W={st} p={pv:.5f}  better on {wins}/{len(SEEDS)} seeds")
    print(f"  (n={len(SEEDS)} seeds -> minimum attainable one-sided p = "
          f"{0.5 ** len(SEEDS):.5f}; two-sided = {2 * 0.5 ** len(SEEDS):.5f})")
    json.dump(sig, open(os.path.join(os.path.dirname(OUT),
                                     "exp4_significance.json"), "w"), indent=2)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print("Saved:", OUT)
    print(out.to_string(index=False))


if __name__ == "__main__":
    run()
