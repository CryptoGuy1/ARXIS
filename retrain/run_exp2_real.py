"""Exp #2 — Real Near-Boundary Stress Test (REAL dataset).

Isolate whether the fail-safe skew comes from the REWARD (cost weighting),
not the network. Comparison models (5 seeds each):
  A) Decision Agent = DuelingDQN + ASYMMETRIC weighted CE  (proposed)
  B) Unweighted DQN = DuelingDQN + plain CE                 (same net, flat cost)
  D) Supervised MLP (capacity-matched)
  E) Cost-Sensitive GBM

Boundary subset = test rows in the bottom quartile of top1-top2 margin
(uncertain between correct severe action and a milder one). We report
danger-miss on the full set AND the boundary subset for each model.
Key claim: A should push boundary rows to safer actions (lower danger-miss)
than B, proving the cost weighting (reward) drives the skew, not the arch.
"""
import os, sys
import numpy as np
import pandas as pd
import json
import torch
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrain import raw_pipeline as rp
from retrain import rewards as RW
from retrain.comparators import SupervisedDQN, SupervisedMLP, CostSensitiveGBM, cost_weighted_sample_weight
from retrain.metrics import evaluate_actions, mean_std, paired_significance

ROOT = r"C:\Users\HP\Downloads\Agentic-AI-for-Pipeline-Leak-Detection-main"
OUTDIR = os.path.join(ROOT, "retrain", "results")
CACHE = os.path.join(OUTDIR, "real_features.csv")
SEEDS = [42, 1337, 7, 2024, 99]
DEVICE = "cpu"
EPOCHS = 80


def boundary_mask(q):
    top2 = np.sort(q, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    return margin <= np.quantile(margin, 0.25)


def rec(rows, name, acc, miss, fa, bmiss):
    am, as_ = mean_std(acc); mm, ms = mean_std(miss); fm, fs = mean_std(fa)
    bm, bs = mean_std(bmiss)
    rows.append(dict(model=name, acc_mean=am, acc_std=as_, miss_mean=mm, miss_std=ms,
                     fa_mean=fm, fa_std=fs, boundary_miss_mean=bm, boundary_miss_std=bs))


def run():
    os.makedirs(OUTDIR, exist_ok=True)
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)

    rows = []
    # A: asym weighted DQN
    A_acc, A_miss, A_fa, A_bm = [], [], [], []
    for s in SEEDS:
        # audit A7: split varies per seed
        _, _, Xtr, gtr, yact_tr, Xte, gte = rp.split_and_prepare(ds, seed=s)
        sw = cost_weighted_sample_weight(gtr, 10, 1)
        net = SupervisedDQN(device=DEVICE, seed=s); net.fit(Xtr, yact_tr, sample_weight=sw, epochs=EPOCHS)
        q = net.predict_q(Xte); acts = q.argmax(1)
        m = evaluate_actions(gte, acts); A_acc.append(m["decision_acc"]); A_miss.append(m["miss_rate"]); A_fa.append(m["false_alarm_rate"])
        bm = evaluate_actions(gte[boundary_mask(q)], acts[boundary_mask(q)])["miss_rate"]; A_bm.append(bm)
    rec(rows, "A_decision_agent", A_acc, A_miss, A_fa, A_bm)

    # B: plain DQN CE
    B_acc, B_miss, B_fa, B_bm = [], [], [], []
    for s in SEEDS:
        _, _, Xtr, gtr, yact_tr, Xte, gte = rp.split_and_prepare(ds, seed=s)
        net = SupervisedDQN(device=DEVICE, seed=s); net.fit(Xtr, yact_tr, epochs=EPOCHS)
        q = net.predict_q(Xte); acts = q.argmax(1)
        m = evaluate_actions(gte, acts); B_acc.append(m["decision_acc"]); B_miss.append(m["miss_rate"]); B_fa.append(m["false_alarm_rate"])
        bm = evaluate_actions(gte[boundary_mask(q)], acts[boundary_mask(q)])["miss_rate"]; B_bm.append(bm)
    rec(rows, "B_plain_dqn", B_acc, B_miss, B_fa, B_bm)

    # D: MLP
    D_acc, D_miss, D_fa, D_bm = [], [], [], []
    for s in SEEDS:
        _, _, Xtr, gtr, yact_tr, Xte, gte = rp.split_and_prepare(ds, seed=s)
        clf = SupervisedMLP(seed=s); clf.fit(Xtr, yact_tr)
        acts = clf.predict(Xte); m = evaluate_actions(gte, acts)
        D_acc.append(m["decision_acc"]); D_miss.append(m["miss_rate"]); D_fa.append(m["false_alarm_rate"])
        q = clf.predict_proba(Xte); bm = evaluate_actions(gte[boundary_mask(q)], acts[boundary_mask(q)])["miss_rate"]; D_bm.append(bm)
    rec(rows, "D_mlp", D_acc, D_miss, D_fa, D_bm)

    # E: GBM
    E_acc, E_miss, E_fa, E_bm = [], [], [], []
    for s in SEEDS:
        _, _, Xtr, gtr, yact_tr, Xte, gte = rp.split_and_prepare(ds, seed=s)
        clf = CostSensitiveGBM(seed=s); clf.fit(Xtr, yact_tr)
        acts = clf.predict(Xte); m = evaluate_actions(gte, acts)
        E_acc.append(m["decision_acc"]); E_miss.append(m["miss_rate"]); E_fa.append(m["false_alarm_rate"])
        q = clf.predict_proba(Xte); bm = evaluate_actions(gte[boundary_mask(q)], acts[boundary_mask(q)])["miss_rate"]; E_bm.append(bm)
    rec(rows, "E_gbm", E_acc, E_miss, E_fa, E_bm)

    # ---- Paired significance: is a gap REAL, not noise? ----
    # Per-seed full-set danger-miss arrays (A vs B, and A vs E which is the
    # non-degenerate comparison since E actually misses danger).
    print("\n=== Paired significance (Wilcoxon, 5 seeds) ===")
    stat_ab, p_ab = paired_significance(A_miss, B_miss, alternative="less")
    print(f"A(asym) vs B(plain) full-set miss:        W={stat_ab} p={p_ab:.4f} "
          f"(A={np.mean(A_miss):.4f}±{np.std(A_miss,ddof=1):.4f}, "
          f"B={np.mean(B_miss):.4f}±{np.std(B_miss,ddof=1):.4f})")
    stat_ae, p_ae = paired_significance(A_miss, E_miss, alternative="less")
    print(f"A(asym) vs E(GBM) full-set miss:          W={stat_ae} p={p_ae:.4f} "
          f"(A={np.mean(A_miss):.4f}±{np.std(A_miss,ddof=1):.4f}, "
          f"E={np.mean(E_miss):.4f}±{np.std(E_miss,ddof=1):.4f})")
    stat_ab_b, p_ab_b = paired_significance(A_bm, B_bm, alternative="less")
    print(f"A(asym) vs B(plain) boundary-subset miss: W={stat_ab_b} p={p_ab_b:.4f} "
          f"(A={np.mean(A_bm):.4f}±{np.std(A_bm,ddof=1):.4f}, "
          f"B={np.mean(B_bm):.4f}±{np.std(B_bm,ddof=1):.4f})")
    stat_ae_b, p_ae_b = paired_significance(A_bm, E_bm, alternative="less")
    print(f"A(asym) vs E(GBM) boundary-subset miss:   W={stat_ae_b} p={p_ae_b:.4f} "
          f"(A={np.mean(A_bm):.4f}±{np.std(A_bm,ddof=1):.4f}, "
          f"E={np.mean(E_bm):.4f}±{np.std(E_bm,ddof=1):.4f})")
    sig = dict(
        A_vs_B_full_miss_W=stat_ab, A_vs_B_full_miss_p=p_ab,
        A_vs_E_full_miss_W=stat_ae, A_vs_E_full_miss_p=p_ae,
        A_vs_B_boundary_miss_W=stat_ab_b, A_vs_B_boundary_miss_p=p_ab_b,
        A_vs_E_boundary_miss_W=stat_ae_b, A_vs_E_boundary_miss_p=p_ae_b,
    )
    with open(os.path.join(OUTDIR, "exp2_significance.json"), "w") as f:
        json.dump(sig, f, indent=2)
    print("Saved exp2_significance.json")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTDIR, "exp2_boundary.csv"), index=False)
    print("Saved exp2_boundary.csv")
    print(out.to_string(index=False))


if __name__ == "__main__":
    run()
