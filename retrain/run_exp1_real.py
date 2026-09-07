"""Exp #1 — Reward-Asymmetry Pareto Frontier (REAL dataset).

Trains the project's DuelingDQN Decision Agent on the REAL raw corpus
(6400 rows -> ~6380 windows) via cost-weighted cross-entropy. Each training
row's weight encodes the cost-ratio curve (missed dangerous gas weighs
miss_cost; false alarm weighs false_cost). Same architecture & features as
the paper's RL Decision Agent, trained on the actual data.

Sweep C = miss_cost:false_cost in [2:1 .. 20:1]; anchors:
  FLOOR = plain CE (1:1, cost-insensitive)
  CEIL  = rule oracle (perfect)
Outputs: results/exp1_pareto.csv + results/exp1_frontier.png
"""
import os, sys
import numpy as np
import pandas as pd
import torch
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrain import raw_pipeline as rp
from retrain import rewards as RW
from retrain.comparators import SupervisedDQN, cost_weighted_sample_weight
from retrain.metrics import evaluate_actions, mean_std

ROOT = r"C:\Users\HP\Downloads\Agentic-AI-for-Pipeline-Leak-Detection-main"
OUTDIR = os.path.join(ROOT, "retrain", "results")
RETRAINED_DIR = os.path.join(ROOT, "models", "retrained", "exp1_pareto")
CACHE = os.path.join(OUTDIR, "real_features.csv")
SEEDS = [42, 1337, 7, 2024, 99]
DEVICE = "cpu"
EPOCHS = 80


def get_data():
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    return ds


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ds = get_data()
    ratios = [(c, 1) for c in [2, 4, 6, 8, 10, 12, 16, 20]]
    rows = []

    def sweep(miss_cost, false_cost, name, C):
        acc, miss, fa = [], [], []
        for s in SEEDS:
            # audit A7: split varies per seed (in lock-step with model init)
            _, _, Xtr, gtr, yact_tr, Xte, gte = rp.split_and_prepare(ds, seed=s)
            sw = cost_weighted_sample_weight(gtr, miss_cost, false_cost) if (miss_cost, false_cost) != (1, 1) else None
            net = SupervisedDQN(device=DEVICE, seed=s)
            net.fit(Xtr, yact_tr, sample_weight=sw, epochs=EPOCHS)
            acts = net.predict(Xte)
            m = evaluate_actions(gte, acts)
            acc.append(m["decision_acc"]); miss.append(m["miss_rate"]); fa.append(m["false_alarm_rate"])
            # Persist in DecisionTool-compatible format with the exact scale
            # constants this seed's training split used (raw_pipeline.prepare).
            train_df, _te = rp.block_wise_holdout(ds, test_frac=0.2, gap=rp.WINDOW_SIZE, seed=s)
            _tr, _te2, feat_scaler, (p1, p99) = rp.prepare(train_df, _te)
            scales = dict(
                anom_p1=float(p1), anom_p99=float(p99),
                feat_scaler_mean=np.asarray(feat_scaler.mean_, dtype=np.float32),
                feat_scaler_scale=np.asarray(feat_scaler.scale_, dtype=np.float32),
            )
            ckpt_path = os.path.join(RETRAINED_DIR, f"exp1_miss{miss_cost}_seed{s}.pth")
            net.save(ckpt_path, scales=scales)
        am2, as_ = mean_std(acc); mm, ms = mean_std(miss); fm, fs = mean_std(fa)
        rows.append(dict(config=name, C=C, acc_mean=am2, acc_std=as_,
                         miss_mean=mm, miss_std=ms, fa_mean=fm, fa_std=fs))
        print(f"{name:26s} C={C:6s} acc={am2:.4f} miss={mm:.4f} fa={fm:.4f}")

    sweep(1, 1, "plain_ce_1to1_FLOOR", "1:1")
    for mc, fc in ratios:
        sweep(mc, fc, f"asym_miss{mc}_false{fc}", f"{mc}:{fc}")

    _, _, _, _, _, Xte, gte = rp.split_and_prepare(ds, seed=SEEDS[0])
    oracle = np.array([RW.rule_oracle_action(g) for g in gte])
    ceil = evaluate_actions(gte, oracle)
    rows.append(dict(config="rule_oracle_CEIL", C="inf",
                     acc_mean=ceil["decision_acc"], acc_std=0.0,
                     miss_mean=ceil["miss_rate"], miss_std=0.0,
                     fa_mean=ceil["false_alarm_rate"], fa_std=0.0))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTDIR, "exp1_pareto.csv"), index=False)

    sweep_rows = out[out["config"].str.startswith("asym")]
    plt.figure(figsize=(8, 6))
    plt.scatter(sweep_rows["fa_mean"], sweep_rows["miss_mean"], c=range(len(sweep_rows)),
                cmap="viridis", s=90, zorder=3)
    for _, r in sweep_rows.iterrows():
        plt.annotate(r["C"], (r["fa_mean"], r["miss_mean"]),
                     fontsize=8, xytext=(3, 3), textcoords="offset points")
    f = out[out.config == "plain_ce_1to1_FLOOR"]
    c = out[out.config == "rule_oracle_CEIL"]
    plt.scatter([f.fa_mean.values[0]], [f.miss_mean.values[0]], marker="*", s=240, c="red", label="FLOOR (1:1)")
    plt.scatter([c.fa_mean.values[0]], [c.miss_mean.values[0]], marker="*", s=240, c="green", label="CEIL (oracle)")
    plt.xlabel("False Alarm Rate (mean)"); plt.ylabel("Danger Miss Rate (mean)")
    plt.title("Exp #1 — Reward-Asymmetry Pareto Frontier\n(REAL dataset, DuelingDQN, cost-weighted CE)")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "exp1_frontier.png"), dpi=150)
    print("Saved exp1_pareto.csv + exp1_frontier.png")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
