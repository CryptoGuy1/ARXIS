"""Generate the two missing figures from real CSV results:
  - retrain/results/exp3_loco.png  (Exp #3 LOCO: miss_mean by held-out class)
  - retrain/results/expzoo.png     (Exp ZOO: acc_mean +- std across 8 models)
Then prints the markdown embed lines. All data comes from the real CSVs.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\HP\Downloads\Agentic-AI-for-Pipeline-Leak-Detection-main"
R = os.path.join(ROOT, "retrain", "results")

# ---------------- Exp #3 LOCO ----------------
df3 = pd.read_csv(os.path.join(R, "exp3_loco.csv"))
# only danger classes have a meaningful miss story
danger = df3[df3["held_out"].isin(["Smoke", "Mixture"])]
classes = ["Smoke", "Mixture"]
models = ["A_decision_agent", "B_plain_dqn", "D_mlp", "E_gbm"]
labels = ["A (asym)", "B (plain)", "D (MLP)", "E (GBM)"]
x = np.arange(len(classes)); w = 0.2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
for i, m in enumerate(models):
    vals = [danger[(danger.held_out == c) & (danger.model == m)]["miss_mean"].values[0] for c in classes]
    ax1.bar(x + (i - 1.5) * w, vals, w, label=labels[i])
    escs = [danger[(danger.held_out == c) & (danger.model == m)]["escalation_mean"].values[0] for c in classes]
    ax2.bar(x + (i - 1.5) * w, escs, w, label=labels[i])
ax1.set_xticks(x); ax1.set_xticklabels(classes); ax1.set_ylabel("danger-miss rate (lower=better)")
ax1.set_title("Exp #3 LOCO — zero-shot danger-miss (primary metric)")
ax1.set_ylim(0, 0.004); ax1.legend(fontsize=8)
ax2.set_xticks(x); ax2.set_xticklabels(classes); ax2.set_ylabel("escalation rate (action>=3)")
ax2.set_title("Exp #3 LOCO — escalation rate (secondary)")
ax2.set_ylim(0, 1.05); ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(R, "exp3_loco.png"), dpi=130); plt.close(fig)

# ---------------- Exp ZOO ----------------
dfz = pd.read_csv(os.path.join(R, "expzoo.csv"))
order = ["A_decision_agent", "D_mlp", "LSTM (raw-window)", "B_plain_dqn", "CQL (offline safe-RL)", "SVM", "RF", "E_gbm"]
# zcsv model names: A_decision_agent, D_mlp, LSTM, B_plain_dqn, CQL, SVM, RF, E_gbm
zmap = {"A_decision_agent": "A (agent)", "D_mlp": "D (MLP)", "LSTM": "LSTM", "B_plain_dqn": "B (plain)",
        "CQL": "CQL", "SVM": "SVM", "RF": "RF", "E_gbm": "E (GBM)"}
dfz["label"] = dfz["model"].map(zmap)
dfz = dfz.sort_values("acc_mean", ascending=False)
fig, ax = plt.subplots(figsize=(9, 4.5))
colors = ["#c0392b" if m == "A (agent)" else "#2c3e50" for m in dfz["label"]]
ax.bar(dfz["label"], dfz["acc_mean"], yerr=dfz["acc_std"], capsize=4, color=colors)
ax.set_ylabel("accuracy (mean ± std, 5 seeds)")
ax.set_title("Exp ZOO — broader model zoo vs Decision Agent (A highlighted)")
ax.set_ylim(0.8, 1.0)
for i, (v, s) in enumerate(zip(dfz["acc_mean"], dfz["acc_std"])):
    ax.text(i, v + s + 0.003, f"{v:.3f}", ha="center", fontsize=8)
plt.xticks(rotation=20, ha="right"); fig.tight_layout()
fig.savefig(os.path.join(R, "expzoo.png"), dpi=130); plt.close(fig)

print("WROTE exp3_loco.png + expzoo.png")
print("embed:\n![Exp3 LOCO](exp3_loco.png)\n![Exp ZOO](expzoo.png)")
