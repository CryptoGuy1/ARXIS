"""Regenerate the paper figures from the v2 results.

Palette validated with the dataviz validator (light surface #fcfcfb):
  lightness band PASS, chroma floor PASS, CVD separation WARN (green/orange
  dE 7.0, inside the 6-8 floor band) -> legal only with secondary encoding,
  so every categorical mark also carries a direct value label.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "retrain", "results_v2")
FIG = os.path.join(ROOT, "figures_v2")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, GREEN, PURPLE, OLIVE = "#2f6f9f", "#c8642f", "#1f8f5f", "#8f4f9f", "#8a7a1a"
CAT = [BLUE, ORANGE, GREEN, PURPLE, OLIVE]
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#1a1a1a", "#4a4a4a", "#8a8a8a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.edgecolor": "#cccccc", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.7,
    "legend.frameon": False, "figure.dpi": 300, "savefig.bbox": "tight",
})

PRETTY = {
    "A_cost_weighted": "Cost-weighted", "Ordinal": "Ordinal cost", "B_unweighted": "Unweighted",
    "D_mlp": "MLP", "E_gbm": "GBM", "SVM": "SVM", "RF": "Random forest",
    "LSTM": "LSTM", "CQL": "CQL", "KNN": "k-NN",
    "ThresholdRule": "Shallow tree", "CUSUM": "CUSUM",
}


def _label_bars(ax, bars, vals, fmt="{:.3f}", dy=0.012, rot=0, fs=7.2):
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy * span,
                fmt.format(v), ha="center", va="bottom", fontsize=fs,
                color=INK2, rotation=rot)


# ---------------------------------------------------------------- LOCO
def fig_loco():
    df = pd.read_csv(os.path.join(RES, "v3_loco_all.csv"))
    classes = ["Smoke", "Mixture"]
    models = [m for m in PRETTY if m in set(df.model)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.78))

    # (a) miss rate, LOG scale -- the linear axis in the original clipped an
    # 18.7% bar to look identical to a 0.4% one.
    ax = axes[0]
    w = 0.38
    floor = 5e-5
    for j, cls in enumerate(classes):
        sub = df[df.held_out == cls].set_index("model")
        vals = [max(float(sub.loc[m, "miss_rate_mean"]), 0.0) if m in sub.index else np.nan
                for m in models]
        x = np.arange(len(models)) + (j - 0.5) * w
        plotted = [v if v > 0 else floor for v in vals]
        b = ax.bar(x, plotted, w * 0.92, color=CAT[j], label=cls,
                   edgecolor=SURFACE, linewidth=1.2, zorder=3)
        for bi, v in zip(b, vals):
            # Nudge the two labels of a pair apart: with eleven models the
            # columns are narrow enough that "1.16%" and "2.54%" collide.
            ax.text(bi.get_x() + bi.get_width() / 2, bi.get_height() * 1.35,
                    ("0" if v == 0 else f"{100*v:.2f}%"), ha="center", va="bottom",
                    fontsize=8, color=INK2, rotation=90 if v > 0 else 0)
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.7, 0.6)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_yticklabels(["0.01%", "0.1%", "1%", "10%"])
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([PRETTY[m] for m in models], rotation=28, ha="right")
    ax.set_ylabel("Missed-hazard rate (log)")
    ax.set_title("(a)  Missed-hazard rate on the unseen class", loc="left")
    ax.grid(axis="y", zorder=0)
    ax.legend(title="Held-out class", ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), fontsize=9.2, title_fontsize=9.9)

    # (b) escalation adequacy
    ax = axes[1]
    for j, cls in enumerate(classes):
        sub = df[df.held_out == cls].set_index("model")
        vals = [float(sub.loc[m, "escalation_mean"]) if m in sub.index else np.nan for m in models]
        err = [float(sub.loc[m, "escalation_std"]) if m in sub.index else 0 for m in models]
        x = np.arange(len(models)) + (j - 0.5) * w
        b = ax.bar(x, vals, w * 0.92, color=CAT[j], label=cls,
                   edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.errorbar(x, vals, yerr=err, fmt="none", ecolor=INK2, elinewidth=0.9,
                    capsize=2.2, zorder=4)
        for bi, v, e in zip(b, vals, err):
            ax.text(bi.get_x() + bi.get_width() / 2, bi.get_height() + e + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2,
                    rotation=90)
    ax.set_ylim(0, 1.30)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([PRETTY[m] for m in models], rotation=28, ha="right")
    ax.set_ylabel("Escalation to alarm-grade action (a ≥ 3)")
    ax.set_title("(b)  Escalation adequacy on the unseen class", loc="left")
    ax.grid(axis="y", zorder=0)
    ax.legend(title="Held-out class", ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), fontsize=9.2, title_fontsize=9.9)

    fig.text(0.005, -0.10, "Bars drawn at the floor of panel (a) are exactly zero. "
             "Ten of eleven models bound their missed-hazard rate on unseen Mixture at or below 3.3%, "
             "yet escalate between 0.5% and 100% of the same windows.",
             ha="left", fontsize=8.4, color=MUTED, style="italic")
    fig.savefig(os.path.join(FIG, "fig_loco.png"))
    plt.close(fig)
    print("wrote fig_loco.png")


# ------------------------------------------------------------ cost sweep
def fig_costsweep():
    df = pd.read_csv(os.path.join(RES, "v2_costsweep.csv"))
    df = df[df.cost_ratio != "label_function"].copy()
    df["C"] = [int(s.split(":")[0]) for s in df.cost_ratio]
    df = df.sort_values("C")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(df.C, df.decision_acc_mean, "-o", color=BLUE, lw=2, ms=7,
            label="Decision accuracy", zorder=3)
    ax.fill_between(df.C, df.decision_acc_mean - df.decision_acc_std,
                    df.decision_acc_mean + df.decision_acc_std,
                    color=BLUE, alpha=0.16, lw=0, zorder=2)
    ax.plot(df.C, df.escalation_mean, "-s", color=GREEN, lw=2, ms=6.5,
            label="Escalation adequacy", zorder=3)
    ax.plot(df.C, df.miss_rate_mean, "-^", color=ORANGE, lw=2, ms=6.5,
            label="Missed-hazard rate", zorder=3)
    best = df.loc[df.decision_acc_mean.idxmax()]
    ax.annotate(f"accuracy peak\nC = {int(best.C)}:1", (best.C, best.decision_acc_mean),
                textcoords="offset points", xytext=(4, 20), fontsize=8.6, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    dep = df[df.C == 8].iloc[0]
    ax.annotate("deployed\nC = 8:1", (8, dep.decision_acc_mean),
                textcoords="offset points", xytext=(6, -32), fontsize=8.6, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.text(df.C.iloc[-1], df.miss_rate_mean.iloc[-1] + 0.03,
            "missed-hazard rate is flat at this level for every ratio",
            ha="right", fontsize=8.2, color=MUTED, style="italic")
    ax.set_xscale("log"); ax.set_xticks(df.C.tolist())
    ax.set_xticklabels([f"{c}:1" for c in df.C], fontsize=9.2)
    ax.set_ylim(-0.04, 1.12)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Cost ratio  C = c_miss : c_false")
    ax.set_ylabel("Rate")
    ax.set_title("Cost asymmetry changes accuracy, not safety behavior", loc="left")
    ax.grid(axis="y", zorder=0); ax.legend(loc="center left", fontsize=9.3)
    fig.savefig(os.path.join(FIG, "fig_costsweep.png"))
    plt.close(fig)
    print("wrote fig_costsweep.png")


# ------------------------------------------------------------------ zoo
def fig_zoo():
    df = pd.read_csv(os.path.join(RES, "v2_zoo.csv"))
    df = df[df.model != "CUSUM"]           # two-class; accuracy not comparable
    df = df.sort_values("decision_acc_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8.4, 4))
    names = [PRETTY.get(m, m) for m in df.model]
    cols = [ORANGE if m == "A_cost_weighted" else "#9fb4c4" for m in df.model]
    x = np.arange(len(df))
    b = ax.bar(x, df.decision_acc_mean, 0.66, color=cols,
               edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.errorbar(x, df.decision_acc_mean, yerr=df.decision_acc_std, fmt="none",
                ecolor=INK2, elinewidth=1.0, capsize=3, zorder=4)
    for bi, v, sd in zip(b, df.decision_acc_mean, df.decision_acc_std):
        ax.text(bi.get_x() + bi.get_width() / 2, v + sd + 0.007, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, color=INK2)
    lo = float(df.decision_acc_mean.min()); hi = float(df.decision_acc_mean.max())
    ax.axhspan(lo, hi, color=MUTED, alpha=0.10, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=24, ha="right")
    ax.set_ylim(min(0.80, lo - 0.06), 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Decision accuracy")
    ax.set_title(f"All models lie within {100*(hi-lo):.1f} accuracy points — "
                 "this design cannot rank them", loc="left")
    ax.grid(axis="y", zorder=0)
    ax.text(0.995, 0.04, "cost-weighted policy highlighted", transform=ax.transAxes,
            ha="right", fontsize=8, color=MUTED, style="italic")
    fig.savefig(os.path.join(FIG, "fig_zoo.png"))
    plt.close(fig)
    print("wrote fig_zoo.png")


# ------------------------------------------------------------ perturbation
def fig_perturb():
    df = pd.read_csv(os.path.join(RES, "v2_perturb.csv"))
    kinds = [("noise", "Additive noise (σ)"), ("drift", "Calibration drift (±)"),
             ("dropout", "Channel dropout (k sensors)")]
    metrics = [("decision_acc_mean", "Decision accuracy"),
               ("escalation_mean", "Escalation adequacy"),
               ("fa_per_clean_mean", "High-severity false-alarm rate")]
    models = [m for m in ["A_cost_weighted", "D_mlp", "E_gbm", "RF", "ThresholdRule"]
              if m in set(df.model)]          # 5 series: hues assigned, never cycled
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 8.53), sharex="col")
    for r, (mk, mlabel) in enumerate(metrics):
        for c, (kind, klabel) in enumerate(kinds):
            ax = axes[r, c]
            for i, m in enumerate(models):
                sub = df[(df.model == m) & (df.perturbation.isin([kind, "clean"]))]
                sub = sub.sort_values("level")
                ax.plot(sub.level, sub[mk], "-o", ms=4.2, lw=1.8,
                        color=CAT[i % len(CAT)], label=PRETTY[m], zorder=3)
            ax.grid(axis="y", zorder=0)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_ylim(-0.05, 1.08)   # one shared scale per row
            if r == 0:
                ax.set_title(klabel, loc="left", fontsize=10.8)
            if r == 2:
                ax.set_xlabel(klabel)
            if c == 0:
                ax.set_ylabel(mlabel)
    axes[2, 1].text(0.5, 0.90, "peak false-alarm rate under drift is 0.06%",
                    transform=axes[2, 1].transAxes, ha="center", fontsize=8.6,
                    color=MUTED, style="italic")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.955), fontsize=9.7)
    fig.suptitle("Degradation under perturbation: accuracy, escalation and false alarms "
                 "must be read together", x=0.005, y=0.995, ha="left", fontsize=12.4)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(os.path.join(FIG, "fig_perturb.png"))
    plt.close(fig)
    print("wrote fig_perturb.png")


# --------------------------------------------------------------- ablation
def fig_ablation():
    df = pd.read_csv(os.path.join(RES, "v2_ablation.csv")).sort_values("decision_acc_mean")
    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    y = np.arange(len(df))
    b = ax.barh(y, df.decision_acc_mean, 0.62, xerr=df.decision_acc_std,
                color=[ORANGE if s == "full_22" else "#9fb4c4" for s in df.state],
                error_kw=dict(ecolor=INK2, elinewidth=1.0, capsize=3),
                edgecolor=SURFACE, linewidth=1.2, zorder=3)
    for bi, v, sd, d in zip(b, df.decision_acc_mean, df.decision_acc_std, df.delta_pp_vs_full):
        ax.text(1.045, bi.get_y() + bi.get_height() / 2,
                f"{v:.3f}" + ("  (reference)" if abs(d) < 1e-9 else f"   {d:+.2f} pp"),
                va="center", ha="left", fontsize=8.4, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("_", " ") for s in df.state])
    ax.set_xlim(0, 1.42)
    ax.set_xticks([0, .2, .4, .6, .8, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Decision accuracy")
    ax.set_title("Feature ablation moves accuracy; missed-hazard rate (0) and\n"
                 "escalation adequacy (1.00) are unchanged in every condition", loc="left")
    ax.grid(axis="x", zorder=0)
    fig.savefig(os.path.join(FIG, "fig_ablation.png"))
    plt.close(fig)
    print("wrote fig_ablation.png")


# ---------------------------------------------------------------- anomaly
def fig_anomaly():
    a = json.load(open(os.path.join(RES, "v2_anomaly.json")))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4))
    ax = axes[0]
    cls = ["NoGas", "Perfume", "Mixture", "Smoke"]
    vals = [a["per_class_mean_recon"][c] for c in cls]
    b = ax.bar(np.arange(4), vals, 0.6, color=[BLUE, OLIVE, PURPLE, ORANGE],
               edgecolor=SURFACE, linewidth=1.2, zorder=3)
    for bi, v in zip(b, vals):
        ax.text(bi.get_x() + bi.get_width() / 2, v * 1.15, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.2, color=INK2)
    ax.set_yscale("log"); ax.set_xticks(np.arange(4)); ax.set_xticklabels(cls)
    ax.set_ylabel("Mean reconstruction error (log)")
    ax.set_title("(a)  Anomaly score by class", loc="left")
    ax.grid(axis="y", zorder=0)
    ax.set_ylim(0.3, 900)
    ax.text(0.02, 0.90, "hazardous classes score highest, but not in\nseverity order: Mixture carries the higher\ntarget severity yet scores below Smoke",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=MUTED, style="italic")

    ax = axes[1]
    labels = ["In-sample\n(as published)", "Held-out\n(this work)"]
    auc = [a["in_sample"]["auc"], a["auc_mean"]]
    tpr = [a["in_sample"]["tpr"], a["tpr_mean"]]
    fpr = [a["in_sample"]["fpr"], a["fpr_mean"]]
    x = np.arange(2); w = 0.26
    for i, (v, lab, col) in enumerate([(auc, "ROC-AUC", BLUE), (tpr, "TPR", GREEN), (fpr, "FPR", ORANGE)]):
        bb = ax.bar(x + (i - 1) * w, v, w * 0.9, color=col, label=lab,
                    edgecolor=SURFACE, linewidth=1.2, zorder=3)
        for bi, vv in zip(bb, v):
            ax.text(bi.get_x() + bi.get_width() / 2, vv + 0.018, f"{vv:.3f}",
                    ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.30); ax.set_ylabel("Rate")
    ax.set_title("(b)  Threshold fitted on its own sample vs. held out", loc="left")
    ax.grid(axis="y", zorder=0)
    ax.legend(fontsize=9.1, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_anomaly.png"))
    plt.close(fig)
    print("wrote fig_anomaly.png")




# ======================= additional figures (v3) =======================
CRIT, WARN_, GOOD = "#a83226", "#bf8a12", "#1f7a52"   # status palette, validated


def fig_protocol():
    """Schematic of the corpus layout and the block-wise holdout."""
    fig, ax = plt.subplots(figsize=(10.4, 3.2))
    classes = ["NoGas", "Smoke", "Mixture", "Perfume"]
    n = 1581
    for i, c in enumerate(classes):
        y = 3 - i
        ax.add_patch(plt.Rectangle((0, y - 0.32), n, 0.64, color="#dfe6ec",
                                   ec=SURFACE, lw=1.5, zorder=2))
        start = int(0.42 * n)
        ax.add_patch(plt.Rectangle((start - 20, y - 0.32), 20, 0.64, color=WARN_,
                                   ec=SURFACE, lw=1.0, zorder=3))
        ax.add_patch(plt.Rectangle((start, y - 0.32), int(0.20 * n), 0.64, color=BLUE,
                                   ec=SURFACE, lw=1.0, zorder=3))
        ax.add_patch(plt.Rectangle((start + int(0.20 * n), y - 0.32), 20, 0.64, color=WARN_,
                                   ec=SURFACE, lw=1.0, zorder=3))
        ax.text(-30, y, c, ha="right", va="center", fontsize=10.3, color=INK)
    ax.annotate("held-out block\n(20%, position drawn from the run seed)",
                (int(0.52 * n), 3.42), ha="center", fontsize=8.6, color=INK2)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#dfe6ec", label="training windows"),
                       Patch(color=WARN_, label="20-window embargo"),
                       Patch(color=BLUE, label="held-out test windows")],
              ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.52), fontsize=9.2)
    ax.set_xlim(-300, n + 30); ax.set_ylim(-0.55, 4.15)
    ax.set_xticks([0, 400, 800, 1200, 1581])
    ax.set_xlabel("window index within each class block (1,581 windows per class)")
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    ax.set_title("Corpus layout and the leakage-controlled split", loc="left")
    fig.savefig(os.path.join(FIG, "fig_protocol.png")); plt.close(fig)
    print("wrote fig_protocol.png")


def fig_dissociation():
    """In-distribution accuracy against out-of-distribution safety."""
    # In-distribution accuracy is the 30-run figure reported in Table 6, so the
    # scatter matches the table a reader is comparing it against. The ordinal
    # cost objective is left out: it was built to trade accuracy for escalation,
    # so including it would answer the question by construction.
    z = pd.read_csv(os.path.join(RES, "v3_powered.csv")).set_index("model")
    z = z.drop(index="Ordinal", errors="ignore")
    l = pd.read_csv(os.path.join(RES, "v3_loco_all.csv"))
    # Several comparators sit within a tenth of an accuracy point of one
    # another, so written-out names collide however they are offset. Points
    # carry an index instead and the key is printed once, under both panels.
    KEYED = [m for m in PRETTY if m in set(z.index)]
    KEY_OF = {m: i + 1 for i, m in enumerate(KEYED)}
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9))
    for ax, (cls, ycol, ylab, logy) in zip(
            axes, [("Smoke", "miss_rate_mean", "Missed-hazard rate on unseen Smoke", True),
                   ("Mixture", "escalation_mean", "Escalation adequacy on unseen Mixture", False)]):
        sub = l[l.held_out == cls]
        xs, ys, names = [], [], []
        for _, r in sub.iterrows():
            if r.model in z.index:
                xs.append(z.loc[r.model, "decision_acc_mean"]); ys.append(r[ycol]); names.append(str(KEY_OF[r.model]))
        floor = 8e-5
        yp = [max(v, floor) if logy else v for v in ys]
        ax.scatter(xs, yp, s=64, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        if logy:
            ax.set_yscale("log"); ax.set_ylim(floor * 0.45, 0.55)
            ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1]); ax.set_yticklabels(["0 (floor)", "0.1%", "1%", "10%"])
        else:
            ax.set_ylim(-0.06, 1.22); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        r = np.corrcoef(xs, ys)[0, 1]
        lo, hi = min(xs), max(xs)
        pad = 0.14 * (hi - lo) + 0.004
        ax.set_xlim(lo - pad, hi + pad)
        # Labels: points sharing a y-value (the zero floor, or full escalation)
        # would otherwise print on top of one another, so a tied group is
        # written vertically above its marker in left-to-right order instead.
        inv = ax.transData.transform          # points, so the test is visual
        placed = []
        for i in np.argsort(xs):
            px, py = inv((xs[i], yp[i]))
            dx, dy = 9.0, 5.0
            for qx, qy in placed:             # stack down off an occupied slot
                if abs(px - qx) < 26 and abs(py + dy - qy) < 11:
                    dy -= 12.0
            ax.annotate(names[i], (xs[i], yp[i]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=8.9, color=INK2, ha="left")
            placed.append((px, py + dy))
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("In-distribution decision accuracy")
        ax.set_ylabel(ylab)
        ax.set_title(f"Pearson r = {r:+.2f}", loc="left", fontsize=10.3, color=INK2)
        ax.grid(zorder=0)
    fig.suptitle("In-distribution accuracy does not order out-of-distribution safety",
                 x=0.005, ha="left", fontsize=12.4)
    key = "   ".join(f"{KEY_OF[m]} {PRETTY[m]}" for m in KEYED)
    fig.text(0.005, 0.012, key, ha="left", va="bottom", fontsize=8.9, color=INK2)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.savefig(os.path.join(FIG, "fig_dissociation.png")); plt.close(fig)
    print("wrote fig_dissociation.png")


def fig_disposition():
    """Where hazardous windows actually go: Monitor / sub-alarm / alarm-grade."""
    l = pd.read_csv(os.path.join(RES, "v3_loco_all.csv"))
    models = [m for m in PRETTY if m in set(l.model)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.86), sharey=True)
    for ax, cls in zip(axes, ["Smoke", "Mixture"]):
        sub = l[l.held_out == cls].set_index("model")
        miss = np.array([sub.loc[m, "miss_rate_mean"] for m in models])
        und = np.array([sub.loc[m, "under_escalation_mean"] for m in models])
        esc = np.array([sub.loc[m, "escalation_mean"] for m in models])
        x = np.arange(len(models))
        ax.bar(x, esc, 0.62, color=GOOD, label="Alarm-grade (a ≥ 3)", ec=SURFACE, lw=1.4, zorder=3)
        ax.bar(x, und, 0.62, bottom=esc, color=WARN_, label="Sub-alarm (a ∈ {1,2})", ec=SURFACE, lw=1.4, zorder=3)
        ax.bar(x, miss, 0.62, bottom=esc + und, color=CRIT, label="Monitor (a = 0)", ec=SURFACE, lw=1.4, zorder=3)
        for xi, e, u in zip(x, esc, und):
            if e > 0.06:
                ax.text(xi, e / 2, f"{e:.2f}", ha="center", va="center", fontsize=8, color="white")
            if u > 0.10:
                ax.text(xi, e + u / 2, f"{u:.2f}", ha="center", va="center", fontsize=8, color="white")
        ax.set_xticks(x); ax.set_xticklabels([PRETTY[m] for m in models], rotation=28, ha="right")
        ax.set_ylim(0, 1.03); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f"Held-out {cls}", loc="left")
        ax.grid(axis="y", zorder=0)
    axes[0].set_ylabel("Share of hazardous windows")
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], lb[::-1], ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.955), fontsize=9.7)
    fig.suptitle("What actually happens to a hazardous window the model has never seen",
                 x=0.005, y=0.995, ha="left", fontsize=12.4)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(FIG, "fig_disposition.png")); plt.close(fig)
    print("wrote fig_disposition.png")


def fig_alarmburden():
    """False-alarm rate expressed as alarms per hour per detector."""
    p = pd.read_csv(os.path.join(RES, "v2_perturb.csv"))
    models = [m for m in ["A_cost_weighted", "D_mlp", "E_gbm", "RF", "ThresholdRule"] if m in set(p.model)]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for i, m in enumerate(models):
        sub = p[(p.model == m) & (p.perturbation.isin(["dropout", "clean"]))].sort_values("level")
        y = np.maximum(sub.fa_per_clean_mean.values * 1000.0, 0.03)
        ax.plot(sub.level, y, "-o", ms=5.5, lw=2, color=CAT[i], label=PRETTY[m], zorder=3)
    ax.axhline(3.33, color=CRIT, lw=1.6, ls="--", zorder=4)
    ax.text(0.05, 0.30, "Contextual reference, not a compliance threshold:\nEEMUA 191 envelope for an entire operator position\n(about 6 annunciated alarms h⁻¹, or 3.3 per 1,000\nwindows at one window every 2 s). This panel shows\nalarm-grade indications, which are not annunciations.",
            ha="left", va="center", fontsize=8.4, color=CRIT)
    ax.set_yscale("log"); ax.set_ylim(0.025, 2600)
    ax.set_yticks([0.05, 1, 3.33, 10, 100, 1000])
    ax.set_yticklabels(["0 (floor)", "1", "3.3", "10", "100", "1,000"])
    ax.set_xlabel("Sensor channels removed (k of 7)")
    ax.set_ylabel("High-severity alarms per 1,000 clean windows (log)")
    ax.set_title("One lost sensor is enough to swamp an operator", loc="left")
    ax.grid(axis="y", zorder=0); ax.legend(fontsize=9.3, loc="lower right", ncol=2)
    fig.savefig(os.path.join(FIG, "fig_alarmburden.png")); plt.close(fig)
    print("wrote fig_alarmburden.png")




# ---------------- pipeline schematic ----------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 3.9))
    box = dict(boxstyle="round,pad=0.42", linewidth=1.4)
    def b(x, y, w, h, txt, fc, ec, fs=8.6, tc=INK):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                                   linewidth=1.4, zorder=2, joinstyle="round"))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=fs, color=tc, zorder=3, linespacing=1.35)
    def arrow(x1, y1, x2, y2, style="-|>", col=MUTED):
        ax.annotate("", (x2, y2), (x1, y1),
                    arrowprops=dict(arrowstyle=style, color=col, lw=1.3,
                                    shrinkA=2, shrinkB=2), zorder=1)
    b(0.2, 3.05, 1.9, 0.9, "Thermal image\n$I_t$", "#eef2f6", "#b9c6d2")
    b(0.2, 1.35, 1.9, 0.9, "Sensor window\n$X_t \\in \\mathbb{R}^{20\\times7}$", "#eef2f6", "#b9c6d2")
    b(2.7, 3.05, 2.2, 0.9, "Perception\nYOLOv8n-cls", "#dde8f2", BLUE)
    b(2.7, 1.35, 2.2, 0.9, "Anomaly\nLSTM autoencoder", "#dde8f2", BLUE)
    b(5.5, 2.2, 2.0, 0.9, "Coordination\nstate $\\varphi_t \\in \\mathbb{R}^{22}$", "#dde8f2", BLUE)
    b(8.1, 2.2, 2.2, 0.9, "Decision\ncost-weighted CE", "#d8ece1", GOOD)
    b(8.1, 0.35, 2.2, 0.9, "Reasoning\nGemma 3 1B, local", "#f6efd9", WARN_)
    b(11.0, 2.2, 1.7, 0.9, "Action $a_t$\nto operator", "#eef2f6", "#b9c6d2")
    b(11.0, 0.35, 1.7, 0.9, "Explanation $E_t$\n(advisory)", "#eef2f6", "#b9c6d2")
    arrow(2.1, 3.5, 2.7, 3.5); arrow(2.1, 1.8, 2.7, 1.8)
    arrow(4.9, 3.5, 5.5, 2.95); arrow(4.9, 1.8, 5.5, 2.35)
    arrow(7.5, 2.65, 8.1, 2.65); arrow(10.3, 2.65, 11.0, 2.65)
    arrow(9.2, 2.2, 9.2, 1.25); arrow(10.3, 0.8, 11.0, 0.8)
    ax.text(5.25, 3.40, "class + confidence", fontsize=8, color=MUTED, ha="center")
    ax.text(5.25, 1.88, "anomaly score $\\tilde\\rho_t$", fontsize=8, color=MUTED, ha="center")
    ax.text(9.35, 1.72, "one way only", fontsize=8, color=CRIT, ha="left", style="italic")
    ax.add_patch(plt.Rectangle((7.95, 0.18), 2.5, 2.98, fill=False,
                               edgecolor=MUTED, linestyle=(0, (4, 3)), lw=1.0, zorder=0))
    ax.text(9.2, 3.25, "advisory layer, no actuation", fontsize=8.4, color=MUTED, ha="center")
    ax.set_xlim(0, 12.9); ax.set_ylim(0, 4.3); ax.axis("off")
    ax.set_title("Pipeline. The decision state carries no visual term, and the language model "
                 "cannot change the action.", loc="left", fontsize=10.8)
    fig.savefig(os.path.join(FIG, "fig_architecture.png")); plt.close(fig)
    print("wrote fig_architecture.png")


# ---------------- perception (replotted from reported results) ----------------
PERC_LABELS = ["Mixture", "NoGas", "Perfume", "Smoke"]
PERC_CM = np.array([[320, 0, 0, 0], [0, 308, 9, 3], [0, 3, 317, 0], [0, 0, 0, 320]], float)
PERC_PRF = {"Mixture": (1.000, 1.000, 1.000), "NoGas": (0.990, 0.963, 0.976),
            "Perfume": (0.972, 0.991, 0.981), "Smoke": (0.991, 1.000, 0.995)}


def fig_perception():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4),
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, (M, ttl, fmt) in zip(axes[:2],
                                 [(PERC_CM, "(a)  Counts", "{:.0f}"),
                                  (PERC_CM / PERC_CM.sum(1, keepdims=True),
                                   "(b)  Row-normalized", "{:.3f}")]):
        im = ax.imshow(M / M.max(), cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                v = M[i, j]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9.1,
                        color="white" if v / M.max() > 0.5 else INK2)
        ax.set_xticks(range(4)); ax.set_xticklabels(PERC_LABELS, rotation=28, ha="right")
        ax.set_yticks(range(4)); ax.set_yticklabels(PERC_LABELS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(ttl, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax = axes[2]
    x = np.arange(4); w = 0.26
    for i, (lab, col) in enumerate([("Precision", BLUE), ("Recall", ORANGE), ("F1", GREEN)]):
        v = [PERC_PRF[c][i] for c in PERC_LABELS]
        bb = ax.bar(x + (i - 1) * w, v, w * 0.9, color=col, label=lab, ec=SURFACE, lw=1.2, zorder=3)
        for b_, vv in zip(bb, v):
            ax.text(b_.get_x() + b_.get_width() / 2, vv + 0.003, f"{vv:.3f}",
                    ha="center", va="bottom", fontsize=8, color=INK2, rotation=90)
    ax.set_ylim(0.94, 1.030); ax.set_xticks(x); ax.set_xticklabels(PERC_LABELS, rotation=28, ha="right")
    ax.set_ylabel("Score (axis truncated at 0.94)")
    ax.set_title("(c)  Per-class scores", loc="left")
    ax.grid(axis="y", zorder=0)
    ax.legend(fontsize=8.6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.text(0.005, -0.06, "Replotted from the reported perception results. The split is randomized over "
             "consecutive frames from one session, so these are an optimistic upper bound.",
             fontsize=8.4, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_perception.png")); plt.close(fig)
    print("wrote fig_perception.png")


# ---------------- edge latency ----------------
EDGE = [("Desktop CPU\nfloat32", 5.8, 9.6, 8.74),
        ("Pi 4 float32\nsynchronous", 14738, 24202, 8.74),
        ("Pi 4 INT8\nsynchronous", 14675, 24028, 4.19),
        ("Pi 4 INT8\nasync reasoning", 235, 413, 4.19)]


def fig_edge():
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    x = np.arange(len(EDGE)); w = 0.36
    means = [e[1] for e in EDGE]; p99 = [e[2] for e in EDGE]
    b1 = ax.bar(x - w / 2, means, w * 0.92, color=BLUE, label="Mean", ec=SURFACE, lw=1.2, zorder=3)
    b2 = ax.bar(x + w / 2, p99, w * 0.92, color=ORANGE, label="P99", ec=SURFACE, lw=1.2, zorder=3)
    for bb, vv in list(zip(b1, means)) + list(zip(b2, p99)):
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() * 1.13,
                f"{vv:,.0f}" if vv >= 10 else f"{vv:.1f}", ha="center", va="bottom",
                fontsize=8, color=INK2)
    ax.axhline(2000, color=CRIT, ls="--", lw=1.6, zorder=4)
    ax.text(3.42, 2400, "2,000 ms sampling budget", ha="right", fontsize=8.6, color=CRIT)
    ax.set_yscale("log"); ax.set_ylim(2, 120000)
    ax.set_xticks(x); ax.set_xticklabels([e[0] for e in EDGE], fontsize=9.1)
    ax.set_ylabel("End-to-end latency, ms (log)")
    ax.set_title("(a)  Only the asynchronous configuration fits the budget", loc="left")
    ax.grid(axis="y", zorder=0); ax.legend(fontsize=9.3, loc="upper left")

    ax = axes[1]
    # single axis: both measures indexed to their pre-throttle value, because a
    # second y-scale would invite a comparison the units do not support.
    series = [("CPU clock", 1500, 1234, GREEN, "MHz"), ("P50 latency", 420, 640, PURPLE, "ms")]
    xx = np.arange(2)
    for i, (nm, a0, a1, col, unit) in enumerate(series):
        vals = [1.0, a1 / a0]
        bb = ax.bar(xx + (i - 0.5) * 0.34, vals, 0.31, color=col, ec=SURFACE, lw=1.2,
                    zorder=3, label=f"{nm} ({unit})")
        for b_, v, raw in zip(bb, vals, [a0, a1]):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 0.025,
                    f"{v*100:.0f}%\n{raw:,}{unit}", ha="center", va="bottom",
                    fontsize=8, color=INK2, linespacing=1.25)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=":", zorder=2)
    ax.set_ylim(0, 1.95)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Indexed to the pre-throttle value")
    ax.set_xticks(xx)
    ax.set_xticklabels(["before throttle", "after throttle\n(from about 24 min)"], fontsize=9.1)
    ax.set_title("(b)  36-minute passively cooled run", loc="left")
    ax.grid(axis="y", zorder=0)
    ax.legend(fontsize=8.9, loc="upper left")
    fig.text(0.005, -0.05, "Panel (b) plots the two reported operating points of the sustained run, "
             "not a continuous trace; the per-minute log was not available for this revision.",
             fontsize=8.4, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_edge.png")); plt.close(fig)
    print("wrote fig_edge.png")


# ---------------- anomaly ROC + score distributions ----------------
def fig_roc():
    d = json.load(open(os.path.join(RES, "v2_roc.json")))
    a = json.load(open(os.path.join(RES, "v2_anomaly.json")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for i, r in enumerate(d["held_out"]):
        ax.plot(r["fpr"], r["tpr"], lw=1.5, color=BLUE, alpha=0.55, zorder=3,
                label="Held-out, per seed" if i == 0 else None)
    ax.plot(d["in_sample"]["fpr"], d["in_sample"]["tpr"], lw=2.2, color=ORANGE,
            zorder=4, label="In-sample (threshold fitted on the same windows)")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.0, ls=":", zorder=2)
    ax.scatter([a["in_sample"]["fpr"]], [a["in_sample"]["tpr"]], s=70, color=ORANGE,
               edgecolor=SURFACE, linewidth=1.4, zorder=6)
    ax.annotate(f"in-sample operating point\nFPR {a['in_sample']['fpr']:.3f}, TPR {a['in_sample']['tpr']:.3f}",
                (a["in_sample"]["fpr"], a["in_sample"]["tpr"]), textcoords="offset points",
                xytext=(58, 22), fontsize=8.2, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.scatter([a["fpr_mean"]], [a["tpr_mean"]], s=70, color=BLUE,
               edgecolor=SURFACE, linewidth=1.4, zorder=6)
    ax.annotate(f"held-out operating point\nFPR {a['fpr_mean']:.3f}, TPR {a['tpr_mean']:.3f}",
                (a["fpr_mean"], a["tpr_mean"]), textcoords="offset points",
                xytext=(40, -34), fontsize=8.2, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.04)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"(a)  ROC. Held-out AUC {a['auc_mean']:.4f} ± {a['auc_std']:.4f}, "
                 f"in-sample {a['in_sample']['auc']:.4f}", loc="left", fontsize=10.3)
    ax.grid(zorder=0); ax.legend(fontsize=8.9, loc="lower right")

    ax = axes[1]
    cls = ["NoGas", "Perfume", "Mixture", "Smoke"]
    cols = [BLUE, OLIVE, PURPLE, ORANGE]
    for c, col in zip(cls, cols):
        v = np.array(d["per_class_scores"][c], float)
        v = np.log10(np.clip(v, 1e-4, None))
        ax.hist(v, bins=60, histtype="stepfilled", alpha=0.55, color=col,
                label=c, zorder=3, linewidth=0)
    ax.set_xlabel("log₁₀ reconstruction error"); ax.set_ylabel("Windows")
    ax.set_title("(b)  Score distribution by class", loc="left")
    ax.grid(axis="y", zorder=0); ax.legend(fontsize=9.1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_roc.png")); plt.close(fig)
    print("wrote fig_roc.png")


# ---------------- in-distribution action-selection matrices ----------------
def fig_actionmatrix():
    df = pd.read_csv(os.path.join(RES, "v2_action_matrix.csv"))
    models = [m for m in PRETTY if m in set(df.model)]
    gases = ["NoGas", "Smoke", "Mixture", "Perfume"]
    acts = ["0 Monitor", "1 Sample", "2 Verify", "3 Alarm", "4 ESD"]
    n = len(models)
    ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.05 * ncol, 3.35 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for k, m in enumerate(models):
        ax = axes[k]
        M = np.zeros((4, 5))
        sub = df[df.model == m]
        for _, r in sub.iterrows():
            M[gases.index(r.gas), int(r.action)] = r.share
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(5):
                if M[i, j] >= 0.005:
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if M[i, j] > 0.55 else INK2)
        ax.set_xticks(range(5)); ax.set_xticklabels(acts, rotation=52, ha="right", fontsize=8)
        ax.set_yticks(range(4)); ax.set_yticklabels(gases, fontsize=8)
        ax.set_title(PRETTY[m], loc="left", fontsize=9.7)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.suptitle("In distribution every model lands on the target action for both hazardous classes. "
                 "No model ever selects action 2.", x=0.005, ha="left", fontsize=11.9)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=3.4)
    fig.savefig(os.path.join(FIG, "fig_actionmatrix.png")); plt.close(fig)
    print("wrote fig_actionmatrix.png")


# ---------------- calibration ----------------
def fig_calibration():
    a = pd.read_csv(os.path.join(RES, "v2_calibration.csv"))
    sw = pd.read_csv(os.path.join(RES, "v2_calibration_binsweep.csv"))
    name = {"raw_softmax": "Raw softmax", "mc_dropout_20": "MC dropout (20)",
            "temp_scaling": "Temperature scaling", "deep_ensemble_5": "Deep ensemble (5)"}
    order = ["raw_softmax", "mc_dropout_20", "temp_scaling", "deep_ensemble_5"]
    a = a.set_index("estimator").loc[order].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.54))
    ax = axes[0]
    x = np.arange(len(a))
    b = ax.bar(x, a.ece_mean, 0.58, color=BLUE, ec=SURFACE, lw=1.2, zorder=3)
    ax.errorbar(x, a.ece_mean, yerr=[a.ece_mean - a.ece_lo, a.ece_hi - a.ece_mean],
                fmt="none", ecolor=INK2, elinewidth=1.1, capsize=3.4, zorder=4)
    for bi, v, hi in zip(b, a.ece_mean, a.ece_hi):
        ax.text(bi.get_x() + bi.get_width() / 2, hi + 0.002, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([name[e] for e in a.estimator], rotation=24, ha="right")
    ax.set_ylim(0, float(a.ece_hi.max()) * 1.32)
    ax.set_ylabel("Expected calibration error")
    ax.set_title("(a)  ECE, adaptive bins, bootstrap 95% CI", loc="left", fontsize=10.3)
    ax.grid(axis="y", zorder=0)
    ax = axes[1]
    w = 0.38
    b1 = ax.bar(x - w / 2, a.ece_mean, w * 0.92, color=BLUE, label="All windows", ec=SURFACE, lw=1.2, zorder=3)
    b2 = ax.bar(x + w / 2, a.ece_danger_mean, w * 0.92, color=ORANGE, label="Hazardous windows only",
                ec=SURFACE, lw=1.2, zorder=3)
    for bb, vv in list(zip(b1, a.ece_mean)) + list(zip(b2, a.ece_danger_mean)):
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.0018, f"{vv:.4f}",
                ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([name[e] for e in a.estimator], rotation=24, ha="right")
    ax.set_ylim(0, float(a.ece_mean.max()) * 1.45)
    ax.set_ylabel("Expected calibration error")
    ax.set_title("(b)  Class-conditional ECE", loc="left", fontsize=10.3)
    ax.grid(axis="y", zorder=0)
    ax.legend(fontsize=8.9, loc="upper right", bbox_to_anchor=(1.0, 0.90))
    ax = axes[2]
    for i, e in enumerate(order):
        g = sw[sw.estimator == e].groupby("n_bins").ece.mean()
        ax.plot(g.index, g.values, "-o", ms=5, lw=1.9, color=CAT[i], label=name[e], zorder=3)
    ax.set_xlabel("Number of bins"); ax.set_ylabel("Expected calibration error")
    ax.set_title("(c)  Sensitivity to bin count", loc="left", fontsize=10.3)
    ax.grid(zorder=0); ax.legend(fontsize=8.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_calibration.png")); plt.close(fig)
    print("wrote fig_calibration.png")


# ---------------- anomaly-score influence on the action ----------------
def fig_anominfluence():
    d = pd.read_csv(os.path.join(RES, "v2_anomaly_influence.csv"))
    order = (d.groupby("model").frac_windows_action_changed.first()
             .sort_values(ascending=False).index.tolist())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.16),
                             gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    fr = [100 * d[d.model == m].frac_windows_action_changed.iloc[0] for m in order]
    sd = [100 * d[d.model == m].frac_changed_std.iloc[0] for m in order]
    x = np.arange(len(order))
    b = ax.bar(x, fr, 0.62, color=BLUE, ec=SURFACE, lw=1.2, zorder=3)
    ax.errorbar(x, fr, yerr=sd, fmt="none", ecolor=INK2, elinewidth=1.0, capsize=3, zorder=4)
    for bi, v, e in zip(b, fr, sd):
        ax.text(bi.get_x() + bi.get_width() / 2, v + e + 1.4, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=8.2, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[m] for m in order], rotation=28, ha="right")
    ax.set_ylim(0, max(fr) * 1.30)
    ax.yaxis.set_major_formatter(lambda v, pos: f"{v:.0f}%")
    ax.set_ylabel("Test windows whose action changes")
    ax.set_title("(a)  Sensitivity of the action to the anomaly input", loc="left")
    ax.grid(axis="y", zorder=0)
    ax.text(0.98, 0.94, "anomaly score swept across [0, 1]\nwith all 21 other features held fixed",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.2, color=MUTED, style="italic")

    ax = axes[1]
    others = [m for m in order if m != "E_gbm"]
    for i, m in enumerate(others):
        sub = d[d.model == m].sort_values("level")
        ax.plot(sub.level, sub.miss_rate_mean, "-o", ms=4, lw=1.6, color="#b8c4ce", zorder=2,
                label="six other models" if i == 0 else None)
    sub = d[d.model == "E_gbm"].sort_values("level")
    ax.plot(sub.level, sub.miss_rate_mean, "-o", ms=7, lw=2.4, color=CRIT, zorder=4,
            label="Gradient boosting")
    ax.annotate(f"{sub.miss_rate_mean.iloc[0]:.4f}", (0.0, sub.miss_rate_mean.iloc[0]),
                textcoords="offset points", xytext=(10, -4), fontsize=8.6, color=CRIT)
    ax.annotate(f"{sub.miss_rate_mean.iloc[-1]:.4f}", (1.0, sub.miss_rate_mean.iloc[-1]),
                textcoords="offset points", xytext=(-6, 12), fontsize=8.6, color=CRIT, ha="right")
    ax.set_xlabel("Anomaly score forced to this value")
    ax.set_ylabel("Missed-hazard rate")
    ax.set_ylim(-0.02, 0.40)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("(b)  What happens when the anomaly channel reads zero", loc="left")
    ax.grid(zorder=0); ax.legend(fontsize=9.1, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_anominfluence.png")); plt.close(fig)
    print("wrote fig_anominfluence.png")


# ---------------- leakage sensitivity ----------------
PROTO_ORDER = ["random", "blocked_noembargo", "blocked_embargo", "leave_one_block"]
PROTO_LABEL = {"random": "Random split", "blocked_noembargo": "Blocked,\nno embargo",
               "blocked_embargo": "Blocked +\nembargo", "leave_one_block": "Train early,\ntest late"}


def fig_leakage():
    d = pd.read_csv(os.path.join(RES, "v3_leakage.csv"))
    piv = d.pivot_table(index="model", columns="protocol", values="decision_acc_mean")[PROTO_ORDER]
    err = d.pivot_table(index="model", columns="protocol", values="decision_acc_std")[PROTO_ORDER]
    models = [m for m in PRETTY if m in piv.index]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.08), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    w = 0.20
    for j, pr in enumerate(PROTO_ORDER):
        x = np.arange(len(models)) + (j - 1.5) * w
        v = [piv.loc[m, pr] for m in models]
        e = [err.loc[m, pr] for m in models]
        ax.bar(x, v, w * 0.9, yerr=e, color=CAT[j], ec=SURFACE, lw=1.0, zorder=3,
               error_kw=dict(ecolor=INK2, elinewidth=0.8, capsize=1.8), label=PROTO_LABEL[pr].replace("\n", " "))
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([PRETTY[m] for m in models], rotation=26, ha="right")
    ax.set_ylim(0.55, 1.05); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Decision accuracy")
    ax.set_title("(a)  Same models, four partitioning protocols", loc="left")
    ax.grid(axis="y", zorder=0)
    ax.legend(fontsize=8.6, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.06))

    ax = axes[1]
    gap = 100 * (piv["random"] - piv["blocked_embargo"])
    order = gap.sort_values(ascending=False).index.tolist()
    v = [gap[m] for m in order]
    b = ax.barh(np.arange(len(order)), v, 0.6,
                color=[ORANGE if x > 0 else GREEN for x in v], ec=SURFACE, lw=1.2, zorder=3)
    for bi, x in zip(b, v):
        ax.text(x + (0.10 if x >= 0 else -0.10), bi.get_y() + bi.get_height() / 2,
                f"{x:+.2f}", va="center", ha="left" if x >= 0 else "right",
                fontsize=8.4, color=INK2)
    ax.axvline(0, color=MUTED, lw=1.0, zorder=2)
    ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([PRETTY[m] for m in order])
    ax.set_xlim(-1.6, 5.2)
    ax.set_xlabel("Accuracy points added by a random split")
    ax.set_title("(b)  Inflation attributable to window overlap", loc="left")
    ax.grid(axis="x", zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_leakage.png")); plt.close(fig)
    print("wrote fig_leakage.png")


# ---------------- powered comparison and equivalence ----------------
def fig_powered():
    d = pd.read_csv(os.path.join(RES, "v3_powered.csv")).set_index("model")
    b = json.load(open(os.path.join(RES, "v3_powered_bayes.json")))
    sig = b["vs_cost_weighted"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.02), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    order = d.decision_acc_mean.sort_values(ascending=False).index.tolist()
    x = np.arange(len(order))
    v = [d.loc[m, "decision_acc_mean"] for m in order]
    e = [d.loc[m, "decision_acc_std"] for m in order]
    cols = [ORANGE if m == "A_cost_weighted" else ("#1f8f5f" if m == "Ordinal" else "#9fb4c4") for m in order]
    bb = ax.bar(x, v, 0.64, yerr=e, color=cols, ec=SURFACE, lw=1.2, zorder=3,
                error_kw=dict(ecolor=INK2, elinewidth=1.0, capsize=2.6))
    for bi, vv, ee in zip(bb, v, e):
        ax.text(bi.get_x() + bi.get_width() / 2, vv + ee + 0.008, f"{vv:.3f}",
                ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([PRETTY.get(m, m) for m in order], rotation=26, ha="right")
    ax.set_ylim(0.70, 1.03); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Decision accuracy")
    ax.set_title(f"(a)  {b['n_runs']} randomized runs", loc="left")
    ax.grid(axis="y", zorder=0)

    ax = axes[1]
    names = [m for m in order if m in sig]
    left = [sig[m]["p_other_better"] for m in names]
    mid = [sig[m]["p_practically_equivalent"] for m in names]
    right = [sig[m]["p_ref_better"] for m in names]
    y = np.arange(len(names))
    ax.barh(y, left, 0.62, color=BLUE, ec=SURFACE, lw=1.2, zorder=3, label="other better")
    ax.barh(y, mid, 0.62, left=left, color="#c9d3da", ec=SURFACE, lw=1.2, zorder=3,
            label="practically equivalent")
    ax.barh(y, right, 0.62, left=np.array(left) + np.array(mid), color=ORANGE, ec=SURFACE,
            lw=1.2, zorder=3, label="cost-weighted better")
    for yi, m in zip(y, mid):
        if m > 0.14:
            ax.text(left[yi] + m / 2, yi, f"{m:.2f}", ha="center", va="center",
                    fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([PRETTY.get(m, m) for m in names])
    ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Posterior probability")
    ax.set_title("(b)  Equivalence to the cost-weighted policy, ROPE ± 1 point", loc="left")
    ax.grid(axis="x", zorder=0); ax.legend(fontsize=8.6, ncol=3, loc="lower center",
                                           bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_powered.png")); plt.close(fig)
    print("wrote fig_powered.png")


ALL_FIGS = ["protocol", "leakage", "architecture", "perception", "anomaly", "roc", "zoo", "powered",
            "dissociation", "loco", "disposition", "actionmatrix", "costsweep",
            "ablation", "perturb", "alarmburden", "anominfluence", "calibration"]

if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ALL_FIGS
    for t in todo:
        try:
            globals()["fig_" + t]()
        except Exception as e:
            print(f"skip {t}: {type(e).__name__}: {e}")


# ================== v10 additions: SPE house-style figures ==================
# Three figures added when the manuscript was restructured to SPE Journal form.
# The first two are schematics in the idiom the journal uses for workflow and
# decision-logic diagrams; the third is a synthesis figure that makes the
# paper's central claim visible in one panel.

def _rbox(ax, x, y, w, h, txt, fc, ec, fs=8.4, tc=INK, lw=1.4, ls="-"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.09",
                                facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3, linespacing=1.32)


def _arrow(ax, p, q, col=MUTED, lw=1.3, style="-|>"):
    ax.annotate("", q, p, arrowprops=dict(arrowstyle=style, color=col, lw=lw,
                                          shrinkA=3, shrinkB=3), zorder=1)


def fig_workflow():
    """End-to-end evaluation workflow, from raw corpus to the reported metric set."""
    fig, ax = plt.subplots(figsize=(11.5, 6.34))
    SPINE_X, W, H = 4.55, 2.9, 0.74
    stages = [
        (5.55, "Raw corpus\n6,400 samples, 4 classes", "#e9eef3", "#b9c6d2"),
        (4.55, "Windowing\n20 steps, boundary windows dropped", "#dde8f2", BLUE),
        (3.55, "Leakage-controlled split\ncontiguous 20%, 20-window embargo", "#dde8f2", BLUE),
        (2.55, "Fit on training partition only\nscaler, anomaly percentiles", "#dde8f2", BLUE),
        (1.55, "Train 11 comparators\none protocol, seed-varying partition", "#d8ece1", GOOD),
        (0.55, "Safety metric set\n5 quantities + Clopper-Pearson bounds", "#f6efd9", WARN_),
    ]
    for y, txt, fc, ec in stages:
        _rbox(ax, SPINE_X, y, W, H, txt, fc, ec)
    for i in range(len(stages) - 1):
        _arrow(ax, (SPINE_X + W / 2, stages[i][0]), (SPINE_X + W / 2, stages[i + 1][0] + H))

    # left rail: what each stage guards against
    guards = [(5.55, "6,324 windows\n1,581 per class"),
              (3.55, "no training window shares\na raw row with a test window"),
              (2.55, "no test statistic\nreaches the model"),
              (1.55, "block position drawn\nfrom the run seed")]
    for y, txt in guards:
        _rbox(ax, 0.75, y + 0.04, 3.3, H - 0.08, txt, SURFACE, "#d5d5d5", fs=7.8, tc=INK2, ls=(0, (4, 3)))
        _arrow(ax, (4.05, y + H / 2), (SPINE_X, y + H / 2), col="#c2c2c2", lw=1.0)

    # right rail: the four evaluation regimes fed from the trained models
    regimes = [(2.30, "In distribution\n30 randomized runs"),
               (1.55, "Leave-one-class-out\nunseen hazard"),
               (0.80, "Graded degradation\nnoise, drift, channel loss"),
               (0.05, "Calibration and\nupstream-input failure")]
    for y, txt in regimes:
        _rbox(ax, 8.25, y + 0.02, 3.2, 0.70, txt, "#eef2f6", "#b9c6d2", fs=7.8, tc=INK2)
        _arrow(ax, (SPINE_X + W, 1.55 + H / 2), (8.25, y + 0.37), col="#c2c2c2", lw=1.0)

    ax.text(0.75, 6.52, "Guard against", fontsize=9.3, color=MUTED, style="italic")
    ax.text(SPINE_X, 6.52, "Protocol", fontsize=9.3, color=MUTED, style="italic")
    ax.text(8.25, 6.52, "Evaluation regime", fontsize=9.3, color=MUTED, style="italic")
    ax.set_xlim(0.4, 11.7); ax.set_ylim(-0.15, 6.85); ax.axis("off")
    fig.savefig(os.path.join(FIG, "fig_workflow.png")); plt.close(fig)
    print("wrote fig_workflow.png")


def fig_decisionlogic():
    """How one window becomes an action, and where each metric reads it."""
    fig, ax = plt.subplots(figsize=(11.5, 4.38))
    _rbox(ax, 0.2, 1.88, 1.9, 0.8, "Sensor window\n$X_t$", "#eef2f6", "#b9c6d2")
    _rbox(ax, 2.55, 1.88, 2.05, 0.8, "Decision state\n$\\varphi_t \\in \\mathbb{R}^{22}$", "#dde8f2", BLUE)
    _rbox(ax, 5.05, 1.88, 1.95, 0.8, "Selected action\n$a_t \\in \\{0,\\dots,4\\}$", "#d8ece1", GOOD)
    _arrow(ax, (2.1, 2.28), (2.55, 2.28)); _arrow(ax, (4.6, 2.28), (5.05, 2.28))

    X0, XW, RH, GAP = 7.9, 2.45, 0.66, 0.17
    ladder = [(0, "Monitor", CRIT, "#f3e2e0"), (1, "Increase sampling", WARN_, "#f7f0da"),
              (2, "Request verification", WARN_, "#f7f0da"), (3, "Raise alarm", GOOD, "#dcefe6"),
              (4, "Emergency shutdown", GOOD, "#dcefe6")]
    ys = {}
    BOUNDARY_GAP = 0.34            # extra room so the boundary label has its own band
    for a_, lab, ec, fc in ladder:
        y = 0.10 + a_ * (RH + GAP) + (BOUNDARY_GAP if a_ >= 3 else 0.0); ys[a_] = y
        _rbox(ax, X0, y, XW, RH, f"{a_}   {lab}", fc, ec, fs=8.3)
    _arrow(ax, (7.0, 2.28), (X0, 2.28))

    # the alarm boundary sits in the gap between rung 2 and rung 3
    yb = (ys[2] + RH + ys[3]) / 2
    ax.plot([X0 - 0.28, X0 + XW + 0.28], [yb, yb], color=INK2, lw=1.2,
            ls=(0, (5, 3)), zorder=5)
    ax.text(X0 + XW / 2, yb + 0.07, "alarm boundary: an operator is notified above this line",
            fontsize=8.3, color=INK2, style="italic", ha="center", va="bottom", zorder=6)

    def bracket(x, y0, y1, col, label, sub):
        ax.plot([x, x], [y0, y1], color=col, lw=2.4, solid_capstyle="round", zorder=3)
        for yy in (y0, y1):
            ax.plot([x - 0.12, x], [yy, yy], color=col, lw=2.4, zorder=3)
        ax.text(x + 0.18, (y0 + y1) / 2 + 0.10, label, fontsize=9.3, color=INK, va="center")
        ax.text(x + 0.18, (y0 + y1) / 2 - 0.17, sub, fontsize=8.2, color=MUTED, va="center")

    XB = X0 + XW + 0.42
    bracket(XB, ys[0], ys[0] + RH, CRIT, "Missed hazard", "hazardous windows at $a = 0$")
    bracket(XB, ys[1], ys[2] + RH, WARN_, "Under-escalation", "hazardous windows at $a \\in \\{1, 2\\}$")
    bracket(XB, ys[3], ys[4] + RH, GOOD, "Escalation adequacy", "hazardous windows at $a \\geq 3$")

    # the fourth metric reads the same top rungs, but for CLEAN windows: call it
    # out from the left so it does not collide with the brackets on the right
    ax.annotate("High-severity false alarm\nclean windows at $a \\geq 3$",
                (X0 - 0.06, ys[3] + RH * 0.9), (4.95, 4.55), fontsize=8.9, color=INK,
                ha="left", va="top",
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.3,
                                connectionstyle="arc3,rad=0.22"))
    ax.set_xlim(0, 12.9); ax.set_ylim(-0.12, 4.85); ax.axis("off")
    ax.set_title("One window, one action, and the boundary each metric reads",
                 loc="left", fontsize=11.3, pad=2)
    fig.savefig(os.path.join(FIG, "fig_decisionlogic.png")); plt.close(fig)
    print("wrote fig_decisionlogic.png")


def fig_metricdisagreement():
    """Which model each metric crowns, and where it cannot choose at all.

    Three states per cell, because "who wins" is only half the story on this
    corpus: a metric that ties most of the field has not chosen a model, it has
    failed to discriminate, and collapsing that into a winner would hide the
    very saturation Section 4.3 is about.
    """
    pw = pd.read_csv(os.path.join(RES, "v3_powered.csv")).set_index("model").drop(index="Ordinal")
    lo = pd.read_csv(os.path.join(RES, "v3_loco_all.csv"))
    pt = pd.read_csv(os.path.join(RES, "v2_perturb.csv"))
    TIE = 3                                  # 3+ models sharing the optimum = no discrimination

    def best(df, col, high):
        v = df[col].dropna()
        if v.empty:
            return None, []
        tied = v.index[v == (v.max() if high else v.min())].tolist()
        return (PRETTY[tied[0]] if len(tied) == 1 else f"{len(tied)} models tied"), tied

    rows = [("In distribution, 30 runs",
             [best(pw, "decision_acc_mean", True), best(pw, "miss_rate_mean", False),
              best(pw, "escalation_mean", True), best(pw, "fa_per_clean_mean", False)])]
    for cls in ("Smoke", "Mixture"):
        s = lo[lo.held_out == cls].set_index("model").drop(index="Ordinal", errors="ignore")
        acc = pw.reindex(s.index).decision_acc_mean.dropna()
        rows.append((f"Unseen {cls}",
                     [(PRETTY[acc.idxmax()], [acc.idxmax()]), best(s, "miss_rate_mean", False),
                      best(s, "escalation_mean", True), (None, [])]))
    for cond, lv, name in (("drift", 0.5, "Drift ±50%"), ("dropout", 1.0, "Dropout k = 1"),
                           ("dropout", 7.0, "Dropout k = 7")):
        s = pt[(pt.perturbation == cond) & (pt.level == lv)].set_index("model")
        rows.append((name, [best(s, "decision_acc_mean", True), best(s, "miss_rate_mean", False),
                            best(s, "escalation_mean", True), best(s, "fa_per_clean_mean", False)]))

    cols = ["Decision\naccuracy", "Missed-hazard\nrate", "Escalation\nadequacy",
            "High-severity\nfalse alarms"]
    STATE = {"none":  (SURFACE,   "#e0e0e0", MUTED),
             "flat":  ("#f2f0e6", "#cfc7a8", INK2),
             "agree": ("#e3efe9", "#a8cbbb", INK),
             "differ": ("#f6e4e1", "#d9a79e", INK)}
    fig, ax = plt.subplots(figsize=(11.5, 5.16))
    nr = len(rows)
    for i, (label, cells) in enumerate(rows):
        y = nr - 1 - i
        acc_win = set(cells[0][1])
        for j, (txt, keys) in enumerate(cells):
            if txt is None:
                st, show = "none", "not defined"
            elif len(keys) >= TIE:
                st, show = "flat", txt
            elif acc_win & set(keys):
                st, show = "agree", txt
            else:
                st, show = "differ", txt
            fc, ec, tc = STATE[st]
            _rbox(ax, j * 2.5 + 0.06, y + 0.08, 2.38, 0.84, show, fc, ec,
                  fs=8.4 if len(show) < 16 else 7.8, tc=tc, lw=1.2)
        ax.text(-0.18, y + 0.5, label, ha="right", va="center", fontsize=9.6, color=INK)
    for j, c in enumerate(cols):
        ax.text(j * 2.5 + 1.25, nr + 0.14, c, ha="center", va="bottom",
                fontsize=9.6, color=INK2, linespacing=1.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=STATE["flat"][0], edgecolor=STATE["flat"][1],
                             label="cannot discriminate (3 or more tied)"),
                       Patch(facecolor=STATE["agree"][0], edgecolor=STATE["agree"][1],
                             label="picks the accuracy winner"),
                       Patch(facecolor=STATE["differ"][0], edgecolor=STATE["differ"][1],
                             label="picks a different model")],
              ncol=3, loc="lower center", bbox_to_anchor=(0.46, -0.15), fontsize=9.3)
    ax.set_xlim(-3.6, 10.0); ax.set_ylim(-0.62, nr + 0.8); ax.axis("off")
    fig.savefig(os.path.join(FIG, "fig_metricdisagreement.png")); plt.close(fig)
    print("wrote fig_metricdisagreement.png")
