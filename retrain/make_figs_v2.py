"""Regenerate the paper figures from the v2 results.

Palette validated with the dataviz validator (light surface #fcfcfb):
  lightness band PASS, chroma floor PASS, CVD separation WARN (green/orange
  dE 7.0, inside the 6-8 floor band) -> legal only with secondary encoding,
  so every categorical mark also carries a direct value label.
"""
import os, json, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "retrain", "results_v2")
FIG = os.path.join(ROOT, "figures_v2")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, GREEN, PURPLE, OLIVE = "#2f6f9f", "#c8642f", "#1f8f5f", "#8f4f9f", "#8a7a1a"
CAT = [BLUE, ORANGE, GREEN, PURPLE, OLIVE]
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#1a1a1a", "#4a4a4a", "#8a8a8a"

RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    # Base type is set so that a figure authored at 6.90 in and placed at the
    # 6.50 in text width lands near 10.4 pt. That is above the model papers'
    # own 7 to 9 pt, chosen deliberately: these figures are read on screen at
    # page width, where the smaller setting was not comfortably legible.
    "font.size": 11.0, "axes.titlesize": 11.8, "axes.labelsize": 11.0,
    "xtick.labelsize": 10.2, "ytick.labelsize": 10.2,
    "axes.edgecolor": "#cccccc", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.7,
    "legend.frameon": False, "figure.dpi": 600, "savefig.dpi": 600,
    "savefig.bbox": "tight",
}
plt.rcParams.update(RC)


def spe_scope():
    """Draw inside the SPE-diagram style without leaking it into the plots.

    ``spe_style.use_style`` sets a smaller base font for the block diagrams.
    Because matplotlib's rcParams are global, whether a result figure got that
    font used to depend on whether a diagram happened to be drawn earlier in the
    same process, which made the rendered figures depend on call order. Every
    diagram now draws inside this context instead.
    """
    return plt.rc_context(RC)

PRETTY = {
    "A_cost_weighted": "Cost-weighted", "Ordinal": "Ordinal cost", "B_unweighted": "Unweighted",
    "D_mlp": "MLP", "E_gbm": "GBM", "SVM": "SVM", "RF": "Random forest",
    "LSTM": "LSTM", "CQL": "CQL", "KNN": "k-NN",
    "ThresholdRule": "Shallow tree", "CUSUM": "CUSUM",
}


def note(fig, text, y=-0.06, fs=9.7, color=None, style="italic"):
    """A wrapped footnote under a figure.

    Unwrapped ``fig.text`` runs off the canvas, and because every figure is
    saved with a tight bounding box the canvas then grows to contain it: the
    LOCO figure was authored at 6.90 in and rendered at 12.16 in for exactly
    this reason, which halved every label once Word scaled it back to the
    column. Wrapping to the figure width keeps the rendered extent equal to
    the authored extent.
    """
    width_in = fig.get_size_inches()[0]
    ncols = max(40, int(width_in * 72 / (fs * 0.52)))
    fig.text(0.005, y, textwrap.fill(text, ncols), ha="left", va="top",
             fontsize=fs, color=MUTED if color is None else color, style=style,
             linespacing=1.35)


def _label_bars(ax, bars, vals, fmt="{:.3f}", dy=0.012, rot=0, fs=8.4):
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
    # Stacked rather than side by side. Two panels across the text width leave
    # about a quarter inch per model, and an upright value label is wider than
    # the gap between the two bars of a pair, so the labels collided. One panel
    # per row gives each pair more than twice the room.
    fig, axes = plt.subplots(2, 1, figsize=(6.90, 5.60), sharex=True)

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
            # The two bars of a pair are about a tenth of an inch apart and an
            # upright label is wider than that, so the pair is staggered in
            # height rather than crowded side by side.
            ax.text(bi.get_x() + bi.get_width() / 2, bi.get_height() * 1.45,
                    ("0" if v == 0 else f"{100*v:.2f}%"), ha="center", va="bottom",
                    fontsize=8.6, color=INK2, rotation=90 if v > 0 else 0)
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.7, 12.0)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_yticklabels(["0.01%", "0.1%", "1%", "10%"])
    ax.set_ylabel("Missed-hazard\nrate (log)", labelpad=2.0, linespacing=1.25)
    ax.set_title("(a)", loc="left")
    ax.grid(axis="y", zorder=0)
    # The legend sits above the panels rather than inside one: every corner of
    # panel (a) is occupied by an upright value label.
    _lh, _ll = ax.get_legend_handles_labels()

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
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8.6, color=INK2,
                    rotation=90)
    ax.set_ylim(0, 1.46)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([PRETTY[m] for m in models], rotation=24, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("Escalation adequacy\n(a ≥ 3)", labelpad=2.0, linespacing=1.25)
    ax.set_title("(b)", loc="left")
    ax.grid(axis="y", zorder=0)

    # The footnote sits inside a reserved strip rather than below the canvas:
    # a note placed outside the figure grows the tight bounding box, and the
    # figure then arrives in the document scaled down by that much.
    fig.legend(_lh, _ll, title="Held-out class", ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), fontsize=9.6, title_fontsize=10.2,
               handlelength=1.4, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.115, 1, 0.945), h_pad=1.2)
    note(fig, "Bars drawn at the floor of panel (a) are exactly zero. "
              "Ten of eleven models bound their missed-hazard rate on unseen Mixture at or below 3.3%, "
              "yet escalate between 0.5% and 100% of the same windows.", y=0.095)
    fig.savefig(os.path.join(FIG, "fig_loco.png"), dpi=600)
    plt.close(fig)
    print("wrote fig_loco.png")


# ------------------------------------------------------------ cost sweep
def fig_costsweep():
    df = pd.read_csv(os.path.join(RES, "v2_costsweep.csv"))
    df = df[df.cost_ratio != "label_function"].copy()
    df["C"] = [int(s.split(":")[0]) for s in df.cost_ratio]
    df = df.sort_values("C")
    fig, ax = plt.subplots(figsize=(6.90, 2.97))
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
                textcoords="offset points", xytext=(4, 20), fontsize=10.0, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    dep = df[df.C == 8].iloc[0]
    ax.annotate("deployed\nC = 8:1", (8, dep.decision_acc_mean),
                textcoords="offset points", xytext=(6, -32), fontsize=10.0, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.text(df.C.iloc[-1], df.miss_rate_mean.iloc[-1] + 0.03,
            "missed-hazard rate is flat at this level for every ratio",
            ha="right", fontsize=9.5, color=MUTED, style="italic")
    ax.set_xscale("log"); ax.set_xticks(df.C.tolist())
    ax.set_xticklabels([f"{c}:1" for c in df.C], fontsize=10.7)
    ax.set_ylim(-0.04, 1.12)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Cost ratio  C = c_miss : c_false")
    ax.set_ylabel("Rate")
    # panel title removed: the caption carries the statement
    ax.grid(axis="y", zorder=0); ax.legend(loc="center left", fontsize=10.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_costsweep.png"), dpi=600)
    plt.close(fig)
    print("wrote fig_costsweep.png")


# ------------------------------------------------------------------ zoo
def fig_zoo():
    df = pd.read_csv(os.path.join(RES, "v2_zoo.csv"))
    df = df[df.model != "CUSUM"]           # two-class; accuracy not comparable
    df = df.sort_values("decision_acc_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(6.90, 2.83))
    names = [PRETTY.get(m, m) for m in df.model]
    cols = [ORANGE if m == "A_cost_weighted" else "#9fb4c4" for m in df.model]
    x = np.arange(len(df))
    b = ax.bar(x, df.decision_acc_mean, 0.66, color=cols,
               edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.errorbar(x, df.decision_acc_mean, yerr=df.decision_acc_std, fmt="none",
                ecolor=INK2, elinewidth=1.0, capsize=3, zorder=4)
    for bi, v, sd in zip(b, df.decision_acc_mean, df.decision_acc_std):
        ax.text(bi.get_x() + bi.get_width() / 2, v + sd + 0.007, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9.3, color=INK2)
    lo = float(df.decision_acc_mean.min()); hi = float(df.decision_acc_mean.max())
    ax.axhspan(lo, hi, color=MUTED, alpha=0.10, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=24, ha="right")
    ax.set_ylim(min(0.80, lo - 0.06), 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Decision accuracy")
    ax.set_title("(a)", loc="left")
    ax.grid(axis="y", zorder=0)
    # in-panel note removed: the caption carries it (cost-weighted note)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_zoo.png"), dpi=600)
    plt.close(fig)
    print("wrote fig_zoo.png")


# ------------------------------------------------------------ perturbation
def fig_perturb():
    df = pd.read_csv(os.path.join(RES, "v2_perturb.csv"))
    kinds = [("noise", "Additive noise (σ)"), ("drift", "Calibration drift (±)"),
             ("dropout", "Channel dropout, k")]
    # Row labels are set on two lines: the third one is too long to sit beside a
    # panel this short and used to run into its neighbours.
    metrics = [("decision_acc_mean", "Decision\naccuracy"),
               ("escalation_mean", "Escalation\nadequacy"),
               ("fa_per_clean_mean", "High-severity\nfalse-alarm rate")]
    models = [m for m in ["A_cost_weighted", "D_mlp", "E_gbm", "RF", "ThresholdRule"]
              if m in set(df.model)]          # 5 series: hues assigned, never cycled
    fig, axes = plt.subplots(3, 3, figsize=(6.90, 6.04), sharex="col")
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
                ax.set_title(f"({'abcdefgh'[c]})", loc="left", fontsize=12.5)
            if r == 2:
                ax.set_xlabel(klabel, fontsize=9.8, labelpad=2.0)
            if c == 0:
                ax.set_ylabel(mlabel, labelpad=2.0, linespacing=1.25)
    axes[2, 1].text(0.5, 0.62, "peak false-alarm rate\nunder drift is 0.06%",
                    transform=axes[2, 1].transAxes, ha="center", fontsize=9.5,
                    color=MUTED, style="italic", linespacing=1.3)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.985),
               fontsize=10.2, columnspacing=1.5, handlelength=1.6)
    # figure title removed: journals carry the statement in the caption
    # A right margin: the widest column label runs past the last panel.
    fig.tight_layout(rect=(0, 0, 0.995, 0.945), h_pad=1.1, w_pad=2.2)
    fig.savefig(os.path.join(FIG, "fig_perturb.png"), dpi=600)
    plt.close(fig)
    print("wrote fig_perturb.png")


# --------------------------------------------------------------- ablation
def fig_ablation():
    df = pd.read_csv(os.path.join(RES, "v2_ablation.csv")).sort_values("decision_acc_mean")
    fig, ax = plt.subplots(figsize=(6.90, 2.76))
    y = np.arange(len(df))
    b = ax.barh(y, df.decision_acc_mean, 0.62, xerr=df.decision_acc_std,
                color=[ORANGE if s == "full_22" else "#9fb4c4" for s in df.state],
                error_kw=dict(ecolor=INK2, elinewidth=1.0, capsize=3),
                edgecolor=SURFACE, linewidth=1.2, zorder=3)
    for bi, v, sd, d in zip(b, df.decision_acc_mean, df.decision_acc_std, df.delta_pp_vs_full):
        ax.text(1.045, bi.get_y() + bi.get_height() / 2,
                f"{v:.3f}" + ("  (reference)" if abs(d) < 1e-9 else f"   {d:+.2f} pp"),
                va="center", ha="left", fontsize=9.7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("_", " ") for s in df.state])
    ax.set_xlim(0, 1.42)
    ax.set_xticks([0, .2, .4, .6, .8, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Decision accuracy")
    # panel title removed: the caption carries the statement
    ax.grid(axis="x", zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_ablation.png"), dpi=600)
    plt.close(fig)
    print("wrote fig_ablation.png")


# ---------------------------------------------------------------- anomaly
def fig_anomaly():
    a = json.load(open(os.path.join(RES, "v2_anomaly.json")))
    fig, axes = plt.subplots(1, 2, figsize=(6.90, 2.83))
    ax = axes[0]
    cls = ["NoGas", "Perfume", "Mixture", "Smoke"]
    # The partition-local fit, which is what every number in the paper uses. The
    # panel used to plot per_class_mean_recon, the whole-corpus fit, while the
    # prose beside it quoted these values: a figure and its own text disagreeing
    # about the very quantity the leakage argument turns on.
    pc = a["per_class_mean_recon_partition"]
    vals = [pc[c]["mean"] for c in cls]
    errs = [pc[c]["std"] for c in cls]
    b = ax.bar(np.arange(4), vals, 0.6, color=[BLUE, OLIVE, PURPLE, ORANGE],
               edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.errorbar(np.arange(4), vals, yerr=errs, fmt="none", ecolor=INK2,
                elinewidth=1.0, capsize=3.0, zorder=4)
    for bi, v, e in zip(b, vals, errs):
        ax.text(bi.get_x() + bi.get_width() / 2, (v + e) * 1.18,
                f"{v:,.3f}" if v < 1 else f"{v:,.1f}",
                ha="center", va="bottom", fontsize=9.5, color=INK2)
    ax.set_yscale("log"); ax.set_xticks(np.arange(4)); ax.set_xticklabels(cls)
    ax.set_ylabel("Mean reconstruction error (log)")
    ax.set_title("(a)", loc="left")
    ax.grid(axis="y", zorder=0)
    ax.set_ylim(5e-3, 4e3)

    ax = axes[1]
    # "as published" implied these were someone else's numbers; they are this
    # study's own earlier whole-corpus fit.
    labels = ["Whole-corpus fit\n(earlier implementation)", "Partition-local fit\n(this work)"]
    auc = [a["in_sample"]["auc"], a["auc_mean"]]
    tpr = [a["in_sample"]["tpr"], a["tpr_mean"]]
    fpr = [a["in_sample"]["fpr"], a["fpr_mean"]]
    x = np.arange(2); w = 0.26
    for i, (v, lab, col) in enumerate([(auc, "ROC-AUC", BLUE), (tpr, "TPR", GREEN), (fpr, "FPR", ORANGE)]):
        bb = ax.bar(x + (i - 1) * w, v, w * 0.9, color=col, label=lab,
                    edgecolor=SURFACE, linewidth=1.2, zorder=3)
        for bi, vv in zip(bb, v):
            ax.text(bi.get_x() + bi.get_width() / 2, vv + 0.018, f"{vv:.3f}",
                    ha="center", va="bottom", fontsize=9.3, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.34); ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylabel("Rate")
    ax.set_title("(b)", loc="left")
    ax.grid(axis="y", zorder=0)
    # Seated inside the headroom above the bars: anchored to the axes top it
    # printed over the uppermost tick label.
    ax.legend(fontsize=9.4, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0),
              columnspacing=1.2, handlelength=1.3, borderpad=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_anomaly.png"), dpi=600)
    plt.close(fig)
    print("wrote fig_anomaly.png")




# ======================= additional figures (v3) =======================
CRIT, WARN_, GOOD = "#a83226", "#bf8a12", "#1f7a52"   # status palette, validated


def fig_protocol():
    """Schematic of the corpus layout and the block-wise holdout."""
    fig, ax = plt.subplots(figsize=(6.90, 2.71))
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
        ax.text(-30, y, c, ha="right", va="center", fontsize=11.9, color=INK)
    ax.annotate("held-out block\n(20%, position drawn from the run seed)",
                (int(0.52 * n), 3.42), ha="center", fontsize=10.0, color=INK2)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#dfe6ec", label="training windows"),
                       Patch(color=WARN_, label="20-window embargo"),
                       Patch(color=BLUE, label="held-out test windows")],
              ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.52), fontsize=10.7)
    ax.set_xlim(-300, n + 30); ax.set_ylim(-0.55, 4.15)
    ax.set_xticks([0, 400, 800, 1200, 1581])
    ax.set_xlabel("window index within each class block (1,581 windows per class)")
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    # panel title removed: the caption carries the statement
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_protocol.png"), dpi=600); plt.close(fig)
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
    fig, axes = plt.subplots(1, 2, figsize=(6.90, 3.47))
    for pi, (ax, (cls, ycol, ylab, logy)) in enumerate(zip(
            axes, [("Smoke", "miss_rate_mean", "Miss rate,\nheld-out Smoke", True),
                   ("Mixture", "escalation_mean", "Escalation adequacy,\nheld-out Mixture", False)])):
        sub = l[l.held_out == cls]
        xs, ys, names = [], [], []
        for _, r in sub.iterrows():
            if r.model in z.index:
                xs.append(z.loc[r.model, "decision_acc_mean"]); ys.append(r[ycol]); names.append(str(KEY_OF[r.model]))
        floor = 8e-5
        yp = [max(v, floor) if logy else v for v in ys]
        ax.scatter(xs, yp, s=64, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        if logy:
            ax.set_yscale("log"); ax.set_ylim(floor * 0.30, 2.2)
            ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1]); ax.set_yticklabels(["0 (floor)", "0.1%", "1%", "10%"])
        else:
            ax.set_ylim(-0.20, 1.42); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        r = np.corrcoef(xs, ys)[0, 1]
        lo, hi = min(xs), max(xs)
        pad = 0.14 * (hi - lo) + 0.004
        ax.set_xlim(lo - pad, hi + pad)
        # Labels: points sharing a y-value (the zero floor, or full escalation)
        # would otherwise print on top of one another, so a tied group is
        # written vertically above its marker in left-to-right order instead.
        # transData returns display pixels while the label offsets are in points,
        # and mixing the two is what let two markers a fifth of an inch apart
        # test as "far enough" and print their labels on top of each other.
        # Each label is now placed by its own rectangle: the first free slot from
        # a list that tries the right of the marker, then the left, then higher
        # and lower, and the rectangles are tested for real overlap.
        _ppp = fig.dpi / 72.0                 # pixels per point
        inv = ax.transData.transform
        LFS = 9.4
        CAND = [(9.0, 5.0, "left"), (-9.0, 5.0, "right"),
                (9.0, 16.0, "left"), (-9.0, 16.0, "right"),
                (9.0, -10.0, "left"), (-9.0, -10.0, "right"),
                (9.0, 27.0, "left"), (-9.0, 27.0, "right"),
                (9.0, -21.0, "left"), (-9.0, -21.0, "right")]

        def _rect(px, py, dx, dy, txt, ha):
            w = 0.62 * LFS * len(txt) + 1.0
            x0 = px + dx if ha == "left" else px + dx - w
            return (x0, x0 + w, py + dy - 5.0, py + dy + 5.0)

        def _hits(r, others):
            return any(r[0] < o[1] + 2 and o[0] < r[1] + 2
                       and r[2] < o[3] + 2 and o[2] < r[3] + 2 for o in others)

        _bb = ax.get_window_extent()
        _ylo, _yhi = _bb.y0 / _ppp, _bb.y1 / _ppp     # axes bounds, in points
        placed = []
        for i in np.argsort(xs):
            _p = inv((xs[i], yp[i]))
            px, py = _p[0] / _ppp, _p[1] / _ppp
            dx, dy, ha = CAND[0]
            for cdx, cdy, cha in CAND:
                if not (_ylo + 3.0 <= py + cdy <= _yhi - 9.0):
                    continue                  # would sit outside the axes
                r = _rect(px, py, cdx, cdy, names[i], cha)
                if not _hits(r, placed):
                    dx, dy, ha = cdx, cdy, cha
                    break
            # A label pushed well clear of its marker is ambiguous on a crowded
            # floor, so anything past the first two slots gets a leader.
            _lead = dict(arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                                         shrinkA=1.0, shrinkB=3.0)) if abs(dy) > 18 else {}
            ax.annotate(names[i], (xs[i], yp[i]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=LFS, color=INK2, ha=ha,
                        **_lead)
            placed.append(_rect(px, py, dx, dy, names[i], ha))
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("In-distribution decision accuracy")
        ax.set_ylabel(ylab, fontsize=10.2, labelpad=2.0, linespacing=1.25)
        ax.set_title(f"({'ab'[pi]})", loc="left")
        ax.grid(zorder=0)
    # figure title removed: journals carry the statement in the caption
    # A no-break space inside each entry keeps the wrap between entries, so a
    # key never breaks across lines as "10" then "Shallow tree".
    key = "   ".join("%d\u00a0%s" % (KEY_OF[m], PRETTY[m].replace(" ", "\u00a0"))
                     for m in KEYED)
    nk = max(40, int(fig.get_size_inches()[0] * 72 / (8.4 * 0.50)))
    key = textwrap.fill(key, nk)
    nlines = key.count("\n") + 1
    fig.text(0.005, 0.012, key, ha="left", va="bottom", fontsize=8.4, color=INK2,
             linespacing=1.35)
    fig.tight_layout(rect=(0, 0.045 + 0.042 * nlines, 1, 0.94))
    fig.savefig(os.path.join(FIG, "fig_dissociation.png"), dpi=600); plt.close(fig)
    print("wrote fig_dissociation.png")


def fig_disposition():
    """Where hazardous windows actually go: Monitor / sub-alarm / alarm-grade."""
    l = pd.read_csv(os.path.join(RES, "v3_loco_all.csv"))
    models = [m for m in PRETTY if m in set(l.model)]
    fig, axes = plt.subplots(1, 2, figsize=(6.90, 3.45), sharey=True)
    for ci, (ax, cls) in enumerate(zip(axes, ["Smoke", "Mixture"])):
        sub = l[l.held_out == cls].set_index("model")
        miss = np.array([sub.loc[m, "miss_rate_mean"] for m in models])
        und = np.array([sub.loc[m, "under_escalation_mean"] for m in models])
        esc = np.array([sub.loc[m, "escalation_mean"] for m in models])
        x = np.arange(len(models))
        ax.bar(x, esc, 0.62, color=GOOD, label="Alarm-grade (a ≥ 3)", ec=SURFACE, lw=1.4, zorder=3)
        ax.bar(x, und, 0.62, bottom=esc, color=WARN_, label="Sub-alarm (a ∈ {1,2})", ec=SURFACE, lw=1.4, zorder=3)
        ax.bar(x, miss, 0.62, bottom=esc + und, color=CRIT, label="Monitor (a = 0)", ec=SURFACE, lw=1.4, zorder=3)
        for xi, e, u in zip(x, esc, und):
            # Set upright: horizontal labels were wider than the bars, so the
            # white type ran off the coloured segment and vanished on the page.
            if e > 0.10:
                ax.text(xi, e / 2, f"{e:.2f}", ha="center", va="center", fontsize=8.8,
                        color="white", rotation=90)
            if u > 0.14:
                ax.text(xi, e + u / 2, f"{u:.2f}", ha="center", va="center", fontsize=8.8,
                        color="white", rotation=90)
        ax.set_xticks(x); ax.set_xticklabels([PRETTY[m] for m in models], rotation=28, ha="right")
        ax.set_ylim(0, 1.03); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f"({'ab'[ci]})", loc="left")
        ax.grid(axis="y", zorder=0)
    axes[0].set_ylabel("Share of hazardous windows", labelpad=1.5)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], lb[::-1], ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.955), fontsize=11.3)
    # figure title removed: journals carry the statement in the caption
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(FIG, "fig_disposition.png"), dpi=600); plt.close(fig)
    print("wrote fig_disposition.png")


def fig_alarmburden():
    """False-alarm rate expressed as alarms per hour per detector."""
    p = pd.read_csv(os.path.join(RES, "v2_perturb.csv"))
    models = [m for m in ["A_cost_weighted", "D_mlp", "E_gbm", "RF", "ThresholdRule"] if m in set(p.model)]
    fig, ax = plt.subplots(figsize=(6.90, 3.54))
    for i, m in enumerate(models):
        sub = p[(p.model == m) & (p.perturbation.isin(["dropout", "clean"]))].sort_values("level")
        y = np.maximum(sub.fa_per_clean_mean.values * 1000.0, 0.03)
        ax.plot(sub.level, y, "-o", ms=5.5, lw=2, color=CAT[i], label=PRETTY[m], zorder=3)
    # The EEMUA reference line is deliberately absent. That figure describes what
    # a whole operator position can absorb, while this axis is one detector's
    # indication density, so a line across the panel invites a comparison the two
    # quantities do not support. The conversion and its caveats are in the text.
    ax.set_yscale("log"); ax.set_ylim(0.025, 2600)
    ax.set_yticks([0.05, 1, 10, 100, 1000])
    ax.set_yticklabels(["0 (floor)", "1", "10", "100", "1,000"])
    ax.set_xlabel("Sensor channels removed (k of 7)")
    ax.set_ylabel("High-severity alarms per\n1,000 clean windows (log)",
                  fontsize=10.2, labelpad=2.0, linespacing=1.25)
    # panel title removed: the caption carries the statement
    ax.grid(axis="y", zorder=0); ax.legend(fontsize=10.8, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_alarmburden.png"), dpi=600); plt.close(fig)
    print("wrote fig_alarmburden.png")




# ---------------- pipeline schematic ----------------
# ---------------- perception (replotted from reported results) ----------------
PERC_LABELS = ["Mixture", "NoGas", "Perfume", "Smoke"]
PERC_CM = np.array([[320, 0, 0, 0], [0, 308, 9, 3], [0, 3, 317, 0], [0, 0, 0, 320]], float)
PERC_PRF = {"Mixture": (1.000, 1.000, 1.000), "NoGas": (0.990, 0.963, 0.976),
            "Perfume": (0.972, 0.991, 0.981), "Smoke": (0.991, 1.000, 0.995)}


def fig_perception():
    # Three panels across the text width leave each about 2 in, so the cell
    # annotations are set small and the zero cells are left blank rather than
    # printed as "0.000", which is what used to run the rows together.
    # Two rows. Three panels across the text width left each confusion matrix
    # about 2 in, where a four-by-four grid of "0.000" cells runs together.
    fig = plt.figure(figsize=(6.90, 5.60))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.86], hspace=0.62, wspace=0.30)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :])]
    for ax, (M, ttl, fmt) in zip(axes[:2],
                                 [(PERC_CM, "(a)", "{:.0f}"),
                                  (PERC_CM / PERC_CM.sum(1, keepdims=True),
                                   "(b)", "{:.3f}")]):
        im = ax.imshow(M / M.max(), cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                v = M[i, j]
                if v <= 0:
                    continue
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9.0,
                        color="white" if v / M.max() > 0.5 else INK2)
        ax.set_xticks(range(4))
        ax.set_xticklabels(PERC_LABELS, rotation=34, ha="right", fontsize=9.5,
                           rotation_mode="anchor")
        ax.set_yticks(range(4)); ax.set_yticklabels(PERC_LABELS, fontsize=9.5)
        ax.tick_params(length=0)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True", labelpad=2.0)
        ax.set_title(ttl, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax = axes[2]
    x = np.arange(4); w = 0.26
    for i, (lab, col) in enumerate([("Precision", BLUE), ("Recall", ORANGE), ("F1", GREEN)]):
        v = [PERC_PRF[c][i] for c in PERC_LABELS]
        bb = ax.bar(x + (i - 1) * w, v, w * 0.9, color=col, label=lab, ec=SURFACE, lw=1.2, zorder=3)
        for b_, vv in zip(bb, v):
            ax.text(b_.get_x() + b_.get_width() / 2, vv + 0.002, f"{vv:.3f}",
                    ha="center", va="bottom", fontsize=8.4, color=INK2, rotation=90)
    ax.set_ylim(0.94, 1.052); ax.set_xticks(x)
    ax.set_xticklabels(PERC_LABELS, rotation=34, ha="right", fontsize=9.5,
                       rotation_mode="anchor")
    ax.set_ylabel("Score\n(axis truncated at 0.94)", fontsize=10.2, labelpad=2.0,
                  linespacing=1.25)
    ax.set_title("(c)", loc="left")
    ax.grid(axis="y", zorder=0)
    ax.legend(fontsize=9.2, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.04),
              columnspacing=1.6, handlelength=1.4)
    note(fig, "Replotted from the reported perception results. The split is randomized over "
              "consecutive frames from one session, so these are an optimistic upper bound.",
         y=0.035)
    fig.savefig(os.path.join(FIG, "fig_perception.png"), dpi=600); plt.close(fig)
    print("wrote fig_perception.png")


# ---------------- edge latency ----------------
# ---------------- anomaly ROC + score distributions ----------------
def fig_roc():
    d = json.load(open(os.path.join(RES, "v2_roc.json")))
    a = json.load(open(os.path.join(RES, "v2_anomaly.json")))
    fig, axes = plt.subplots(1, 2, figsize=(6.90, 2.97))
    ax = axes[0]
    for i, r in enumerate(d["held_out"]):
        ax.plot(r["fpr"], r["tpr"], lw=1.5, color=BLUE, alpha=0.55, zorder=3,
                label="Held-out, per seed" if i == 0 else None)
    ax.plot(d["in_sample"]["fpr"], d["in_sample"]["tpr"], lw=2.2, color=ORANGE,
            zorder=4, label="In-sample (same windows)")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.0, ls=":", zorder=2)
    ax.scatter([a["in_sample"]["fpr"]], [a["in_sample"]["tpr"]], s=70, color=ORANGE,
               edgecolor=SURFACE, linewidth=1.4, zorder=6)
    ax.annotate(f"in-sample operating point\nFPR {a['in_sample']['fpr']:.3f}, TPR {a['in_sample']['tpr']:.3f}",
                (a["in_sample"]["fpr"], a["in_sample"]["tpr"]), textcoords="offset points",
                xytext=(74, -30), fontsize=9.3, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.scatter([a["fpr_mean"]], [a["tpr_mean"]], s=70, color=BLUE,
               edgecolor=SURFACE, linewidth=1.4, zorder=6)
    ax.annotate(f"held-out operating point\nFPR {a['fpr_mean']:.3f}, TPR {a['tpr_mean']:.3f}",
                (a["fpr_mean"], a["tpr_mean"]), textcoords="offset points",
                xytext=(66, -76), fontsize=9.3, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.04)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("(a)", loc="left")
    ax.grid(zorder=0)
    ax.legend(fontsize=8.4, loc="lower right", handlelength=1.3,
              labelspacing=0.35, borderpad=0.25, handletextpad=0.5)

    ax = axes[1]
    cls = ["NoGas", "Perfume", "Mixture", "Smoke"]
    cols = [BLUE, OLIVE, PURPLE, ORANGE]
    for c, col in zip(cls, cols):
        v = np.array(d["per_class_scores"][c], float)
        v = np.log10(np.clip(v, 1e-4, None))
        ax.hist(v, bins=60, histtype="stepfilled", alpha=0.55, color=col,
                label=c, zorder=3, linewidth=0)
    ax.set_xlabel("log₁₀ reconstruction error"); ax.set_ylabel("Windows")
    ax.set_title("(b)", loc="left")
    ax.grid(axis="y", zorder=0); ax.legend(fontsize=10.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_roc.png"), dpi=600); plt.close(fig)
    print("wrote fig_roc.png")


# ---------------- in-distribution action-selection matrices ----------------
def fig_actionmatrix():
    df = pd.read_csv(os.path.join(RES, "v2_action_matrix.csv"))
    models = [m for m in PRETTY if m in set(df.model)]
    gases = ["NoGas", "Smoke", "Mixture", "Perfume"]
    acts = ["0 Monitor", "1 Sample", "2 Verify", "3 Alarm", "4 ESD"]
    n = len(models)
    # Three columns rather than four: the four-column grid was authored 12.2 in
    # wide and Word scaled it to the column, which made every cell annotation
    # unreadable. Tick labels are carried on the outer panels only, so the cells
    # themselves stay as large as the page allows.
    ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.90, 2.02 * nrow))
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
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.6,
                            color="white" if M[i, j] > 0.55 else INK2)
        bottom = k >= n - ncol
        left = k % ncol == 0
        ax.set_xticks(range(5))
        ax.set_xticklabels(acts if bottom else [""] * 5, rotation=52, ha="right", fontsize=9.3)
        ax.set_yticks(range(4))
        ax.set_yticklabels(gases if left else [""] * 4, fontsize=9.3)
        ax.tick_params(length=0)
        ax.set_title(f"({'abcdefghijkl'[k]})  {PRETTY[m]}", loc="left", fontsize=10.0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for k in range(n, len(axes)):
        axes[k].axis("off")
    # figure title removed: journals carry the statement in the caption
    fig.tight_layout(h_pad=1.6, w_pad=1.1)
    fig.savefig(os.path.join(FIG, "fig_actionmatrix.png"), dpi=600); plt.close(fig)
    print("wrote fig_actionmatrix.png")


# ---------------- calibration ----------------
def fig_calibration():
    a = pd.read_csv(os.path.join(RES, "v2_calibration.csv"))
    sw = pd.read_csv(os.path.join(RES, "v2_calibration_binsweep.csv"))
    # Short tick labels: the counts are in the caption, and the long forms
    # collided once the base type was raised.
    name = {"raw_softmax": "Raw softmax", "mc_dropout_20": "MC dropout",
            "temp_scaling": "Temp. scaling", "deep_ensemble_5": "Deep ensemble"}
    order = ["raw_softmax", "mc_dropout_20", "temp_scaling", "deep_ensemble_5"]
    a = a.set_index("estimator").loc[order].reset_index()
    # Three panels across the text width leave about 2.1 in each, so the axis
    # label is set on two lines and the value labels upright.
    # Two rows, same reason as the perception figure: three panels across the
    # text width leave about 2 in each, and four estimator names do not fit
    # under a 2 in axis at readable type.
    fig = plt.figure(figsize=(6.90, 5.40))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.95, wspace=0.34,
                          top=0.90, bottom=0.075, left=0.105, right=0.985)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :])]
    ELAB = "Expected calibration\nerror"
    ax = axes[0]
    x = np.arange(len(a))
    b = ax.bar(x, a.ece_mean, 0.58, color=BLUE, ec=SURFACE, lw=1.2, zorder=3)
    ax.errorbar(x, a.ece_mean, yerr=[a.ece_mean - a.ece_lo, a.ece_hi - a.ece_mean],
                fmt="none", ecolor=INK2, elinewidth=1.1, capsize=3.4, zorder=4)
    for bi, v, hi in zip(b, a.ece_mean, a.ece_hi):
        ax.text(bi.get_x() + bi.get_width() / 2, hi + 0.0015, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8.4, color=INK2, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([name[e] for e in a.estimator], rotation=34,
                       ha="right", fontsize=9.5, rotation_mode="anchor")
    ax.set_ylim(0, float(a.ece_hi.max()) * 1.55)
    ax.set_ylabel(ELAB, fontsize=10.2, labelpad=2.0, linespacing=1.25)
    ax.set_title("(a)", loc="left", fontsize=11.9)
    ax.grid(axis="y", zorder=0)
    ax = axes[1]
    w = 0.38
    b1 = ax.bar(x - w / 2, a.ece_mean, w * 0.92, color=BLUE, label="All windows", ec=SURFACE, lw=1.2, zorder=3)
    b2 = ax.bar(x + w / 2, a.ece_danger_mean, w * 0.92, color=ORANGE, label="Hazardous windows only",
                ec=SURFACE, lw=1.2, zorder=3)
    for bb, vv in list(zip(b1, a.ece_mean)) + list(zip(b2, a.ece_danger_mean)):
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() * 1.25, f"{vv:.4f}",
                ha="center", va="bottom", fontsize=8.0, color=INK2, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([name[e] for e in a.estimator], rotation=34,
                       ha="right", fontsize=9.5, rotation_mode="anchor")
    # Log scale: the hazardous-window values are two hundred times smaller than
    # the all-window ones, so on a linear axis their bars and labels disappear
    # into the axis line.
    ax.set_yscale("log")
    ax.set_ylim(1e-4, float(a.ece_mean.max()) * 6.0)
    # Decade ticks only: the minor labels collided with the rotated names.
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylabel(ELAB, fontsize=10.2, labelpad=2.0, linespacing=1.25)
    ax.set_title("(b)", loc="left", fontsize=11.9)
    ax.grid(axis="y", zorder=0)
    _ch, _cl = ax.get_legend_handles_labels()
    fig.legend(_ch, _cl, fontsize=9.0, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, handlelength=1.3, borderpad=0.3, columnspacing=1.6)
    ax = axes[2]
    # The four traces fill the panel, so a legend box would have to sit on top of
    # one of them; each trace is named at its right-hand end instead.
    SHORT = {"raw_softmax": "Raw softmax", "mc_dropout_20": "MC dropout",
             "temp_scaling": "Temp. scaling", "deep_ensemble_5": "Deep ensemble"}
    DY = {"temp_scaling": 0.0011, "deep_ensemble_5": -0.0011}   # the two traces nearly coincide
    for i, e in enumerate(order):
        g = sw[sw.estimator == e].groupby("n_bins").ece.mean()
        ax.plot(g.index, g.values, "-o", ms=4.4, lw=1.8, color=CAT[i], zorder=3)
        ax.text(float(g.index[-1]) + 1.4, float(g.values[-1]) + DY.get(e, 0.0), SHORT[e],
                fontsize=8.6, color=CAT[i], va="center", ha="left", zorder=4)
    ax.set_xlim(3.5, 42.0)
    ax.set_xticks([5, 10, 15, 20, 30])
    ax.set_xlabel("Number of bins")
    ax.set_ylabel(ELAB, fontsize=10.2, labelpad=2.0, linespacing=1.25)
    ax.set_title("(c)", loc="left", fontsize=11.9)
    ax.grid(zorder=0)
    fig.savefig(os.path.join(FIG, "fig_calibration.png"), dpi=600); plt.close(fig)
    print("wrote fig_calibration.png")


# ---------------- anomaly-score influence on the action ----------------
def fig_anominfluence():
    d = pd.read_csv(os.path.join(RES, "v2_anomaly_influence.csv"))
    order = (d.groupby("model").frac_windows_action_changed.first()
             .sort_values(ascending=False).index.tolist())
    fig, axes = plt.subplots(1, 2, figsize=(6.90, 3.36),
                             gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    fr = [100 * d[d.model == m].frac_windows_action_changed.iloc[0] for m in order]
    sd = [100 * d[d.model == m].frac_changed_std.iloc[0] for m in order]
    x = np.arange(len(order))
    b = ax.bar(x, fr, 0.62, color=BLUE, ec=SURFACE, lw=1.2, zorder=3)
    ax.errorbar(x, fr, yerr=sd, fmt="none", ecolor=INK2, elinewidth=1.0, capsize=3, zorder=4)
    for bi, v, e in zip(b, fr, sd):
        ax.text(bi.get_x() + bi.get_width() / 2, v + e + 1.4, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=9.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[m] for m in order], rotation=28, ha="right")
    ax.set_ylim(0, max(fr) * 1.30)
    ax.yaxis.set_major_formatter(lambda v, pos: f"{v:.0f}%")
    ax.set_ylabel("Test windows whose\naction changes", fontsize=10.2,
                  labelpad=2.0, linespacing=1.25)
    ax.set_title("(a)", loc="left")
    ax.grid(axis="y", zorder=0)
    # in-panel note removed: the caption carries it (sweep note)

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
                textcoords="offset points", xytext=(10, -4), fontsize=10.0, color=CRIT)
    ax.annotate(f"{sub.miss_rate_mean.iloc[-1]:.4f}", (1.0, sub.miss_rate_mean.iloc[-1]),
                textcoords="offset points", xytext=(-6, 12), fontsize=10.0, color=CRIT, ha="right")
    ax.set_xlabel("Anomaly score forced to this value")
    ax.set_ylabel("Missed-hazard rate")
    ax.set_ylim(-0.02, 0.40)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("(b)", loc="left")
    ax.grid(zorder=0); ax.legend(fontsize=10.6, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_anominfluence.png"), dpi=600); plt.close(fig)
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
    fig, axes = plt.subplots(1, 2, figsize=(6.90, 2.89), gridspec_kw={"width_ratios": [1.35, 1]})
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
    ax.set_title("(a)", loc="left")
    ax.grid(axis="y", zorder=0)
    # The key belongs to both panels and is set across the figure: anchored to
    # panel (a) it covered that panel's title.
    _lh, _ll = ax.get_legend_handles_labels()

    ax = axes[1]
    gap = 100 * (piv["random"] - piv["blocked_embargo"])
    order = gap.sort_values(ascending=False).index.tolist()
    v = [gap[m] for m in order]
    b = ax.barh(np.arange(len(order)), v, 0.6,
                color=[ORANGE if x > 0 else GREEN for x in v], ec=SURFACE, lw=1.2, zorder=3)
    for bi, x in zip(b, v):
        ax.text(x + (0.10 if x >= 0 else -0.10), bi.get_y() + bi.get_height() / 2,
                f"{x:+.2f}", va="center", ha="left" if x >= 0 else "right",
                fontsize=9.7, color=INK2)
    ax.axvline(0, color=MUTED, lw=1.0, zorder=2)
    ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([PRETTY[m] for m in order])
    ax.set_xlim(-3.2, 7.4)
    ax.set_xlabel("Accuracy points added\nby a random split", fontsize=9.6, labelpad=2.0,
                  linespacing=1.25)
    ax.set_title("(b)", loc="left")
    ax.grid(axis="x", zorder=0)
    fig.legend(_lh, _ll, fontsize=9.2, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.4, handlelength=1.3,
               handletextpad=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    fig.savefig(os.path.join(FIG, "fig_leakage.png"), dpi=600); plt.close(fig)
    print("wrote fig_leakage.png")


# ---------------- powered comparison and equivalence ----------------
def fig_powered():
    d = pd.read_csv(os.path.join(RES, "v3_powered.csv")).set_index("model")
    b = json.load(open(os.path.join(RES, "v3_powered_bayes.json")))
    sig = b["vs_cost_weighted"]
    fig, axes = plt.subplots(1, 2, figsize=(6.90, 3.28), gridspec_kw={"width_ratios": [1.18, 1]})
    ax = axes[0]
    order = d.decision_acc_mean.sort_values(ascending=False).index.tolist()
    x = np.arange(len(order))
    v = [d.loc[m, "decision_acc_mean"] for m in order]
    e = [d.loc[m, "decision_acc_std"] for m in order]
    cols = [ORANGE if m == "A_cost_weighted" else ("#1f8f5f" if m == "Ordinal" else "#9fb4c4") for m in order]
    bb = ax.bar(x, v, 0.64, yerr=e, color=cols, ec=SURFACE, lw=1.2, zorder=3,
                error_kw=dict(ecolor=INK2, elinewidth=1.0, capsize=2.6))
    for bi, vv, ee in zip(bb, v, e):
        # Eleven comparators sit within four accuracy points of one another, so
        # horizontal value labels overlapped; they are set upright instead.
        ax.text(bi.get_x() + bi.get_width() / 2, vv + ee + 0.006, f"{vv:.3f}",
                ha="center", va="bottom", fontsize=8.4, color=INK2, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([PRETTY.get(m, m) for m in order], rotation=40,
                       ha="right", fontsize=10.2, rotation_mode="anchor")
    ax.set_ylim(0.70, 1.10); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Decision accuracy", labelpad=1.5)
    ax.set_title("(a)", loc="left")
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
            # Set to the bar height: at the larger size the digits stood a little
            # proud of the band they sit in.
            ax.text(left[yi] + m / 2, yi, f"{m:.2f}", ha="center", va="center",
                    fontsize=8.4, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([PRETTY.get(m, m) for m in names], fontsize=10.2)
    ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Posterior probability")
    ax.set_title("(b)", loc="left")
    ax.grid(axis="x", zorder=0)
    h_, l_ = ax.get_legend_handles_labels()
    fig.legend(h_, l_, fontsize=9.7, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.4, handlelength=1.5)
    fig.tight_layout(rect=(0, 0.095, 1, 1), w_pad=2.4)
    fig.savefig(os.path.join(FIG, "fig_powered.png"), dpi=600); plt.close(fig)
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

def _rbox(ax, x, y, w, h, txt, fc, ec, fs=9.7, tc=INK, lw=1.4, ls="-"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.09",
                                facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3, linespacing=1.32)


def _arrow(ax, p, q, col=MUTED, lw=1.3, style="-|>"):
    ax.annotate("", q, p, arrowprops=dict(arrowstyle=style, color=col, lw=lw,
                                          shrinkA=3, shrinkB=3), zorder=1)


def fig_workflow():
    """Fig. 1, built around the paper's argument rather than its contents list.

    Four bands of boxes, however arranged, only inventory the study. The shape
    here is the claim: everything passes through one narrow protocol gate, drawn
    inset from the page so it reads as a constriction, and the same trained
    models then fan out into five conditions that make the five metrics
    disagree. The two connecting arrows carry what they guarantee, so the figure
    states an argument rather than a sequence.
    """
    from retrain import spe_flow as S
    with plt.rc_context(S.RC):          # a serif figure, sealed off from the plots
        fh, fsn, fss = 11.8, 8.2, 7.0
        h1 = 3.4 + 2.6 + fh + S.caption_h(fsn, fss) + 1.5
        ph, chip_h = 8.0, 11.6
        h2 = 3.4 + ph + 3.0 + chip_h + 1.8
        GAP1, FAN, rh, mh = 7.4, 13.4, 7.2, 3.6
        FOOT = 2.6              # the line under the metric bar
        H = 1.2 + FOOT + h1 + GAP1 + h2 + FAN + rh + 2.8 + mh + 1.5
        fig, ax = S.canvas(H)

        # ---- what the corpus holds, and what each class is worth -------------
        p1 = (1.0, H - 1.2 - h1, 98.0, h1)
        S.panel(ax, *p1, "The corpus, and the action each class maps to")
        fy = p1[1] + h1 - 3.4 - 2.6 - fh
        S.frames(ax, 2.4, fy, 54.0, fh, fsn, fss)
        S.trace(ax, fig, 62.0, fy, 34.0, fh, H, fs=fsn, subs=2)

        # ---- the single gate every comparator passes through -----------------
        gx, gw = 9.0, 82.0
        p2 = (gx, p1[1] - GAP1 - h2, gw, h2)
        S.panel(ax, *p2, "One protocol, and every comparator trained under it")
        bw = (gw - 3.6 - 2 * 3.4) / 3.0
        bx, by = gx + 1.8, p2[1] + h2 - 3.4 - ph
        S.box(ax, bx, by, bw, ph, "Windowing\nlength 20\nboundary windows dropped",
              S.BLU_F, S.BLU_E, 7.4)
        S.barrow(ax, bx + bw + 1.7, by + ph / 2, 2.8, 3.6, "right")
        S.box(ax, bx + bw + 3.4, by, bw, ph,
              "Block-wise holdout\ncontiguous 20% per class, 20-window\n"
              "embargo, seed-varying position", S.BLU_F, S.BLU_E, 7.4)
        S.barrow(ax, bx + 2 * bw + 5.1, by + ph / 2, 2.8, 3.6, "right")
        S.cyl(ax, bx + 2 * (bw + 3.4), by, bw, ph,
              "4,900 train / 1,264 test\n316 windows per class", 7.4)
        S.chips(ax, gx + 1.8, p2[1] + 1.8, gw - 3.6, chip_h, S.COMPARATORS, 6,
                S.PCH_F, S.PCH_E)

        S.barrow(ax, 50.0, (p1[1] + p2[1] + h2) / 2.0,
                 p1[1] - (p2[1] + h2) - 0.7, 5.2, "down")
        ax.text(53.4, (p1[1] + p2[1] + h2) / 2.0,
                "no training window shares a raw row with a test window",
                fontsize=7.6, style="italic", ha="left", va="center", color="#333333")

        # ---- the fan into five conditions, and the one metric bar ------------
        rw, rgap = (94.0 - 4 * 1.4) / 5.0, 1.4
        ry = p2[1] - FAN - rh
        rfs = S.fit_fs(ax, [a + "\n" + b for a, b in S.REGIMES], rw, start=7.4,
                       floor=5.0, margin=0.5)
        apex = (50.0, p2[1] - 0.5)
        for i, (head, sub) in enumerate(S.REGIMES):
            rx = 3.0 + i * (rw + rgap)
            ax.add_patch(FancyArrowPatch(apex, (rx + rw / 2, ry + rh + 0.25),
                                         arrowstyle="-|>", mutation_scale=8.5,
                                         color=S.INK, lw=0.9, zorder=2,
                                         shrinkA=1.5, shrinkB=0.0))
            S.box(ax, rx, ry, rw, rh, head + "\n" + sub, S.GRN_F, S.GRN_E, rfs)
            S.tarrow(ax, (rx + rw / 2, ry - 0.2), (rx + rw / 2, ry - 2.3))
        ax.text(50.0, p2[1] - 3.6, "the same trained models, one fit per condition",
                fontsize=7.8, style="italic", ha="center", va="center",
                color="#333333", zorder=6,
                bbox=dict(boxstyle="round,pad=0.30", fc="#ffffff", ec="none"))
        S.metric_bar(ax, 3.0, ry - 2.8 - mh, 94.0, mh)   # plus its footnote

        fig.savefig(os.path.join(FIG, "fig_workflow.png"), dpi=600)
        plt.close(fig)
        print("wrote fig_workflow.png")


def fig_decisionlogic():
    """Fig. 3: one window, one action, and where each metric reads the ladder."""
    from retrain import spe_style as S
    with spe_scope():
        S.use_style()
        fig, ax = S.canvas(S.FULL, S.FULL * 0.385)
        H = ax.get_ylim()[1]

        S.group(ax, 1.0, 2.0, 98.0, H - 3.6, label="Scoring one window", ec=S.GREY_E)

        yc = H * 0.52
        S.box(ax, 3.0, yc - 4.4, 12.6, 8.8, "Sensor window\n$X_t$", fc=S.GREY_F,
              ec=S.GREY_E, fs=7.0)
        S.arrow(ax, (15.6, yc), (18.6, yc))
        S.box(ax, 18.6, yc - 4.4, 14.0, 8.8, "Decision state\n$\\varphi_t \\in \\mathbb{R}^{22}$",
              fc=S.BLUE_F, ec=S.BLUE_E, fs=7.0)
        S.arrow(ax, (32.6, yc), (35.6, yc))
        S.box(ax, 35.6, yc - 4.4, 14.0, 8.8, "Selected action\n$a_t \\in \\{0,\\dots,4\\}$",
              fc=S.MINT_F, ec=S.MINT_E, fs=7.0)
        S.arrow(ax, (49.6, yc), (52.4, yc))

        # The ladder and the metric block to its right are sized from the text
        # they have to hold, measured at the chosen type size, not from a
        # character count. The type steps down only if the measured block will
        # not fit between the chain of boxes and the panel edge.
        rungs = [(4, "Recommend ESD assessment"), (3, "Raise alarm"),
                 (2, "Request verification"), (1, "Increase sampling"), (0, "Monitor")]
        BR_TITLES = ["Escalation adequacy", "High-severity false alarm",
                     "Under-escalation", "Missed hazard"]
        BR_SUBS = ["hazardous windows at $a \\geq 3$", "clean windows at $a \\geq 3$",
                   "hazardous windows at $a \\in \\{1,2\\}$",
                   "hazardous windows at $a = 0$"]
        LEFT_EDGE, RIGHT_EDGE, BR_GAP, BR_PAD = 53.6, 97.2, 2.6, 1.2
        for _scale in (1.0, 0.95, 0.90, 0.85, 0.80):
            RFS, BFS_T, BFS_S = 6.8 * _scale, 6.9 * _scale, 6.1 * _scale
            lw = max(S.text_width(ax, f"{a_}    {lab}", RFS)
                     for a_, lab in rungs) + 2.6
            br_w = max([S.text_width(ax, s, BFS_T, fontweight="bold") for s in BR_TITLES]
                       + [S.text_width(ax, s, BFS_S) for s in BR_SUBS])
            lx = RIGHT_EDGE - (lw + BR_GAP + BR_PAD + br_w)
            if lx >= LEFT_EDGE:
                break
        rh, gap, bgap = 4.0, 0.55, 4.2      # bgap: a clear band for the boundary rule
        ys = {}
        y = H - 6.2 - rh
        for a_, lab in rungs:
            if a_ == 2:
                y -= bgap
            ys[a_] = y
            fc = S.MINT_F if a_ >= 3 else (S.SAND_F if a_ >= 1 else S.ROSE_F)
            ec = S.MINT_E if a_ >= 3 else (S.SAND_E if a_ >= 1 else S.ROSE_E)
            S.box(ax, lx, y, lw, rh, f"{a_}    {lab}", fc=fc, ec=ec, fs=RFS)
            y -= rh + gap

        yb = (ys[3] + ys[2] + rh) / 2
        ax.plot([lx - 1.6, lx + lw + 0.8], [yb, yb], color=S.INK2, lw=1.0,
                ls=(0, (4.0, 2.6)), zorder=8)
        # Short label, set inside the band and over the ladder. The full sentence
        # ran the width of two rungs and collided with the box chain on one side
        # and the metric brackets on the other; the body carries it instead.
        ax.text(lx + lw / 2, yb + 0.45, "alarm boundary", fontsize=RFS,
                color=S.INK2, style="italic", ha="center", va="bottom", zorder=8)

        # brackets: which rungs each metric reads
        bx = lx + lw + BR_GAP
        def bracket(y0, y1, col, label, sub):
            ax.plot([bx, bx], [y0, y1], color=col, lw=1.6, solid_capstyle="round", zorder=6)
            for yy in (y0, y1):
                ax.plot([bx - 0.8, bx], [yy, yy], color=col, lw=1.6, zorder=6)
            ax.text(bx + BR_PAD, (y0 + y1) / 2 + 0.95, label, fontsize=BFS_T, color=S.INK,
                    va="center", zorder=6, fontweight="bold")
            ax.text(bx + BR_PAD, (y0 + y1) / 2 - 0.95, sub, fontsize=BFS_S, color=S.MUTED,
                    va="center", zorder=6)
        _esc_c = (ys[3] + ys[4] + rh) / 2
        ax.plot([bx, bx], [ys[3], ys[4] + rh], color=S.MINT_E, lw=1.6,
                solid_capstyle="round", zorder=6)
        for _yy in (ys[3], ys[4] + rh):
            ax.plot([bx - 0.8, bx], [_yy, _yy], color=S.MINT_E, lw=1.6, zorder=6)
        # Two metrics read these same two rungs: one on hazardous windows and one
        # on clean windows. They are stacked in the bracket rather than written
        # over it, which is what used to collide.
        ax.text(bx + BR_PAD, _esc_c + 3.0, "Escalation adequacy", fontsize=BFS_T,
                color=S.INK, va="center", zorder=6, fontweight="bold")
        ax.text(bx + BR_PAD, _esc_c + 1.2, "hazardous windows at $a \\geq 3$",
                fontsize=BFS_S, color=S.MUTED, va="center", zorder=6)
        ax.text(bx + BR_PAD, _esc_c - 1.2, "High-severity false alarm", fontsize=BFS_T,
                color=S.BLUE_E, va="center", zorder=6, fontweight="bold")
        ax.text(bx + BR_PAD, _esc_c - 3.0, "clean windows at $a \\geq 3$",
                fontsize=BFS_S, color=S.BLUE_E, va="center", zorder=6)
        bracket(ys[1], ys[2] + rh, S.SAND_E, "Under-escalation", "hazardous windows at $a \\in \\{1,2\\}$")
        bracket(ys[0], ys[0] + rh, S.ROSE_E, "Missed hazard", "hazardous windows at $a = 0$")

        fig.savefig(os.path.join(FIG, "fig_decisionlogic.png"), dpi=600)
        plt.close(fig)
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
    fig, ax = plt.subplots(figsize=(6.90, 3.66))
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
                  fs=9.3 if len(show) < 16 else 7.4, tc=tc, lw=1.2)
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
              ncol=3, loc="lower center", bbox_to_anchor=(0.50, -0.15), fontsize=8.8,
              columnspacing=0.9, handlelength=1.2, handletextpad=0.5)
    # The row labels are drawn to the left of x = 0, so the left limit has to
    # hold the longest of them: anything that spills widens the saved figure.
    ax.set_xlim(-4.7, 10.0); ax.set_ylim(-0.62, nr + 0.8); ax.axis("off")
    fig.tight_layout()
    # The axes carries no ticks, so it can run to the edges of the canvas; left
    # to itself the layout inset it and the saved figure came out narrow.
    fig.subplots_adjust(left=0.004, right=0.996, top=0.985, bottom=0.015)
    fig.savefig(os.path.join(FIG, "fig_metricdisagreement.png"), dpi=600); plt.close(fig)
    print("wrote fig_metricdisagreement.png")
