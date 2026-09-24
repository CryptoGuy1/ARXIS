"""Component architecture figures, drawn in the model papers' idiom.

Three components are named in the manuscript but never drawn: the LSTM
autoencoder that produces the anomaly feature, the dueling network trained with
cost-weighted cross-entropy that selects the action, and the YOLOv8-cls
classifier on the thermal path. Each figure shows what the component actually
does to a window, with the tensor shapes the code produces, so a reader can
check the architecture against Supplementary Sections S1 and S5.

Every shape here is read off the implementation, not invented: the encoder and
decoder are single-layer with hidden dimension 32 (retrain/raw_pipeline.py), the
dueling head splits into a scalar value stream and a five-way advantage stream
over the 22-dimensional decision state (retrain/agent_rl.py), and the thermal
classifier is YOLOv8n-cls over four classes at 224 by 224.

    python3 -m retrain.make_figs_arch
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle

from retrain import spe_style as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures_v2")


# ---------------------------------------------------------------- LSTM AE
def fig_lstm():
    """Anomaly path: a window in, a reconstruction out, the residual as ρ̃."""
    S.use_style()
    fig, ax = S.canvas(S.FULL, S.FULL * 0.345)
    H = ax.get_ylim()[1]

    S.group(ax, 1.5, 3.0, 97.0, H - 5.0, label="Anomaly component", ec=S.BLUE_E)

    yc = H * 0.545
    # input window
    S.slab(ax, 4.5, yc - 6.0, 6.2, 12.0, n=3, fc=S.BLUE_F, ec=S.BLUE_E,
           label="$X_t$\n20 × 7")

    S.arrow(ax, (13.0, yc), (16.6, yc))

    # encoder
    S.box(ax, 16.6, yc - 5.2, 12.4, 10.4, "LSTM encoder\n1 layer, 32 units",
          fc=S.MINT_F, ec=S.MINT_E, fs=7.0)
    S.arrow(ax, (29.0, yc), (32.6, yc))

    # latent
    S.slab(ax, 32.6, yc - 3.2, 3.4, 6.4, n=2, fc="#e6d9ef", ec="#7a4f9a",
           label="$h$\n32")
    S.arrow(ax, (37.4, yc), (41.0, yc))

    # decoder
    S.box(ax, 41.0, yc - 5.2, 12.4, 10.4, "LSTM decoder\n1 layer, 32 units",
          fc=S.MINT_F, ec=S.MINT_E, fs=7.0)
    S.arrow(ax, (53.4, yc), (57.0, yc))

    # reconstruction
    S.slab(ax, 57.0, yc - 6.0, 6.2, 12.0, n=3, fc=S.GREY_F, ec=S.GREY_E,
           label=r"$\hat{X}_t$" + "\n20 × 7")

    # residual: the input window is differenced against its reconstruction
    S.arrow(ax, (65.5, yc), (69.2, yc))
    S.op(ax, 70.6, yc, "−", r=1.15)
    y_top = yc + 10.2
    ax.plot([7.6, 7.6], [yc + 7.0, y_top], color=S.INK2, lw=0.9, zorder=5,
            solid_capstyle="round")
    ax.plot([7.6, 70.6], [y_top, y_top], color=S.INK2, lw=0.9, zorder=5,
            solid_capstyle="round")
    S.arrow(ax, (70.6, y_top), (70.6, yc + 1.2))

    S.arrow(ax, (72.0, yc), (75.6, yc))
    S.box(ax, 75.6, yc - 4.4, 10.6, 8.8, "Mean squared\nerror over\n20 steps",
          fc=S.SAND_F, ec=S.SAND_E, fs=6.8)
    S.arrow(ax, (86.2, yc), (89.6, yc))
    S.box(ax, 89.6, yc - 3.0, 6.6, 6.0, r"$\tilde{\rho}_t$", fc=S.ROSE_F,
          ec=S.ROSE_E, fs=9.0, bold=True)

    # the two things a reader needs to know about how it is fitted
    ax.text(50.0, 6.3, "fitted on clean-air windows of the training partition only;  "
                       r"$\tilde{\rho}_t$ min-max scaled to [0, 1] on the 1st and 99th training percentiles",
            fontsize=6.7, color=S.MUTED, ha="center", va="center", zorder=9)
    return S.save(fig, os.path.join(FIG, "fig_arch_lstm.png"))


# ---------------------------------------------------------------- dueling net
def fig_dqn():
    """Decision path: 22-dimensional state to one of five actions."""
    S.use_style()
    fig, ax = S.canvas(S.FULL, S.FULL * 0.40)
    H = ax.get_ylim()[1]

    S.group(ax, 1.5, 3.0, 97.0, H - 5.0, label="Decision component", ec=S.MINT_E)

    yc = H * 0.555
    # decision state, drawn as the 22 cells grouped 7/7/7/1
    x = 4.6
    cw, gap = 0.95, 0.30
    groups = [("current", 7, S.BLUE_F, S.BLUE_E), ("Δ", 7, S.MINT_F, S.MINT_E),
              ("SD", 7, "#e4e0c8", "#8a7a1a"), (r"$\tilde{\rho}$", 1, S.ROSE_F, S.ROSE_E)]
    for name, n, fc, ec in groups:
        gx0 = x
        for _ in range(n):
            ax.add_patch(Rectangle((x, yc - 5.4), cw, 10.8, fc=fc, ec=ec,
                                   lw=0.6, zorder=4))
            x += cw + 0.16
        ax.text((gx0 + x - 0.16) / 2, yc - 6.2, name, fontsize=6.3,
                color=S.INK2, ha="center", va="top", zorder=6)
        x += gap
    x_end = x - gap
    ax.text((4.6 + x_end) / 2, yc + 6.6, r"$\varphi_t \in \mathbb{R}^{22}$",
            fontsize=8.0, color=S.INK, ha="center", va="bottom", zorder=6)

    S.arrow(ax, (x_end + 0.6, yc), (x_end + 4.2, yc))

    # shared trunk
    tx = x_end + 4.2
    S.box(ax, tx, yc - 5.0, 11.0, 10.0, "Shared trunk\n2 × 128, ReLU",
          fc=S.MINT_F, ec=S.MINT_E, fs=7.0)

    # split into value and advantage
    bx = tx + 11.0
    ax.plot([bx, bx + 3.0], [yc, yc], color=S.INK2, lw=0.9, zorder=5)
    ax.plot([bx + 3.0, bx + 3.0], [yc - 7.4, yc + 7.4], color=S.INK2, lw=0.9, zorder=5)
    S.arrow(ax, (bx + 3.0, yc + 7.4), (bx + 6.6, yc + 7.4))
    S.arrow(ax, (bx + 3.0, yc - 7.4), (bx + 6.6, yc - 7.4))

    # One size for both streams, taken from the wider of the two labels: set by
    # eye, "Advantage stream" ran out of its box.
    _vs = "Value stream\n$V(\\varphi)$, 1 unit"
    _as = "Advantage stream\n$A(\\varphi,a)$, 5 units"
    _sfs = S.fit_fontsize(ax, [_vs, _as], 11.6, start=6.8, floor=5.2, margin=0.5)
    S.box(ax, bx + 6.6, yc + 3.8, 11.6, 7.2, _vs, fc=S.BLUE_F, ec=S.BLUE_E, fs=_sfs)
    S.box(ax, bx + 6.6, yc - 11.0, 11.6, 7.2, _as, fc=S.BLUE_F, ec=S.BLUE_E, fs=_sfs)

    # recombine
    rx = bx + 18.2
    ax.plot([rx, rx + 2.6], [yc + 7.4, yc + 7.4], color=S.INK2, lw=0.9, zorder=5)
    ax.plot([rx, rx + 2.6], [yc - 7.4, yc - 7.4], color=S.INK2, lw=0.9, zorder=5)
    ax.plot([rx + 2.6, rx + 2.6], [yc - 7.4, yc + 7.4], color=S.INK2, lw=0.9, zorder=5)
    S.op(ax, rx + 4.2, yc, "+", r=1.15)
    ax.plot([rx + 2.6, rx + 3.05], [yc, yc], color=S.INK2, lw=0.9, zorder=5)

    S.arrow(ax, (rx + 5.4, yc), (rx + 8.6, yc))
    S.box(ax, rx + 8.6, yc - 5.0, 10.2, 10.0,
          "Action scores\n$Q(\\varphi, a)$", fc=S.SAND_F, ec=S.SAND_E, fs=6.9)

    # the ladder
    lx = rx + 19.4
    S.arrow(ax, (rx + 18.8, yc), (lx - 0.2, yc))
    labels = ["4  ESD assess", "3  Alarm", "2  Verify", "1  Sample", "0  Monitor"]
    cmap = plt.cm.YlOrRd
    BGAP = 3.6                      # extra room so the boundary label has its own band
    ys = {}
    for i, lab in enumerate(labels):
        yy = yc + 6.9 - i * 3.1 - (BGAP if i >= 2 else 0.0)
        ys[i] = yy
        ax.add_patch(Rectangle((lx, yy - 1.25), 2.0, 2.5,
                               fc=cmap(0.18 + 0.62 * (4 - i) / 4.0), ec="none", zorder=4))
        ax.text(lx + 2.7, yy, lab, fontsize=6.5, color=S.INK, va="center", zorder=6)
    # alarm boundary, in the gap the spacing above reserves for it
    yb = (ys[1] - 1.25 + ys[2] + 1.25) / 2
    # The rule starts clear of the action-scores box: drawn from its edge it read
    # as part of the box, and the label sat on top of the "3  Alarm" row.
    ax.plot([lx - 0.2, lx + 9.4], [yb, yb], color=S.INK2, lw=0.9,
            ls=(0, (3.2, 2.2)), zorder=7)
    ax.text(lx + 9.4, yb + 0.45, "alarm boundary", fontsize=6.1, color=S.INK2,
            va="bottom", ha="right", zorder=7, style="italic")

    ax.text(50.0, 6.0, "trained with cost-weighted cross-entropy on the target action of Eq. 2a; "
                       "no reward, no bootstrap, no environment",
            fontsize=6.7, color=S.MUTED, ha="center", va="center", zorder=9)
    return S.save(fig, os.path.join(FIG, "fig_arch_dqn.png"))


# ---------------------------------------------------------------- YOLOv8-cls
def fig_yolo():
    """Thermal path: a 224 by 224 frame to a four-class posterior."""
    S.use_style()
    fig, ax = S.canvas(S.FULL, S.FULL * 0.335)
    H = ax.get_ylim()[1]

    S.group(ax, 1.0, 3.0, 98.0, H - 5.0, label="Perception\ncomponent", ec="#7a4f9a",
            label_indent=1.9)

    yc = H * 0.545

    # a real released thermal frame, not a drawn placeholder
    src = os.path.join(ROOT, "data", "Mixture", "26_Mixture.png")
    if os.path.exists(src):
        import matplotlib.image as mpimg
        im = mpimg.imread(src)
        ax.imshow(im, extent=[4.4, 11.8, yc - 7.2, yc + 7.2], aspect="auto",
                  zorder=4, interpolation="lanczos")
        ax.add_patch(Rectangle((4.4, yc - 7.2), 7.4, 14.4, fc="none",
                               ec="#7a4f9a", lw=0.9, zorder=6))
    else:
        S.slab(ax, 4.4, yc - 7.2, 7.4, 14.4, n=1, fc="#2b2f6b", ec="#7a4f9a")
    ax.text(8.1, yc - 8.0, "$I_t$, released frame,\nresized to 224 × 224",
            fontsize=6.0, color=S.INK2, ha="center", va="top", zorder=7,
            linespacing=1.2)

    S.arrow(ax, (12.4, yc), (15.6, yc))

    # backbone: CSP stages with decreasing spatial size
    stages = [("Stem\nConv", 6.6, 12.6), ("C2f\n×1", 6.0, 11.0),
              ("C2f\n×2", 5.6, 9.4), ("C2f\n×2", 5.2, 7.8), ("SPPF", 5.0, 6.4)]
    x = 15.6
    for i, (lab, w, h) in enumerate(stages):
        S.box(ax, x, yc - h / 2, w, h, lab, fc=S.MINT_F, ec=S.MINT_E, fs=6.4)
        if i < len(stages) - 1:
            S.arrow(ax, (x + w, yc), (x + w + 2.0, yc))
        x += w + 2.0
    S.span(ax, 15.6, x - 2.0, yc + 9.4, "Backbone", color=S.MINT_E, fs=6.9, drop=-2.4)

    S.arrow(ax, (x - 2.0, yc), (x + 1.2, yc))
    x += 1.2

    S.box(ax, x, yc - 4.2, 9.2, 8.4, "Global\naverage pool", fc=S.BLUE_F,
          ec=S.BLUE_E, fs=6.7)
    S.arrow(ax, (x + 9.2, yc), (x + 12.4, yc))
    x += 12.4
    S.box(ax, x, yc - 4.2, 8.0, 8.4, "Linear\n1280 → 4", fc=S.BLUE_F,
          ec=S.BLUE_E, fs=6.7)
    S.arrow(ax, (x + 8.0, yc), (x + 11.0, yc))
    S.span(ax, x - 12.4, x + 8.0, yc + 9.4, "Classification head", color=S.BLUE_E,
           fs=6.9, drop=-2.4)
    x += 11.0

    # four-class posterior, illustrative shape only
    classes = ["NoGas", "Perfume", "Smoke", "Mixture"]
    bars = [0.04, 0.07, 0.11, 0.78]
    bx0, bw = x + 4.6, 7.2
    for i, (c, v) in enumerate(zip(classes, bars)):
        yy = yc + 4.6 - i * 3.1
        ax.add_patch(Rectangle((bx0, yy - 1.05), bw, 2.1, fc="none",
                               ec="#cfcfcf", lw=0.5, zorder=4))
        ax.add_patch(Rectangle((bx0, yy - 1.05), bw * v, 2.1, fc="#7a4f9a",
                               alpha=0.75, ec="none", zorder=5))
        ax.text(bx0 - 0.5, yy, c, fontsize=6.0, color=S.INK2, ha="right",
                va="center", zorder=6)
    ax.text(bx0 + bw / 2, yc + 7.2, "class posterior", fontsize=6.6,
            color=S.INK2, ha="center", va="bottom", zorder=6)

    ax.text(50.0, 6.0, "the class posterior reaches the explanation path only; it contributes no term to "
                       r"$\varphi_t$ and no safety result depends on it",
            fontsize=6.7, color=S.MUTED, ha="center", va="center", zorder=9)
    return S.save(fig, os.path.join(FIG, "fig_arch_yolo.png"))


def main():
    for f in (fig_lstm, fig_dqn, fig_yolo):
        print("wrote", os.path.basename(f()))


if __name__ == "__main__":
    main()
