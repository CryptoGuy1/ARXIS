"""Flowchart primitives in the idiom of the model paper's system figure.

A Times-metric serif throughout, thin black module frames with a bold title set
inside the top, square-cornered boxes with a pale fill and a saturated outline of
the same hue, data stores drawn as horizontal drums, and chunky outlined block
arrows carrying one stage into the next.

Box widths and type sizes are measured against the strings they have to hold
rather than chosen by eye, which is what keeps a label inside its box when the
wording changes.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from matplotlib.patches import Rectangle, Ellipse, Arc, Polygon, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures_v2")
FULL, DPI = 6.90, 600

# --- palette, sampled from the reference -------------------------------------
INK = "#000000"
YEL_F, YEL_E = "#fbf0ce", "#b8912f"      # data stores
BLU_F, BLU_E = "#d4e8f4", "#5f9ec4"      # stage one and the protocol
PCH_F, PCH_E = "#fbe3ce", "#d99a62"      # the comparators
GRN_F, GRN_E = "#e4eed8", "#8faf6c"      # the evaluation regimes
GRY_F, GRY_E = "#ebedee", "#8a9096"
ROSE_E = "#a34b4b"
ARR_F, ARR_E = "#cfe4f2", "#7fb3d5"      # block arrows
MUTED = "#6a6a6a"

FRAME_AR = 640.0 / 480.0                 # the released frames are 480 by 640
PT = 100.0 / (FULL * 72.0)                # canvas units per typographic point


def lh(fs):
    """One line of type, in canvas units."""
    return fs * 1.30 * PT


def caption_h(fs_name, fs_sub, nsub=3, lead=0.9):
    return lead + lh(fs_name) + nsub * lh(fs_sub)

CLASSES = [
    ("NoGas", "clean air", "a = 0", "Monitor", GRY_E, False),
    ("Perfume", "deodorant vapor", "a = 1", "Increase sampling", YEL_E, False),
    ("Smoke", "incense smoke", "a = 3", "Raise alarm", ROSE_E, True),
    ("Mixture", "smoke and vapor", "a = 4", "Recommend ESD", ROSE_E, True),
]
COMPARATORS = ["Cost-weighted\npolicy", "Unweighted\nnetwork", "Gradient\nboosting",
               "Random\nforest", "Recurrent\n(LSTM)", "Conservative\nQ-learning",
               "Perceptron", "k-nearest\nneighbors", "Support-vector\nclassifier",
               "Shallow tree\ndepth 3", "CUSUM"]
REGIMES = [("In distribution", "30 randomized runs"),
           ("Class exclusion", "one hazardous class withheld"),
           ("Graded degradation", "noise, drift, channel loss"),
           ("Raw-space faults", "stuck, zero, clamp, noise"),
           ("Reject option", "confidence threshold")]
METRICS = ("Missed hazard     Escalation adequacy     Under-escalation     "
           "High-severity false alarm     Alarm burden")

RC = {
    "font.family": "Liberation Serif",
    "mathtext.fontset": "stix",
    "figure.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
    "figure.dpi": DPI, "savefig.dpi": DPI,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
}


# ----------------------------------------------------------------- primitives
def canvas(h_units):
    fig = plt.figure(figsize=(FULL, FULL * h_units / 100.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, h_units)
    ax.axis("off")
    return fig, ax


def measure(ax, s, fs, **kw):
    """Width of a laid-out string in canvas units."""
    fig = ax.figure
    try:
        r = fig.canvas.get_renderer()
    except AttributeError:
        from matplotlib.backend_bases import _get_renderer
        r = _get_renderer(fig)
    t = ax.text(0, 0, s, fontsize=fs, alpha=0.0, **kw)
    bb = t.get_window_extent(renderer=r)
    t.remove()
    p = ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    return abs(p[1][0] - p[0][0])


def fit_fs(ax, labels, width, start, floor=4.4, margin=0.8, **kw):
    lines = [ln for lab in labels for ln in str(lab).split("\n") if ln.strip()]
    fs = start
    while fs > floor:
        if max(measure(ax, ln, fs, **kw) for ln in lines) <= width - 2 * margin:
            return round(fs, 2)
        fs -= 0.15
    return floor


def panel(ax, x, y, w, h, title, fs=10.4, pad=1.4):
    ax.add_patch(Rectangle((x, y), w, h, fc="#ffffff", ec=INK, lw=1.15, zorder=1))
    ax.text(x + w / 2, y + h - pad, title, fontsize=fs, fontweight="bold",
            ha="center", va="top", color=INK, zorder=3)


def box(ax, x, y, w, h, text, fc, ec, fs, bold=False, z=3, lw=0.9):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, fontsize=fs, ha="center", va="center",
            color=INK, zorder=z + 1, linespacing=1.3,
            fontweight="bold" if bold else "normal")


def cyl(ax, x, y, w, h, text, fs, fc=YEL_F, ec=YEL_E, z=4):
    ew = min(h * 0.44, w * 0.15)
    bx0, bx1 = x + ew / 2, x + w - ew / 2
    ax.add_patch(Rectangle((bx0, y), bx1 - bx0, h, fc=fc, ec="none", zorder=z))
    ax.plot([bx0, bx1], [y, y], color=ec, lw=0.9, zorder=z + 1)
    ax.plot([bx0, bx1], [y + h, y + h], color=ec, lw=0.9, zorder=z + 1)
    ax.add_patch(Ellipse((bx1, y + h / 2), ew, h, fc=fc, ec=ec, lw=0.9, zorder=z + 2))
    ax.add_patch(Arc((bx0, y + h / 2), ew, h, theta1=90, theta2=270, color=ec,
                     lw=0.9, zorder=z + 2))
    ax.text((x + bx1) / 2, y + h / 2, text, fontsize=fs, ha="center", va="center",
            color=INK, zorder=z + 3, linespacing=1.3)


def barrow(ax, cx, cy, length, thick, direction="down", fc=ARR_F, ec=ARR_E, z=3):
    """The outlined block arrow the reference uses between stages."""
    L, T, hd = length, thick, min(thick * 0.9, length * 0.55)
    if direction in ("down", "up"):
        s = -1 if direction == "down" else 1
        y0, y1 = cy - s * L / 2, cy + s * L / 2      # tail, tip
        # The head base sits back from the tip, towards the tail. Written with
        # the sign the other way the head was drawn beyond the tip, so every
        # vertical block arrow overran the box it pointed at.
        hb = y1 - s * hd
        pts = [(cx - T * 0.30, y0), (cx + T * 0.30, y0),
               (cx + T * 0.30, hb), (cx + T / 2, hb),
               (cx, y1), (cx - T / 2, hb), (cx - T * 0.30, hb)]
    else:
        s = 1 if direction == "right" else -1
        x0, x1 = cx - s * L / 2, cx + s * L / 2
        pts = [(x0, cy - T * 0.30), (x0, cy + T * 0.30),
               (x1 - s * hd, cy + T * 0.30), (x1 - s * hd, cy + T / 2),
               (x1, cy), (x1 - s * hd, cy - T / 2), (x1 - s * hd, cy - T * 0.30)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=0.8, zorder=z))


def tarrow(ax, p0, p1, label=None, fs=8.0, dx=0.9, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8.5,
                                 color=INK, lw=0.9, zorder=z, shrinkA=0, shrinkB=0))
    if label:
        ax.text((p0[0] + p1[0]) / 2 + dx, (p0[1] + p1[1]) / 2, label, fontsize=fs,
                color=INK, ha="left", va="center", zorder=z)


def span(ax, x0, x1, y, label, color, fs):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="<|-|>",
                                 mutation_scale=7.0, color=color, lw=0.85,
                                 zorder=6, shrinkA=0, shrinkB=0))
    ax.text((x0 + x1) / 2, y - 0.4, label, fontsize=fs, color=color, ha="center",
            va="top", style="italic", fontweight="bold", zorder=6)


# ------------------------------------------------------------- content blocks
def frames(ax, x, y, w, fh, fs_name=8.6, fs_sub=7.2, spans=True):
    """One released frame per class, on a column grid sized by the captions."""
    colw = w / 4.0
    fw = min(colw - 1.6, fh / FRAME_AR)
    for i, (cls, what, a_id, a_lab, ec, hazard) in enumerate(CLASSES):
        cx = x + i * colw + colw / 2
        fx = cx - fw / 2
        src = os.path.join(ROOT, "data", cls, f"frame_{cls}.png")
        if os.path.exists(src):
            ax.imshow(mpimg.imread(src), extent=[fx, fx + fw, y, y + fh],
                      aspect="auto", zorder=4, interpolation="lanczos")
        else:
            ax.add_patch(Rectangle((fx, y), fw, fh, fc=GRY_F, ec=GRY_E, lw=0.8, zorder=4))
        ax.add_patch(Rectangle((fx, y), fw, fh, fc="none", ec=ec,
                               lw=1.5 if hazard else 0.9, zorder=6))
        yy = y - 0.9
        for txt, fs, col, bold in ((cls, fs_name, INK, True),
                                   (what, fs_sub, "#333333", False),
                                   (a_id, fs_sub, ec, True),
                                   (a_lab, fs_sub, ec, True)):
            ax.text(cx, yy, txt, fontsize=fs, color=col, ha="center", va="top",
                    fontweight="bold" if bold else "normal", zorder=6)
            yy -= lh(fs)
    if spans:
        left = x + colw / 2 - fw / 2
        span(ax, left, left + colw + fw, y + fh + 1.9, "nonhazardous", "#333333", fs_sub + 0.4)
        span(ax, left + 2 * colw, left + 3 * colw + fw, y + fh + 1.9,
             "hazardous surrogate", ROSE_E, fs_sub + 0.4)
    return y - caption_h(fs_name, fs_sub)


def trace(ax, fig, x, y, w, h, H, fs=8.6, title=True, subs=3):
    """The seven-channel window the decision component actually receives."""
    axw = fig.add_axes([x / 100.0, y / H, w / 100.0, h / H])
    try:
        d = pd.read_csv(os.path.join(ROOT, "data", "Gas_Sensors_Measurements.csv"))
        seg = d[d["Gas"] == "Smoke"].iloc[120:140]
        for k, c in enumerate(["MQ2", "MQ3", "MQ5", "MQ6", "MQ7", "MQ8", "MQ135"]):
            axw.plot(range(20), seg[c].to_numpy(float), lw=1.0,
                     color=plt.cm.viridis(k / 6.0 * 0.82 + 0.05), label=c)
        lo, hi = axw.get_ylim()
        axw.set_ylim(lo, hi + (hi - lo) * 0.44)
        axw.legend(fontsize=fs * 0.66, ncol=4, loc="upper center",
                   bbox_to_anchor=(0.5, 1.03), columnspacing=0.8,
                   handlelength=1.0, handletextpad=0.4, borderpad=0.2,
                   frameon=False)
    except Exception:
        pass
    axw.set_xticks([]); axw.set_yticks([])
    for s in axw.spines.values():
        s.set_visible(True); s.set_color(GRY_E); s.set_linewidth(0.8)
    if not title:
        return y
    SUB = ["the only evidence the decision component receives",
           "the thermal path adds no term to the decision state",
           "and is scored separately from it"][:subs]
    yy = y - 0.9
    ax.text(x + w / 2, yy, "Seven-channel MOX window, 20 × 7", fontsize=fs,
            fontweight="bold", ha="center", va="top", color=INK, zorder=6)
    yy -= lh(fs)
    for s in SUB:
        ax.text(x + w / 2, yy, s, fontsize=fs * 0.84, color="#333333",
                ha="center", va="top", zorder=6)
        yy -= lh(fs * 0.84)
    return yy


PROTO = [("Windowing", "length 20, boundary\nwindows dropped"),
         ("Block-wise holdout", "contiguous 20% per class\n20-window embargo\nseed-varying position")]


def protocol_row(ax, x, y, w, h, fs=7.6):
    """Windowing, holdout, counts, laid out left to right."""
    aw = w * 0.045
    bw = (w - 2 * aw) / 3.0
    xs = [x, x + bw + aw, x + 2 * (bw + aw)]
    box(ax, xs[0], y, bw, h, "Windowing\nlength 20\nboundary windows dropped",
        BLU_F, BLU_E, fs)
    barrow(ax, xs[0] + bw + aw / 2, y + h / 2, aw * 0.9, h * 0.44, "right")
    box(ax, xs[1], y, bw, h,
        "Block-wise holdout\ncontiguous 20% per class, 20-window\nembargo, seed-varying position",
        BLU_F, BLU_E, fs)
    barrow(ax, xs[1] + bw + aw / 2, y + h / 2, aw * 0.9, h * 0.44, "right")
    cyl(ax, xs[2], y, bw, h, "4,900 train / 1,264 test\n316 windows per class", fs)


def protocol_col(ax, x, y, w, h, fs=7.6):
    """The same three steps, stacked. Type measured against the column width."""
    labels = ["Windowing\nlength 20, boundary windows dropped",
              "Block-wise holdout\ncontiguous 20% per class, 20-window\nembargo, seed-varying position",
              "4,900 train / 1,264 test\n316 windows per class"]
    fs = min(fs, fit_fs(ax, labels, w, start=fs, floor=4.6, margin=0.55))
    ah = h * 0.10
    bh = (h - 2 * ah) / 3.0
    ys = [y + 2 * (bh + ah), y + bh + ah, y]
    box(ax, x, ys[0], w, bh, "Windowing\nlength 20, boundary windows dropped",
        BLU_F, BLU_E, fs)
    barrow(ax, x + w / 2, ys[0] - ah / 2, ah * 0.9, w * 0.10, "down")
    box(ax, x, ys[1], w, bh,
        "Block-wise holdout\ncontiguous 20% per class, 20-window\nembargo, seed-varying position",
        BLU_F, BLU_E, fs)
    barrow(ax, x + w / 2, ys[1] - ah / 2, ah * 0.9, w * 0.10, "down")
    cyl(ax, x, ys[2], w, bh, "4,900 train / 1,264 test\n316 windows per class", fs)


def comparators(ax, x, y, w, h, ncol, fs=None, gap=0.7):
    nrow = -(-len(COMPARATORS) // ncol)
    bw = (w - (ncol - 1) * gap) / ncol
    bh = (h - (nrow - 1) * gap) / nrow
    if fs is None:
        fs = fit_fs(ax, COMPARATORS, bw, start=7.4, floor=4.6, margin=0.5)
    for i, nm in enumerate(COMPARATORS):
        r, c = divmod(i, ncol)
        # the last row is centred when it is short
        in_row = min(ncol, len(COMPARATORS) - r * ncol)
        off = (w - (in_row * bw + (in_row - 1) * gap)) / 2 if in_row < ncol else 0.0
        box(ax, x + off + c * (bw + gap), y + h - (r + 1) * bh - r * gap, bw, bh,
            nm, PCH_F, PCH_E, fs)
    return fs


def regimes(ax, x, y, w, h, fs=None, gap=1.3, metric_h=3.1, rh_max=7.4):
    """The five regimes over the metric bar, as one block centred in the module.

    Left to fill whatever height the module has, the regime boxes grew to three
    times the height of their own text in the taller layouts.
    """
    rw = (w - 4 * gap) / 5.0
    if fs is None:
        fs = fit_fs(ax, [a + "\n" + b for a, b in REGIMES], rw, start=7.4,
                    floor=4.8, margin=0.5)
    rh = min(rh_max, max(4.6, h - metric_h - 3.4))
    block = rh + 2.4 + metric_h
    top = y + h - max(0.0, (h - block) / 2.0)
    for i, (head, sub) in enumerate(REGIMES):
        bx = x + i * (rw + gap)
        box(ax, bx, top - rh, rw, rh, head + "\n" + sub, GRN_F, GRN_E, fs)
        tarrow(ax, (bx + rw / 2, top - rh - 0.15), (bx + rw / 2, top - rh - 2.1))
    mfs = fit_fs(ax, [METRICS], w, start=8.2, floor=5.8, margin=1.0, fontweight="bold")
    box(ax, x, top - rh - 2.4 - metric_h, w, metric_h, METRICS, "#ffffff", ROSE_E,
        mfs, bold=True, lw=1.15)
    return fs


def gap_arrow(ax, cx, y_top, y_bot, thick=5.0, inset=0.35):
    """A block arrow drawn strictly inside the gap between two module frames."""
    y1, y0 = y_top - inset, y_bot + inset
    barrow(ax, cx, (y0 + y1) / 2.0, y1 - y0, thick, "down")


def gap_arrow_h(ax, cy, x_left, x_right, thick=4.4, inset=0.35):
    x0, x1 = x_left + inset, x_right - inset
    barrow(ax, (x0 + x1) / 2.0, cy, x1 - x0, thick, "right")


def source(ax, x, y, w, h, fs=8.4):
    cyl(ax, x, y, w, h, "MultimodalGasData corpus, 6,400 labeled samples", fs)


# The five rows of Table 5: decision accuracy beside four safety-relevant rates.
# Alarm burden is not a sixth quantity, it is the false-alarm rate in operational
# units by Eq. 9, so it is noted under the bar rather than given a cell.
METRIC_CELLS = ["Decision accuracy", "Missed hazard", "Escalation adequacy",
                "Under-escalation", "High-severity false alarm"]


def metric_bar(ax, x, y, w, h, fs=None):
    """The five quantities, one cell each, in a single ruled bar."""
    if fs is None:
        fs = fit_fs(ax, METRIC_CELLS, w / 5.0, start=8.4, floor=5.6, margin=0.7,
                    fontweight="bold")
    ax.add_patch(Rectangle((x, y), w, h, fc="#ffffff", ec=ROSE_E, lw=1.3, zorder=3))
    cw = w / 5.0
    for i, m in enumerate(METRIC_CELLS):
        if i:
            ax.plot([x + i * cw, x + i * cw], [y + 0.5, y + h - 0.5],
                    color=ROSE_E, lw=0.7, zorder=4)
        ax.text(x + (i + 0.5) * cw, y + h / 2, m, fontsize=fs, fontweight="bold",
                ha="center", va="center", color=INK, zorder=5)
    ax.text(x + w / 2, y - 0.8,
            "alarm burden (Eq. 9) is the high-severity false-alarm rate in "
            "operational units, not a sixth quantity",
            fontsize=7.0, style="italic", ha="center", va="top", color="#333333",
            zorder=5)


def chips(ax, x, y, w, h, labels, ncol, fc, ec, gap=0.7):
    """A compact grid of interchangeable items, last row centred."""
    nrow = -(-len(labels) // ncol)
    bw = (w - (ncol - 1) * gap) / ncol
    bh = (h - (nrow - 1) * gap) / nrow
    fs = fit_fs(ax, labels, bw, start=7.0, floor=4.6, margin=0.5)
    for i, nm in enumerate(labels):
        r, c = divmod(i, ncol)
        in_row = min(ncol, len(labels) - r * ncol)
        off = (w - (in_row * bw + (in_row - 1) * gap)) / 2 if in_row < ncol else 0.0
        box(ax, x + off + c * (bw + gap), y + h - (r + 1) * bh - r * gap,
            bw, bh, nm, fc, ec, fs)
    return fs


