"""Drawing primitives in the idiom of the SPE Journal model papers.

Four recent SPE Journal papers were measured to fix these conventions rather
than guess at them (spe-205365, spe-230320, spe-231403, spe-234709):

  page geometry   single column 3.35 in, one-and-a-half 4.77 in, full 6.81 in
  raster density  published figures sit at 300 to 430 dpi after the publisher
                  downsamples; source art is delivered at 600
  grouping        a dashed rounded container per functional block, with a bold
                  label set inside its top-left corner
  boxes           solid rounded rectangles, one light fill per family
  operators       small circled glyphs on the wire for add, subtract, multiply
  tensors         feature maps as stacked slabs drawn in weak perspective
  spans           a double-headed arrow under a region, labelled beneath

Everything here is geometry and drawing only. No figure content lives in this
module.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, Rectangle, Circle, FancyArrowPatch,
                                Polygon, Ellipse, Arc)

# --- page geometry, from the model papers -------------------------------------
COL1, COL15, FULL = 3.35, 4.77, 6.81
DPI = 600

# --- palette ------------------------------------------------------------------
# The model papers use one pale fill per functional family with a saturated
# outline of the same hue. Kept deliberately few so a reader can tell families
# apart at print size.
BLUE_F, BLUE_E = "#cfe4f2", "#2f6f9f"      # data and tensors
MINT_F, MINT_E = "#cfe2da", "#2f7f63"      # learned blocks
SAND_F, SAND_E = "#f2e6cc", "#9a7526"      # outputs and metrics
ROSE_F, ROSE_E = "#f2d9d9", "#a34b4b"      # hazard and failure
GREY_F, GREY_E = "#e8eaec", "#8a9096"      # inert and context
SURFACE = "#ffffff"
INK, INK2, MUTED = "#111111", "#3a3a3a", "#7a7a7a"


def use_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 7.6, "axes.labelsize": 7.6,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "axes.edgecolor": "#bbbbbb", "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "grid.color": "#e4e4e4", "grid.linewidth": 0.5,
        "legend.frameon": False,
        "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def canvas(width_in, height_in):
    """A blank drawing axes in inch-like units: 100 x-units to the full width."""
    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * height_in / width_in)
    ax.axis("off")
    return fig, ax


def group(ax, x, y, w, h, label=None, ec=BLUE_E, fc="none", z=1, label_dx=1.4,
          label_indent=0.0, label_fs=8.2, label_ls=1.12):
    """Dashed rounded container with a label inside its top-left corner.

    A label carrying a newline is drawn one line per text artist rather than as
    one multi-line artist, so `label_indent` can step the continuation lines in
    under the first. Matplotlib has no hanging-indent control on a multi-line
    string, and leading spaces are not a reliable substitute at this size.
    """
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.0,rounding_size=1.2",
                                fc=fc, ec=ec, lw=0.9, ls=(0, (4, 2.4)), zorder=z))
    if label:
        fig = ax.get_figure()
        # 100 x-units span the full figure width, so a point is this many units.
        unit = 100.0 / (fig.get_size_inches()[0] * 72.0)
        step = label_fs * label_ls * unit
        for k, line in enumerate(label.split("\n")):
            ax.text(x + label_dx + (label_indent if k else 0.0),
                    y + h - 1.5 - k * step, line, fontsize=label_fs, color=ec,
                    fontweight="bold", style="italic", va="top", ha="left",
                    zorder=z + 4)
    return x, y, w, h


def box(ax, x, y, w, h, text, fc=BLUE_F, ec=BLUE_E, fs=7.2, z=3, bold=False,
        lw=0.9, round_size=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.0,rounding_size={round_size}",
                                fc=fc, ec=ec, lw=lw, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, fontsize=fs, color=INK, ha="center",
            va="center", zorder=z + 1,
            fontweight="bold" if bold else "normal", linespacing=1.28)
    return (x, y, w, h)


def diamond(ax, cx, cy, w, h, text, fc=SAND_F, ec=SAND_E, fs=6.8, z=3):
    ax.add_patch(Polygon([[cx, cy + h / 2], [cx + w / 2, cy],
                          [cx, cy - h / 2], [cx - w / 2, cy]],
                         closed=True, fc=fc, ec=ec, lw=0.9, zorder=z))
    ax.text(cx, cy, text, fontsize=fs, color=INK, ha="center", va="center",
            zorder=z + 1, linespacing=1.22)


def arrow(ax, p0, p1, color=INK2, lw=0.9, z=5, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=7.5,
                                 color=color, lw=lw, zorder=z, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def elbow(ax, p0, p1, via_x=None, via_y=None, color=INK2, lw=0.9, z=5):
    """Orthogonal two-segment route, the routing the model papers use."""
    (x0, y0), (x1, y1) = p0, p1
    if via_x is not None:
        ax.plot([x0, via_x], [y0, y0], color=color, lw=lw, zorder=z,
                solid_capstyle="round")
        ax.plot([via_x, via_x], [y0, y1], color=color, lw=lw, zorder=z,
                solid_capstyle="round")
        arrow(ax, (via_x, y1), (x1, y1), color=color, lw=lw, z=z)
    else:
        vy = via_y if via_y is not None else (y0 + y1) / 2
        ax.plot([x0, x0], [y0, vy], color=color, lw=lw, zorder=z,
                solid_capstyle="round")
        ax.plot([x0, x1], [vy, vy], color=color, lw=lw, zorder=z,
                solid_capstyle="round")
        arrow(ax, (x1, vy), (x1, y1), color=color, lw=lw, z=z)


def op(ax, cx, cy, glyph="+", r=0.95, z=6, ec=INK2):
    """Circled operator on a wire."""
    ax.add_patch(Circle((cx, cy), r, fc="#ffffff", ec=ec, lw=0.9, zorder=z))
    ax.text(cx, cy, glyph, fontsize=7.4, ha="center", va="center",
            color=ec, zorder=z + 1)


def slab(ax, x, y, w, h, n=3, fc=BLUE_F, ec=BLUE_E, dx=0.55, dy=0.45, z=4,
         label=None, fs=6.4):
    """A feature tensor as stacked slabs in weak perspective."""
    for i in range(n - 1, -1, -1):
        ax.add_patch(Rectangle((x + i * dx, y + i * dy), w, h, fc=fc, ec=ec,
                               lw=0.7, zorder=z + (n - i)))
    if label:
        ax.text(x + w / 2 + (n - 1) * dx / 2, y - 1.15, label, fontsize=fs,
                color=INK2, ha="center", va="top", zorder=z + n + 1)


def span(ax, x0, x1, y, label, color=INK2, fs=7.0, z=6, drop=1.35):
    """Double-headed span arrow with a label beneath, as under Backbone/Neck/Head."""
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="<|-|>",
                                 mutation_scale=6.5, color=color, lw=0.8, zorder=z,
                                 shrinkA=0, shrinkB=0))
    ax.text((x0 + x1) / 2, y - drop, label, fontsize=fs, color=color,
            ha="center", va="top", fontweight="bold", style="italic", zorder=z)


def band(ax, x, y, w, h, fc="#dbeaf5", z=0):
    """A pale horizontal band, the device Fig. 1 of the YOLOv13 paper uses."""
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="none", zorder=z))


def blockarrow(ax, x, y, w, h, color="#7fb3d5", z=2, down=True):
    """Fat tapered arrow between stages."""
    if down:
        pts = [[x + w * 0.30, y + h], [x + w * 0.70, y + h], [x + w * 0.70, y + h * 0.42],
               [x + w, y + h * 0.42], [x + w / 2, y], [x, y + h * 0.42],
               [x + w * 0.30, y + h * 0.42]]
    else:
        pts = [[x, y + h * 0.30], [x, y + h * 0.70], [x + w * 0.58, y + h * 0.70],
               [x + w * 0.58, y + h], [x + w, y + h / 2], [x + w * 0.58, y],
               [x + w * 0.58, y + h * 0.30]]
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec="none", zorder=z))



def panel(ax, x, y, w, h, title, ec="#1a1a1a", lw=1.1, fs=10.0, pad=1.5, z=1):
    """A module frame: thin solid rule, white fill, bold title inside the top.

    This is the container the SPE model papers use for a functional stage, in
    place of the dashed grouping used for a sub-block of a network diagram.
    """
    ax.add_patch(Rectangle((x, y), w, h, fc=SURFACE, ec=ec, lw=lw, zorder=z))
    ax.text(x + w / 2, y + h - pad, title, fontsize=fs, fontweight="bold",
            ha="center", va="top", color=INK, zorder=z + 2)
    return x, y, w, h


def cylinder(ax, x, y, w, h, text, fc="#fbf0cd", ec="#9a7526", fs=7.4, z=4,
             bold=False):
    """A data store, drawn as a horizontal drum with the cap on the right."""
    ew = min(h * 0.42, w * 0.16)
    bx0, bx1 = x + ew / 2, x + w - ew / 2
    ax.add_patch(Rectangle((bx0, y), bx1 - bx0, h, fc=fc, ec="none", zorder=z))
    ax.plot([bx0, bx1], [y, y], color=ec, lw=0.9, zorder=z + 1)
    ax.plot([bx0, bx1], [y + h, y + h], color=ec, lw=0.9, zorder=z + 1)
    ax.add_patch(Ellipse((bx1, y + h / 2), ew, h, fc=fc, ec=ec, lw=0.9, zorder=z + 2))
    ax.add_patch(Arc((bx0, y + h / 2), ew, h, theta1=90, theta2=270, color=ec,
                     lw=0.9, zorder=z + 2))
    ax.text((x + bx1) / 2, y + h / 2, text, fontsize=fs, ha="center", va="center",
            color=INK, zorder=z + 3, linespacing=1.3,
            fontweight="bold" if bold else "normal")


def stagearrow(ax, cx, y0, y1, w=5.0, fc="#cfe4f2", ec="#5d97bf", z=3):
    """The fat arrow that carries one stage into the next."""
    sh = min(2.6, abs(y1 - y0) * 0.45)          # head depth
    pts = [[cx - w * 0.28, y0], [cx + w * 0.28, y0], [cx + w * 0.28, y1 + sh],
           [cx + w / 2, y1 + sh], [cx, y1], [cx - w / 2, y1 + sh],
           [cx - w * 0.28, y1 + sh]]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=0.8, zorder=z))


def rightarrow(ax, x0, x1, cy, h=3.4, fc="#cfe4f2", ec="#5d97bf", z=3):
    """The same arrow, pointing right, for a chain inside one stage."""
    sw = min(2.4, abs(x1 - x0) * 0.45)
    pts = [[x0, cy - h * 0.28], [x1 - sw, cy - h * 0.28], [x1 - sw, cy - h / 2],
           [x1, cy], [x1 - sw, cy + h / 2], [x1 - sw, cy + h * 0.28],
           [x0, cy + h * 0.28]]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=0.8, zorder=z))


def text_width(ax, s, fs, **kw):
    """Width of a string in the axes' data units, measured rather than guessed.

    Box widths in these diagrams used to be chosen by eye against a character
    count, which is how "4    Recommend ESD assessment" came to be wider than the
    rung it sat in. Measuring the laid-out string removes the guess.
    """
    fig = ax.figure
    try:
        r = fig.canvas.get_renderer()
    except AttributeError:                       # backends without one to hand
        from matplotlib.backend_bases import _get_renderer
        r = _get_renderer(fig)
    t = ax.text(0, 0, s, fontsize=fs, alpha=0.0, **kw)
    bb = t.get_window_extent(renderer=r)
    t.remove()
    p = ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    return abs(p[1][0] - p[0][0])


def fit_fontsize(ax, labels, width, start=8.0, floor=4.8, margin=0.9, step=0.2,
                 **kw):
    """The largest size at which every label fits the given box width.

    Multi-line labels are measured line by line, since it is the widest line that
    decides whether the text stays inside the box.
    """
    lines = [ln for lab in labels for ln in str(lab).split("\n") if ln.strip()]
    fs = start
    while fs > floor:
        if max(text_width(ax, ln, fs, **kw) for ln in lines) <= width - 2 * margin:
            return fs
        fs -= step
    return floor


def save(fig, path):
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
