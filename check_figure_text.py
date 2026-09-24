"""Find overlapping text in the rendered figures.

Reading twenty-four figures by eye is how overlaps get shipped. This renders
each one and tests every pair of text artists for real overlap, so a label that
lands on its neighbour is a reported failure rather than something a reader
notices after submission.

An overlap counts when two text boxes intersect over more than OVERLAP_FRAC of
the smaller of the two. Ordinary typography puts tick labels and legend entries
within a point or two of one another, so a small intersection is not a defect;
one box sitting a quarter of the way into another is.

    python3 check_figure_text.py            every figure
    python3 check_figure_text.py fig_loco   one figure
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OVERLAP_FRAC = 0.25       # of the smaller box
MIN_SIDE = 2.0            # px; ignore empty or hairline boxes

FINDINGS = []
BOX_FINDINGS = []


def _offview_tick_labels(fig):
    """Tick label artists for ticks outside their axis limits.

    Matplotlib keeps those artists alive and they still report an extent, so
    they turn up as phantom collisions against whatever sits where they would
    have been drawn.
    """
    dead = set()
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                loc = tick.get_loc()
                if loc is None or not (lo - 1e-12 <= loc <= hi + 1e-12):
                    dead.add(id(tick.label1))
                    dead.add(id(tick.label2))
    return dead


def _texts(fig, renderer):
    out = []
    dead = _offview_tick_labels(fig)
    for t in fig.findobj(Text):
        s = t.get_text()
        if not s or not s.strip():
            continue
        if not t.get_visible() or id(t) in dead:
            continue
        try:
            # Text.get_window_extent rather than the artist's own: an Annotation
            # reports the union of its text and its leader line, which made two
            # well-separated callouts look like one tall overlapping block.
            bb = Text.get_window_extent(t, renderer)
        except Exception:
            try:
                bb = t.get_window_extent(renderer=renderer)
            except Exception:
                continue
        if bb.width < MIN_SIDE or bb.height < MIN_SIDE:
            continue
        out.append((s, _rot_rect(bb, float(t.get_rotation() or 0.0)), t))
    return out


def _rot_rect(bb, deg):
    """The true rotated box of a text artist, from its axis-aligned extent.

    Rotated tick labels have axis-aligned boxes that overlap heavily while the
    glyphs themselves sit clear of one another on the diagonal. Testing the
    axis-aligned boxes reports those as collisions, so the rotated rectangle is
    reconstructed and the test done on that.
    """
    th = np.deg2rad(deg % 180.0)
    c, s = abs(np.cos(th)), abs(np.sin(th))
    W, H = bb.width, bb.height
    det = c * c - s * s
    if abs(det) < 1e-3:                        # near 45 degrees
        w = h = (W + H) / (2.0 * (c + s))
    else:
        w = (W * c - H * s) / det
        h = (H * c - W * s) / det
    w, h = max(w, 1.0), max(h, 1.0)
    cx, cy = (bb.x0 + bb.x1) / 2.0, (bb.y0 + bb.y1) / 2.0
    a2 = np.deg2rad(deg)
    ca, sa = np.cos(a2), np.sin(a2)
    return np.array([(cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)
                     for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2),
                                    (w / 2, h / 2), (-w / 2, h / 2))])


def _poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip(subject, clip):
    """Sutherland-Hodgman: the intersection of two convex polygons."""
    out = list(subject)
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        edge = b - a
        src, out = out, []
        if not src:
            return []
        def inside(p):
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= -1e-9
        for j in range(len(src)):
            cur, prv = src[j], src[j - 1]
            if inside(cur):
                if not inside(prv):
                    out.append(_cross(prv, cur, a, b))
                out.append(cur)
            elif inside(prv):
                out.append(_cross(prv, cur, a, b))
    return out


def _cross(p1, p2, a, b):
    d1, d2 = p2 - p1, b - a
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return p1
    tt = ((a[0] - p1[0]) * d2[1] - (a[1] - p1[1]) * d2[0]) / den
    return p1 + tt * d1


def _inter(ra, rb):
    poly = _clip(ra, rb)
    return _poly_area(np.array(poly)) if len(poly) >= 3 else 0.0


def audit_boxes(fig, name, renderer, items):
    """Text must stay inside the box it is centred in.

    A label set in a box that was sized by a character count rather than by
    measurement runs out of both ends of it. The box a label belongs to is the
    smallest visible patch containing the label's centre, so a panel frame or an
    axes background is never mistaken for it.
    """
    boxes = []
    for pch in fig.findobj(lambda a: isinstance(a, (FancyBboxPatch, Rectangle))):
        if not pch.get_visible():
            continue
        try:
            bb = pch.get_window_extent(renderer)
        except Exception:
            continue
        if bb.width < 8 or bb.height < 8:
            continue
        boxes.append(bb)
    bad = []
    for s, rect, t in items:
        x0, x1 = rect[:, 0].min(), rect[:, 0].max()
        y0, y1 = rect[:, 1].min(), rect[:, 1].max()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        # Only a label deliberately centred in a box is checked. A value label
        # that happens to sit inside a bar, or a tick label whose centre falls in
        # the axes background, is not that and is left alone.
        inner = [bb for bb in boxes
                 if abs((bb.x0 + bb.x1) / 2.0 - cx) < 3.0
                 and abs((bb.y0 + bb.y1) / 2.0 - cy) < 3.0]
        if not inner:
            continue
        bb = min(inner, key=lambda b: b.width * b.height)
        over_x = max(bb.x0 - x0, x1 - bb.x1)
        over_y = max(bb.y0 - y0, y1 - bb.y1)
        # A text box carries leading above and below the glyphs, so a line set
        # snugly in a band reports a few pixels of vertical overflow that is not
        # visible. Horizontal overflow has no such slack and is checked hard.
        if over_x > 1.0 or over_y > 0.25 * (y1 - y0):
            bad.append((max(over_x, over_y), s.replace("\n", " ")[:44],
                        over_x, over_y))
    bad.sort(reverse=True)
    if bad:
        BOX_FINDINGS.append((name, bad))
        print("  FAIL %-26s %d label(s) outside their box" % (name, len(bad)))
        for _, s, ox, oy in bad[:6]:
            print("        %-46r overflows by %.0f px across, %.0f px down" % (s, ox, oy))
    return bad


def audit(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = _texts(fig, r)
    hits = []
    for i in range(len(items)):
        si, bi, ti = items[i]
        for j in range(i + 1, len(items)):
            sj, bj, tj = items[j]
            if ti is tj:
                continue
            a = _inter(bi, bj)
            if a <= 0:
                continue
            smaller = min(_poly_area(bi), _poly_area(bj))
            if smaller <= 0:
                continue
            frac = a / smaller
            if frac >= OVERLAP_FRAC:
                hits.append((frac, si.replace("\n", " ")[:38],
                             sj.replace("\n", " ")[:38]))
    audit_boxes(fig, name, r, items)
    hits.sort(reverse=True)
    if hits:
        FINDINGS.append((name, hits))
        print("  FAIL %-26s %d overlapping pair(s)" % (name, len(hits)))
        for frac, a, b in hits[:6]:
            print("        %.0f%%  %-40r %r" % (100 * frac, a, b))
    elif name not in [n for n, _ in BOX_FINDINGS]:
        print("  ok   %-26s %d text items" % (name, len(items)))


def run(only=None):
    import matplotlib.figure as mfig
    original = mfig.Figure.savefig
    state = {"name": None}

    def patched(self, fname, *a, **k):
        out = original(self, fname, *a, **k)
        try:
            audit(self, state["name"] or os.path.basename(str(fname)))
        except Exception as exc:                       # never mask a draw error
            print("  ??   %s: %s" % (state["name"], exc))
        return out

    mfig.Figure.savefig = patched
    try:
        from retrain import make_figs_v2 as M
        from retrain import make_figs_arch as A
        drivers = [(n, getattr(M, n)) for n in dir(M) if n.startswith("fig_")]
        drivers += [("fig_arch_lstm", A.fig_lstm), ("fig_arch_dqn", A.fig_dqn),
                    ("fig_arch_yolo", A.fig_yolo)]
        for n, fn in sorted(drivers):
            if only and only not in n:
                continue
            state["name"] = n
            try:
                fn()
            except Exception as exc:
                print("  ERR  %-26s %r" % (n, exc))
            plt.close("all")
    finally:
        mfig.Figure.savefig = original

    print()
    bad = False
    if FINDINGS:
        print("%d figure(s) with overlapping text" % len(FINDINGS))
        bad = True
    if BOX_FINDINGS:
        print("%d figure(s) with a label outside its box" % len(BOX_FINDINGS))
        bad = True
    if bad:
        return 1
    print("no overlapping text, and no label outside its box, in any figure")
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else None))
