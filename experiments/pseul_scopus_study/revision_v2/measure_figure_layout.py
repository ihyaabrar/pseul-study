# -*- coding: utf-8 -*-
"""Measure the two flowcharts against published graph-drawing aesthetics.

lint_figures.py catches defects that are wrong at any size: text outside its
card, text on text, an arrowhead inside a box. This measures the things that
make a correct diagram hard to read, using the aesthetics that were actually
tested on human readers rather than the ones that feel right.

  * edge crossings -- Purchase (Graph Drawing 1997, LNCS 1353) found this the
    single aesthetic with a measurable effect on comprehension time and error
    rate, ahead of bends and symmetry
  * bends per edge and total edge length -- second-order in the same study, and
    the quantities the Sugiyama, Tagawa and Toda framework (IEEE T-SMC 11(2),
    1981) optimises after crossings
  * alignment guides -- Wong, "Design of data figures" (Nat Methods 7:665, 2010):
    shared coordinates produce invisible lines that make a composition read as
    ordered. Counted as the number of distinct box centres per axis.
  * canvas balance -- the emptiest quadrant against the fullest, which is what
    reads as "unfinished" even when nothing is wrong.

Run alongside lint_figures.py after any change to generate_figures.py.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.pseul_scopus_study.revision_v2 import generate_figures as gf  # noqa: E402

EPS = 1e-9


def _seg_cross(p, q, r, s):
    """True where segments pq and rs cross at an interior point of both."""
    def d(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = d(r, s, p), d(r, s, q)
    d3, d4 = d(p, q, r), d(p, q, s)
    if abs(d1) < EPS or abs(d2) < EPS or abs(d3) < EPS or abs(d4) < EPS:
        return False  # touching or collinear is a junction, not a crossing
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


def capture(fn):
    plt.close("all")
    captured = {}
    original = gf._save
    gf._save = lambda fig, stem: captured.__setitem__("fig", fig)
    try:
        fn()
    finally:
        gf._save = original
    return captured["fig"]


def measure(fn, name):
    fig = capture(fn)
    ax = fig.axes[0]
    fig.canvas.draw()

    segments = []
    for line in ax.lines:
        xs, ys = line.get_xdata(), line.get_ydata()
        for i in range(len(xs) - 1):
            segments.append(((xs[i], ys[i]), (xs[i + 1], ys[i + 1])))
    for p in ax.patches:
        if isinstance(p, FancyArrowPatch) and hasattr(p, "_posA_posB"):
            a, b = p._posA_posB
            segments.append((tuple(a), tuple(b)))

    crossings = sum(
        1
        for i, (p, q) in enumerate(segments)
        for r, s in segments[i + 1:]
        if _seg_cross(p, q, r, s)
    )

    length = sum(((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2) ** 0.5 for p, q in segments)

    # a bend is a shared endpoint where direction changes
    ends = Counter()
    for p, q in segments:
        ends[(round(p[0], 4), round(p[1], 4))] += 1
        ends[(round(q[0], 4), round(q[1], 4))] += 1
    bends = sum(1 for v in ends.values() if v == 2)

    centres_x, centres_y, occupied = [], [], []
    for p in ax.patches:
        if isinstance(p, FancyBboxPatch):
            x, y = p.get_x(), p.get_y()
            w, h = p.get_width(), p.get_height()
            centres_x.append(round(x + w / 2, 3))
            centres_y.append(round(y + h / 2, 3))
            occupied.append((x, y, x + w, y + h))
        elif isinstance(p, Polygon):
            xy = p.get_xy()
            xs, ys = xy[:, 0], xy[:, 1]
            centres_x.append(round((xs.min() + xs.max()) / 2, 3))
            centres_y.append(round((ys.min() + ys.max()) / 2, 3))
            occupied.append((xs.min(), ys.min(), xs.max(), ys.max()))

    # coverage of a 4x4 grid of the canvas, to find the dead regions
    cells = []
    for row in range(4):
        for col in range(4):
            cx0, cx1 = col / 4, (col + 1) / 4
            cy0, cy1 = row / 4, (row + 1) / 4
            area = sum(
                max(0.0, min(x1, cx1) - max(x0, cx0)) * max(0.0, min(y1, cy1) - max(y0, cy0))
                for x0, y0, x1, y1 in occupied
            )
            cells.append((area / 0.0625, col, row))
    cells.sort()

    print(f"=== {name} ===")
    print(f"  {len(occupied)} nodes, {len(segments)} segments")
    print(f"  edge crossings ....... {crossings}")
    print(f"  bends ................ {bends}")
    print(f"  total edge length .... {length:.2f} (canvas widths)")
    print(f"  distinct box centres . {len(set(centres_x))} x, {len(set(centres_y))} y")
    print(f"  emptiest cell ........ col {cells[0][1]} row {cells[0][2]} at {cells[0][0]:.0%} full")
    print(f"  fullest cell ......... col {cells[-1][1]} row {cells[-1][2]} at {cells[-1][0]:.0%} full")
    print()
    return crossings, bends, length


def main():
    gf._style()
    measure(gf.figure_study_workflow, "Fig. 1  study workflow")
    measure(gf.figure_pseul_framework, "Fig. 2  PSEUL framework")


if __name__ == "__main__":
    main()
