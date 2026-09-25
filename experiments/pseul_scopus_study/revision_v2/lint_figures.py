# -*- coding: utf-8 -*-
"""Measure the two flowcharts instead of eyeballing them.

Every defect found in these diagrams so far was geometric: text wider than the
card holding it, an arrowhead landing inside a box because the card had been
widened without moving the route, a label overlapping the box beside it. Those
are all measurable, so this measures them.

It renders each figure, walks the drawn artists, and reports:

  * any text whose rendered box crosses the card it belongs to
  * any text that overlaps another text
  * arrow tips that fall inside a card rather than on its edge

Run it after any change to generate_figures.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.pseul_scopus_study.revision_v2 import generate_figures as gf  # noqa: E402

PAD = 0.002  # tolerance in axes fractions


def bounds(artist, ax, renderer):
    """Artist extent in axes-fraction coordinates."""
    bb = artist.get_window_extent(renderer=renderer)
    inv = ax.transAxes.inverted()
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return x0, y0, x1, y1


def overlaps(a, b):
    return not (a[2] <= b[0] + PAD or b[2] <= a[0] + PAD
                or a[3] <= b[1] + PAD or b[3] <= a[1] + PAD)


def audit(fn, name):
    plt.close("all")
    # the figure functions save and then close, so the figure has to be caught
    # on its way past _save rather than fetched afterwards
    captured = {}
    original = gf._save

    def spy(fig, stem):
        captured["fig"] = fig

    gf._save = spy
    try:
        fn()
    finally:
        gf._save = original
    fig = captured["fig"]
    ax = fig.axes[0]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    cards, texts, arrows, diamonds = [], [], [], []
    for p in ax.patches:
        if isinstance(p, FancyBboxPatch):
            cards.append((p, bounds(p, ax, renderer)))
        elif isinstance(p, FancyArrowPatch):
            arrows.append(p)
        elif isinstance(p, Polygon) and len(p.get_xy()) == 5:
            diamonds.append(p.get_xy()[:4])
    for t in ax.texts:
        if t.get_text().strip():
            texts.append((t, bounds(t, ax, renderer)))

    problems = []

    # text that spills out of the card it sits in
    for t, tb in texts:
        cx, cy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
        host = next((cb for _, cb in cards
                     if cb[0] <= cx <= cb[2] and cb[1] <= cy <= cb[3]), None)
        if host is None:
            continue
        if tb[0] < host[0] - PAD or tb[2] > host[2] + PAD:
            over = max(host[0] - tb[0], tb[2] - host[2])
            problems.append(f"text spills {over:.3f} past its card: "
                            f"{t.get_text()[:44]!r}")

    # text sitting on top of other text
    for i, (t1, b1) in enumerate(texts):
        for t2, b2 in texts[i + 1:]:
            if overlaps(b1, b2):
                problems.append(f"text overlaps text: {t1.get_text()[:26]!r} / "
                                f"{t2.get_text()[:26]!r}")

    # a diamond is narrower than its bounding box everywhere off the centre
    # line, so text that fits the box can still cross the slanted edges
    for pts in diamonds:
        cx = sum(x for x, _ in pts) / 4
        cy = sum(y for _, y in pts) / 4
        hw = max(x for x, _ in pts) - cx
        hh = max(y for _, y in pts) - cy
        for t, tb in texts:
            if not (cx - hw <= (tb[0] + tb[2]) / 2 <= cx + hw
                    and cy - hh <= (tb[1] + tb[3]) / 2 <= cy + hh):
                continue
            for x, y in ((tb[0], tb[1]), (tb[0], tb[3]), (tb[2], tb[1]), (tb[2], tb[3])):
                if abs(x - cx) / hw + abs(y - cy) / hh > 1.0:
                    problems.append(f"text crosses a diamond edge: "
                                    f"{t.get_text()[:36]!r}")
                    break

    # arrow tips that end inside a card instead of on its edge
    # the attribute is _posA_posB; the earlier _posB_posA spelling silently
    # disabled this whole check, so every arrow passed for free
    for a in arrows:
        if not hasattr(a, "_posA_posB"):
            problems.append("arrow has no endpoint pair; the check is not running")
            continue
        tip = a._posA_posB[1]
        for _, cb in cards:
            if cb[0] + PAD < tip[0] < cb[2] - PAD and cb[1] + PAD < tip[1] < cb[3] - PAD:
                problems.append(f"arrow tip inside a card at "
                                f"({tip[0]:.3f}, {tip[1]:.3f})")
                break

    print(f"=== {name} ===")
    print(f"  {len(cards)} cards, {len(texts)} text items, {len(arrows)} arrows")
    if problems:
        for p in sorted(set(problems)):
            print("   !!", p)
    else:
        print("   no geometric problems")
    print()
    return problems


def main():
    gf._style()
    total = audit(gf.figure_study_workflow, "Fig. 1  study workflow")
    total += audit(gf.figure_pseul_framework, "Fig. 2  PSEUL framework")
    print(f"{len(total)} problem(s) in total")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
