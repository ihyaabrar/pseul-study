from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import blended_transform_factory
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch, Polygon,
                                Rectangle)

from .study_registry import STUDY_SPECS


HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "outputs"
FIGURE_DIR = OUTPUT_ROOT / "figures"

COLORS = {
    "ink": "#344054",
    "muted": "#667085",
    "grid": "#E4E7EC",
    # The original series colours were pastels sitting in a luminance band of
    # 0.516 to 0.680: the worst pair separated by a contrast ratio of 1.29:1 and
    # mint against rose by 1.03:1, which is the same grey once the page is
    # printed or photocopied. ECTI-CIT publishes without reformatting, so that
    # ships as submitted. These four were chosen by searching for the spread that
    # maximises the weakest pairwise contrast while keeping every series at 3:1
    # or better against white paper; the weakest pair is now 1.46:1 and the
    # marker shapes carry identity alongside the hue.
    "blue": "#C0564C",
    "mint": "#5B9CB3",
    "peach": "#5A4210",
    "lavender": "#20262B",
    # the staged two-gate baseline; L=0.462 keeps the weakest greyscale pair at
    # 1.49:1, marginally better than the 1.46:1 the four-series palette already had
    "slate": "#B5B5B5",
    "rose": "#8794A0",
    "sand": "#9A7420",
    "pale": "#F7F8FA",
}

METHOD_COLORS = {
    "All Features": COLORS["blue"],
    "PSEUL-Select": COLORS["mint"],
    "PSEUL-Auto (leakage metadata null)": COLORS["peach"],
    "Oracle Trap Exclusion + SHAP": COLORS["lavender"],
    "Semantic Veto + SHAP": COLORS["slate"],
}

METHOD_MARKERS = {
    "All Features": "o",
    "PSEUL-Select": "s",
    "PSEUL-Auto (leakage metadata null)": "D",
    "Oracle Trap Exclusion + SHAP": "^",
    "Semantic Veto + SHAP": "v",
}

SHORT_METHOD = {
    "All Features": "All features",
    "PSEUL-Select": "PSEUL-Select",
    "PSEUL-Auto (leakage metadata null)": "PSEUL-Auto",
    "Oracle Trap Exclusion + SHAP": "Oracle exclusion",
    "Semantic Veto + SHAP": "Veto + SHAP",
}

SCENARIO_LABELS = {
    "adherence": "Adherence",
    "nhanes": "NHANES",
    "brfss": "BRFSS",
    "diabetes130_admission": "D130\nadmission",
    "diabetes130_discharge": "D130\ndischarge",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["ink"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _box(ax, xy, width, height, text, color, fontsize=8.5, radius=0.02):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=0.9,
        edgecolor=COLORS["grid"],
        facecolor=color,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        transform=ax.transAxes,
        zorder=3,
    )
    return patch


def _arrow(ax, start, end, connectionstyle="arc3,rad=0"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.1,
        color=COLORS["muted"],
        connectionstyle=connectionstyle,
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(arrow)


def _flow_helpers(ax, ink, muted):
    """Anchors, cards, diamonds and orthogonal routes shared by Figs 1 and 2.

    Every arrow used to stop at a card's nominal edge, which sits one boxstyle
    pad inside the edge that actually gets drawn, so twenty-four arrowheads were
    painted on top of the card they pointed at. Anchors are computed from the
    padded geometry here, so that cannot recur.
    """
    pad = 0.010

    def anchors(x, y, w, h):
        return {
            "left": (x - w / 2 - pad, y),
            "right": (x + w / 2 + pad, y),
            "top": (x, y + h / 2 + pad),
            "bottom": (x, y - h / 2 - pad),
        }

    def card(x, y, w, h, title, subtitle, edge, fill="white",
             title_size=10.9, subtitle_size=9.4):
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle=f"round,pad={pad},rounding_size=0.012",
                linewidth=1.25, edgecolor=edge, facecolor=fill,
                transform=ax.transAxes, zorder=3, clip_on=False,
            )
        )
        ax.text(x, y + 0.016, title, ha="center", va="center", fontsize=title_size,
                fontweight=600, color=ink, transform=ax.transAxes, zorder=4)
        ax.text(x, y - 0.019, subtitle, ha="center", va="center", fontsize=subtitle_size,
                color=muted, transform=ax.transAxes, zorder=4)
        return anchors(x, y, w, h)

    def terminator(x, y, label, w=0.115, h=0.060):
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle=f"round,pad={pad},rounding_size={h / 2}",
                linewidth=1.25, edgecolor=ink, facecolor="white",
                transform=ax.transAxes, zorder=3, clip_on=False,
            )
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=11.8, fontweight=600,
                color=ink, transform=ax.transAxes, zorder=4)
        return anchors(x, y, w, h)

    def diamond(x, y, label, edge, fill, w=0.280, h=0.124, fontsize=9.8):
        ax.add_patch(
            Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
                    closed=True, linewidth=1.25, edgecolor=edge, facecolor=fill,
                    transform=ax.transAxes, zorder=3, clip_on=False)
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, fontweight=500,
                color=ink, linespacing=1.2, transform=ax.transAxes, zorder=4)
        return {"left": (x - w / 2, y), "right": (x + w / 2, y),
                "top": (x, y + h / 2), "bottom": (x, y - h / 2)}

    def route(points, *, dashed=False, color=None, arrow=True):
        color = muted if color is None else color
        style = (0, (4, 3)) if dashed else "solid"
        last = len(points) - 2 if arrow else len(points) - 1
        for i in range(last):
            ax.plot([points[i][0], points[i + 1][0]], [points[i][1], points[i + 1][1]],
                    color=color, linewidth=1.25, linestyle=style, solid_capstyle="round",
                    solid_joinstyle="round", transform=ax.transAxes, clip_on=False, zorder=1)
        if arrow:
            ax.add_patch(
                FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=11,
                                linewidth=1.25, linestyle=style, color=color,
                                transform=ax.transAxes, zorder=5, shrinkA=0, shrinkB=0)
            )

    def junction(x, y, color):
        ax.add_patch(Circle((x, y), 0.0055, facecolor=color, edgecolor="none",
                            transform=ax.transAxes, zorder=6))

    return card, terminator, diamond, route, junction


def figure_study_workflow() -> None:
    """Render the revision-v2 workflow as a three-lane layered flowchart.

    Laid out against aesthetics that have been measured on readers rather than
    ones that feel right. All three lanes share a single six-row grid, so the
    composition carries the invisible alignment lines Wong describes in "Design
    of data figures" (Nat Methods 7:665, 2010); the earlier version used fifteen
    distinct row heights for sixteen boxes and so lined up with nothing. Routes
    are assigned to explicit vertical channels ordered so that none cross, which
    is the horizontal-coordinate phase of the Sugiyama, Tagawa and Toda framework
    (IEEE T-SMC 11(2):109-125, 1981) and the aesthetic Purchase (Graph Drawing
    1997, LNCS 1353:248-261) found dominates comprehension.
    """

    fig, ax = plt.subplots(figsize=(10.0, 6.55))
    ax.set_axis_off()

    ink, muted = "#26323F", "#6B7785"
    data_edge, model_edge, audit_edge = "#738493", "#2F748B", "#4E8B6A"
    decision_edge, rose_edge = "#B8892B", "#B75A62"
    decision_fill, rose_fill = "#FFF9EA", "#FFF4F5"

    card, terminator, diamond, route, junction = _flow_helpers(ax, ink, muted)

    R = (0.875, 0.735, 0.595, 0.455, 0.315, 0.175)
    LX, MX, AX = 0.170, 0.505, 0.845          # lane centres
    LW, MW, AW = 0.275, 0.275, 0.295          # widths, sized to each lane's longest label
    H = 0.094
    SEAL_CH, LOOP_CH, YES_CH, HANDOFF_CH = 0.010, 0.337, 0.662, 0.676

    for x, label, color in ((LX, "DATA PREPARATION", data_edge),
                            (MX, "NESTED MODELLING", model_edge),
                            (AX, "AUDIT & SYNTHESIS", audit_edge)):
        ax.text(x, 0.968, label, ha="center", va="center", fontsize=11.4,
                fontweight=600, color=color, transform=ax.transAxes)

    # ---- data preparation --------------------------------------------------
    start = terminator(LX, R[0], "Start")
    freeze = card(LX, R[1], LW, H, "Freeze analysis registry",
                  "population · outcome · landmark", data_edge)
    split = card(LX, R[2], LW, H, "Stratified 80:20 split", "fixed random seed 42", data_edge)
    dev = card(LX, R[3], LW, H, "Development partition", "model development only", data_edge)
    sealed = card(LX, R[5], LW, H, "Reserved internal test", "sealed until the end",
                  rose_edge, fill=rose_fill)

    route([start["bottom"], freeze["top"]])
    route([freeze["bottom"], split["top"]])
    route([split["bottom"], dev["top"]])
    route([split["left"], (SEAL_CH, R[2]), (SEAL_CH, R[5]), sealed["left"]], color=rose_edge)
    ax.text(SEAL_CH + 0.012, R[4] + 0.015, "20% sealed", fontsize=9.6, color=rose_edge,
            ha="left", va="center", fontstyle="italic", transform=ax.transAxes)

    # ---- nested modelling --------------------------------------------------
    outer = card(MX, R[0], MW, H, "Outer training fold", "shared across methods", model_edge)
    prep = card(MX, R[1], MW, H, "Train-only preprocessing", "fold-local imputation", model_edge)
    inner = card(MX, R[2], MW, H, "Inner selector fitting", "3 folds · PSEUL + comparators",
                 model_edge)
    predict = card(MX, R[3], MW, H, "Predict outer validation", "one OOF per record", model_edge)
    folds = diamond(MX, R[4], "All outer folds\ncomplete?", decision_edge, decision_fill)

    route([outer["bottom"], prep["top"]])
    route([prep["bottom"], inner["top"]])
    route([inner["bottom"], predict["top"]])
    route([predict["bottom"], folds["top"]])

    # The development partition and the fold loop both terminate at the outer
    # training fold, so they share one climbing channel and one arrowhead.
    route([dev["right"], (LOOP_CH, R[3])], arrow=False)
    junction(LOOP_CH, R[3], muted)
    route([folds["left"], (LOOP_CH, R[4]), (LOOP_CH, R[0]), outer["left"]])
    ax.text(LOOP_CH + 0.006, R[4] + 0.010, "no", fontsize=9.6, color=muted,
            ha="left", va="bottom", transform=ax.transAxes)

    # ---- audit and synthesis -----------------------------------------------
    pool = card(AX, R[0], AW, H, "Pool OOF predictions", "paired OOF performance", audit_edge)
    audit = card(AX, R[1], AW, H, "Audit selected subsets", "stability · traps", audit_edge)
    refit = card(AX, R[2], AW, H, "Freeze and refit pipeline", "full development set", audit_edge)
    final = card(AX, R[3], AW, H, "Internal-test estimate", "single evaluation",
                 rose_edge, fill=rose_fill)
    synth = card(AX, R[4], AW, H, "Landmark-aware synthesis", "cost · limitations", audit_edge)
    end = terminator(AX, R[5], "End")

    route([folds["right"], (YES_CH, R[4]), (YES_CH, R[0]), pool["left"]], color=audit_edge)
    # beside the climbing channel rather than at the branch corner: the widened
    # diamond leaves only 0.017 there, and both the yes channel and the sealed
    # hand-off pass through it
    ax.text(YES_CH - 0.006, R[4] + 0.062, "yes", fontsize=9.6, color=audit_edge,
            ha="right", va="center", transform=ax.transAxes)
    route([pool["bottom"], audit["top"]])
    route([audit["bottom"], refit["top"]])
    route([refit["bottom"], final["top"]])
    route([final["bottom"], synth["top"]])
    route([synth["bottom"], end["top"]])

    # The sealed partition reaches the final estimate through its own low
    # corridor, climbing in its own channel to the right of the yes branch, so
    # it crosses nothing on the way.
    route([sealed["right"], (0.330, R[5]), (0.330, 0.055), (HANDOFF_CH, 0.055),
           (HANDOFF_CH, R[3]), final["left"]], dashed=True, color=rose_edge)
    ax.text(0.500, 0.068, "sealed-test hand-off,\nafter the full-development refit",
            ha="center", va="bottom", fontsize=9.6, fontstyle="italic", color=rose_edge,
            linespacing=1.25, transform=ax.transAxes)

    _save(fig, "fig01_study_workflow")


def figure_pseul_framework() -> None:
    """Render the PSEUL formula map in the same visual system as Fig. 1.

    The previous version carried two edge crossings, both produced by routing the
    minimum-utility gate down the same corridor that carries the availability and
    leakage lanes upward. Ordering the signal column so that each source sits
    above its target makes that corridor planar: reading top to bottom, the
    channels now target rows 1, 1, 1, 1, 2, 3, 4, which admits a crossing-free
    routing. The gate's dependence on U_j is stated in the soft-controls subtitle
    rather than drawn as the one edge that inverts the order.
    """

    fig, ax = plt.subplots(figsize=(10.0, 6.55))
    ax.set_axis_off()

    ink, muted = "#26323F", "#6B7785"
    signal_edge, score_edge, output_edge = "#738493", "#2F748B", "#4E8B6A"
    decision_edge, rose_edge, auto_edge = "#B8892B", "#B75A62", "#B47A46"
    decision_fill, rose_fill, auto_fill = "#FFF9EA", "#FFF4F5", "#FFF7EF"

    card, terminator, diamond, route, junction = _flow_helpers(ax, ink, muted)

    R = (0.870, 0.735, 0.600, 0.465, 0.330, 0.195, 0.060)
    SX, CX, OX = 0.165, 0.505, 0.845
    SW, CW, OW = 0.255, 0.290, 0.290
    H = 0.090
    BUS_CH, AVAIL_CH, LEAK_CH = 0.313, 0.326, 0.339   # ordered so that none cross
    EXCL_A_CH, EXCL_V_CH, RERUN_CH = 0.668, 0.677, 0.671
    TS, SS = 10.4, 9.0                                 # title and subtitle sizes

    def small(x, y, w, h, title, subtitle, edge, fill="white"):
        return card(x, y, w, h, title, subtitle, edge, fill=fill,
                    title_size=TS, subtitle_size=SS)

    for x, label, color in ((SX, "SIGNAL ESTIMATION", signal_edge),
                            (CX, "SCORING & CONTROLS", score_edge),
                            (OX, "OUTPUTS & AUDIT", output_edge)):
        ax.text(x, 0.968, label, ha="center", va="center", fontsize=11.4,
                fontweight=600, color=color, transform=ax.transAxes)

    # ---- signal estimation, Eqs. (1)-(8) -----------------------------------
    p = small(SX, R[0], SW, H, r"Predictive utility  $P_j$", "permutation AUC loss", signal_edge)
    s = small(SX, R[1], SW, H, r"SHAP stability  $S_j$", "fold consistency", signal_edge)
    e = small(SX, R[2], SW, H, r"Evidence relevance  $E_j$", "evidence + topic similarity",
              signal_edge)
    u = small(SX, R[3], SW, H, r"Author-rated utility  $U_j$", "intervention · feasibility",
              signal_edge)
    a = small(SX, R[4], SW, H, r"Availability  $A_j$", "at the prediction landmark",
              rose_edge, fill=rose_fill)
    leak = small(SX, R[5], SW, H, r"Leakage controls  $L_{jk},V_j$",
                 r"LD · DO · TP → $V_j$;  TA → $L_{jk}$", rose_edge, fill=rose_fill)
    auto_in = small(SX, R[6], SW, H, "PSEUL-Auto control", r"$LD=DO=TP=0$",
                    auto_edge, fill=auto_fill)

    # ---- scoring and controls ----------------------------------------------
    hj = small(CX, R[0], CW, H, r"Audit score  $H_j$",
               r"$\alpha P_j+\beta S_j+\gamma E_j+\delta U_j$", score_edge)
    gate_a = diamond(CX, R[1], "Available at landmark?", decision_edge, decision_fill,
                     w=0.290, h=0.104)
    gate_v = diamond(CX, R[2], "Semantic veto absent?", decision_edge, decision_fill,
                     w=0.290, h=0.104)
    soft = small(CX, R[3], CW, H, r"Soft controls  $G_j × Q_{jk}$",
                 r"utility gate on $U_j$ × leakage penalty", rose_edge, fill=rose_fill)
    score = small(CX, R[4], CW, H, r"SelectScore$_j(\mathcal{S})$", "fold mean · redundancy",
                  score_edge)
    greedy = small(CX, R[5], CW, H, "Greedy subset construction",
                   r"≤ $k_{\max}$ · stop below $\tau_S$", score_edge)

    # The four scoring signals tap one bus instead of running four parallel lanes.
    route([p["right"], (BUS_CH, R[0])], arrow=False)
    for anchor in (s, e, u):
        route([anchor["right"], (BUS_CH, anchor["right"][1])], arrow=False)
        junction(BUS_CH, anchor["right"][1], muted)
    route([(BUS_CH, R[3]), (BUS_CH, R[0]), hj["left"]])

    route([a["right"], (AVAIL_CH, R[4]), (AVAIL_CH, R[1]), gate_a["left"]], color=rose_edge)
    route([leak["right"], (LEAK_CH, R[5]), (LEAK_CH, R[3])], arrow=False, color=rose_edge)
    junction(LEAK_CH, R[3], rose_edge)
    route([(LEAK_CH, R[3]), soft["left"]], color=rose_edge)
    route([(LEAK_CH, R[3]), (LEAK_CH, R[2]), gate_v["left"]], color=rose_edge)

    route([hj["bottom"], gate_a["top"]])
    route([gate_a["bottom"], gate_v["top"]], color=output_edge)
    route([gate_v["bottom"], soft["top"]], color=output_edge)
    route([soft["bottom"], score["top"]])
    route([score["bottom"], greedy["top"]])
    for y, cond in ((R[1], r"yes · $A_j=1$"), (R[2], r"yes · $V_j=0$")):
        ax.text(CX + 0.009, y - 0.069, cond, fontsize=9.2, color=output_edge,
                ha="left", va="center", transform=ax.transAxes)

    # PSEUL-Auto nulls the semantic metadata and reruns the identical path.
    route([auto_in["top"], leak["bottom"]], dashed=True, color=auto_edge)

    # ---- outputs and audit --------------------------------------------------
    audit_out = small(OX, R[0], OW, H, "PSEUL-Audit", "diagnostic only", output_edge)
    excl = small(OX, (R[1] + R[2]) / 2, OW, 0.150, "Hard exclusion",
                 r"record $A_j=0$ or $V_j=1$", rose_edge, fill=rose_fill)
    select = small(OX, R[5], OW, H, "PSEUL-Select", "registry-concordant subset", output_edge)
    auto_out = small(OX, R[6], OW, H, "PSEUL-Auto", "null-control rerun",
                     auto_edge, fill=auto_fill)

    route([hj["right"], audit_out["left"]], color=output_edge)
    route([gate_a["right"], (EXCL_A_CH, R[1]), (EXCL_A_CH, 0.700), (excl["left"][0], 0.700)],
          color=rose_edge)
    route([gate_v["right"], (EXCL_V_CH, R[2]), (EXCL_V_CH, 0.630), (excl["left"][0], 0.630)],
          color=rose_edge)
    # each label goes on the side its route leaves from, so neither sits in the
    # corner where the route turns towards the exclusion card
    for anchor, y, dy, va in ((gate_a, R[1], 0.010, "bottom"), (gate_v, R[2], -0.010, "top")):
        ax.text(anchor["right"][0] + 0.007, y + dy, "no", fontsize=9.2,
                color=rose_edge, ha="left", va=va, transform=ax.transAxes)

    route([greedy["right"], (RERUN_CH, R[5])], arrow=False, color=output_edge)
    junction(RERUN_CH, R[5], output_edge)
    route([(RERUN_CH, R[5]), select["left"]], color=output_edge)
    route([(RERUN_CH, R[5]), (RERUN_CH, R[6]), auto_out["left"]], dashed=True, color=auto_edge)
    ax.text(RERUN_CH + 0.005, (R[5] + R[6]) / 2, "rerun", fontsize=9.2, fontstyle="italic",
            color=auto_edge, ha="left", va="center", transform=ax.transAxes)

    # The output lane has nothing to show opposite the two soft-scoring rows,
    # because neither produces an output. That gap holds the key instead, which
    # the diagram previously left the reader to infer from the caption.
    kx, ky, kw, kh = OX, (R[3] + R[4]) / 2, OW, 0.150
    ax.add_patch(
        FancyBboxPatch((kx - kw / 2, ky - kh / 2), kw, kh,
                       boxstyle="round,pad=0.010,rounding_size=0.012", linewidth=0.9,
                       edgecolor=COLORS["grid"], facecolor="#FAFBFC",
                       transform=ax.transAxes, zorder=2)
    )
    ax.text(kx, ky + 0.055, "KEY", ha="center", va="center", fontsize=8.8, fontweight=600,
            color=muted, transform=ax.transAxes, zorder=4)
    for dy, color, dashed, label in ((0.022, muted, False, "registered analysis path"),
                                     (-0.014, rose_edge, False, "leakage-driven exclusion"),
                                     (-0.050, auto_edge, True, "metadata-null rerun")):
        ax.plot([kx - kw / 2 + 0.018, kx - kw / 2 + 0.060], [ky + dy, ky + dy], color=color,
                linewidth=1.25, linestyle=(0, (4, 3)) if dashed else "solid",
                transform=ax.transAxes, zorder=4)
        ax.text(kx - kw / 2 + 0.070, ky + dy, label, ha="left", va="center", fontsize=8.8,
                color=ink, transform=ax.transAxes, zorder=4)

    _save(fig, "fig02_pseul_framework")


def _load_combined() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled = pd.read_csv(OUTPUT_ROOT / "combined_pooled_oof_metrics.csv")
    traps = pd.read_csv(OUTPUT_ROOT / "combined_trap_performance.csv")
    stability = pd.read_csv(OUTPUT_ROOT / "combined_selection_stability.csv")
    return pooled, traps, stability


def _clean_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    ax.set_axisbelow(True)


PLOTTED_METHODS = [
    "All Features",
    "PSEUL-Select",
    "PSEUL-Auto (leakage metadata null)",
    "Oracle Trap Exclusion + SHAP",
    "Semantic Veto + SHAP",
]

MATRIX_METHOD_LABELS = {
    "All Features": "All\nfeatures",
    "PSEUL-Select": "PSEUL-\nSelect",
    "PSEUL-Auto (leakage metadata null)": "PSEUL-\nAuto",
    "Oracle Trap Exclusion + SHAP": "Oracle\nexclusion",
    "Semantic Veto + SHAP": "Veto +\nSHAP",
}


def _relative_luminance(rgb) -> float:
    """WCAG 2.1 relative luminance, used to pick readable text over a cell."""
    channels = []
    for c in rgb[:3]:
        channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ramp(name, end):
    """White to `end`. A single-hue ramp is monotone in luminance, so it keeps
    its ordering when ECTI prints the page in greyscale."""
    return LinearSegmentedColormap.from_list(name, ["#FFFFFF", end])


def figure_enforcement_cost() -> None:
    """Discrimination given up when the feature-safety contract is enforced.

    This was a line chart across the five scenarios. The scenarios are separate
    datasets, not an ordered variable, so the connecting lines drew a trend that
    does not exist -- the fall from BRFSS to D130 read as a decline rather than a
    change of dataset, which is the failure Rougier, Droettboom and Bourne warn
    against in rule 7 (PLoS Comput Biol 10:e1003833, 2014). What the section
    actually reports is a paired difference per scenario, so the paired
    difference is what the figure now encodes: a shaded band spanning the AUC
    given up, with the delta stated beside it.
    """
    pooled, _, _ = _load_combined()
    studies = list(STUDY_SPECS)
    table = (pooled[pooled["method"].isin(PLOTTED_METHODS)]
             .pivot_table(index="study", columns="method", values="auc")
             .reindex(studies))

    fig, ax = plt.subplots(figsize=(9.4, 3.9))
    y = np.arange(len(studies))
    offsets = np.linspace(-0.28, 0.28, len(PLOTTED_METHODS))

    for row, study in enumerate(studies):
        lo = min(table.loc[study, "All Features"], table.loc[study, "PSEUL-Select"])
        hi = max(table.loc[study, "All Features"], table.loc[study, "PSEUL-Select"])
        ax.add_patch(
            Rectangle((lo, row - 0.38), hi - lo, 0.76, facecolor="#FBEEF0",
                      edgecolor="none", zorder=0)
        )
    outside = blended_transform_factory(ax.transAxes, ax.transData)
    for row, study in enumerate(studies):
        delta = table.loc[study, "PSEUL-Select"] - table.loc[study, "All Features"]
        ax.text(1.035, row, f"{delta:+.3f}", ha="left", va="center", fontsize=8.5,
                color=COLORS["blue"], fontweight=500, transform=outside)

    for offset, method in zip(offsets, PLOTTED_METHODS):
        ax.scatter(table[method], y + offset, s=46, marker=METHOD_MARKERS[method],
                   color=METHOD_COLORS[method], edgecolor=COLORS["ink"], linewidth=0.4,
                   label=SHORT_METHOD[method], zorder=3)

    # no in-figure title: the caption beneath already names the figure, and Figs 1
    # and 2 carry none either
    ax.set_xlabel("Pooled out-of-fold ROC AUC")
    ax.set_yticks(y, [SCENARIO_LABELS[s].replace(chr(10), " ") for s in studies])
    ax.set_xlim(0.54, 1.01)
    ax.set_ylim(len(studies) - 0.5, -0.5)
    ax.text(1.035, -0.62, "Δ all → Select", ha="left", va="center", fontsize=8.5,
            style="italic", color=COLORS["muted"], transform=outside)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              handletextpad=0.3, columnspacing=1.2)
    # savefig trims to content, so these margins only set the legend's distance
    # from the axis; the embedded height is fixed by what is drawn, not by them
    fig.subplots_adjust(bottom=0.28, left=0.13, right=0.90, top=0.86)
    _save(fig, "fig03_enforcement_cost")


def _matrix_panel(ax, table, studies, cmap, title, vmin=0.0, note=None):
    values = table.to_numpy(dtype=float)
    ax.imshow(values, cmap=cmap, vmin=vmin, vmax=1.0, aspect="auto",
              interpolation="nearest")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(v):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#F2F4F7",
                                       edgecolor=COLORS["grid"], linewidth=1.2, zorder=2))
                ax.text(j, i, "—", ha="center", va="center", fontsize=9,
                        color=COLORS["muted"], zorder=3)
                continue
            face = cmap((v - vmin) / (1.0 - vmin))
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.6,
                    fontweight=500, zorder=3,
                    color="white" if _relative_luminance(face) < 0.42 else COLORS["ink"])
    ax.set_xticks(range(values.shape[1]),
                  [MATRIX_METHOD_LABELS[m] for m in table.columns], fontsize=8.4)
    ax.set_yticks(range(len(studies)),
                  [SCENARIO_LABELS[s].replace(chr(10), " ") for s in studies], fontsize=9)
    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color=COLORS["grid"], linewidth=1.2)
    ax.tick_params(which="both", length=0)
    ax.spines[:].set_visible(False)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight=500, pad=8)
    if note:
        ax.text(0.0, 1.0, note, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.6, style="italic", color=COLORS["muted"])


def figure_traps_and_stability() -> None:
    """Trap retention and fold-level selection stability, as printed matrices.

    Both quantities are one number per scenario per method -- a complete 5 x 5
    grid -- and both were drawn as offset scatters, where the encoding failed on
    ties. Of the twenty-five stability values, seventeen share an x position with
    another marker, so the reader saw a vertical stack at Jaccard 1.00 and had to
    match marker shapes against a legend to read it; and thirty-two of the
    forty-four retention values are exactly 0 or 1, which is a near-binary
    quantity spread over a continuous axis. A small complete grid with its values
    printed is unambiguous at both, and the single-hue ramps stay ordered in
    greyscale.
    """
    _, traps, stability = _load_combined()
    studies = list(STUDY_SPECS)

    def grid(df, column):
        return (df[df["method"].isin(PLOTTED_METHODS)]
                .pivot_table(index="study", columns="method", values=column)
                .reindex(index=studies, columns=PLOTTED_METHODS))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.5),
                             gridspec_kw={"wspace": 0.10, "width_ratios": [1, 1]})
    _matrix_panel(axes[0], grid(traps, "mean_trap_selection_frequency"), studies,
                  _ramp("traps", COLORS["blue"]),
                  "a  Designated traps retained   ·   higher is worse")
    _matrix_panel(axes[1], grid(stability, "mean_pairwise_jaccard"), studies,
                  _ramp("stability", COLORS["mint"]),
                  "b  Selection stability, mean pairwise Jaccard   ·   higher is steadier",
                  vmin=0.60)
    axes[1].set_yticklabels([])
    fig.text(0.008, 0.005, "—  no designated traps in that scenario        "
             "shading: panel a 0–1, panel b 0.60–1", fontsize=8.6,
             style="italic", color=COLORS["muted"])
    fig.subplots_adjust(left=0.135, right=0.995, top=0.84, bottom=0.13)
    _save(fig, "fig04_traps_and_stability")


def figure_veto_threshold_sensitivity() -> None:
    """How far the supplied semantic ratings can move before the result does.

    The paper's central limitation is that exclusion follows deterministically
    from author-supplied ratings, so the quantity a reader needs is the width of
    the interval around the locked operating point over which nothing changes.
    Unlike the scenario axis in Fig. 3, tau_V is an ordered variable and the veto
    is a threshold test, so a step function is the faithful encoding: the
    admissible set is piecewise constant in tau_V and changes only where a rating
    is crossed.

    Every value is derived here rather than restated: panel b takes the refitted
    AUC wherever the admissible set differs from the locked one and the locked
    AUC everywhere else, and the mapping is asserted to cover the whole sweep.
    """
    sens = pd.read_csv(OUTPUT_ROOT / "tau_v_sensitivity.csv")
    refit = pd.read_csv(OUTPUT_ROOT / "tau_v_refit.csv")
    pooled, _, _ = _load_combined()
    locked = (pooled[pooled["method"].eq("PSEUL-Select")]
              .set_index("study")["auc"].to_dict())

    studies = list(STUDY_SPECS)
    colours = [COLORS["blue"], COLORS["mint"], COLORS["peach"],
               COLORS["lavender"], COLORS["slate"]]
    markers = ["o", "s", "D", "^", "v"]
    # NHANES and BRFSS both settle at AUC 0.793, so colour alone leaves one drawn
    # underneath the other; the dash patterns separate them where they coincide
    # and carry the identity into greyscale as well
    dashes = ["solid", (0, (5, 2)), (0, (1, 1.6)), (0, (6, 1.6, 1, 1.6)), (0, (3, 1.4))]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4),
                             gridspec_kw={"wspace": 0.24})

    flat = []
    for study, colour, marker, dash in zip(studies, colours, markers, dashes):
        g = sens[sens.study.eq(study)].sort_values("tau_v")
        taus = g.tau_v.to_numpy()
        n_traps = int(g.n_traps.iloc[0])

        # Panel a carries only the scenarios that move. Three of the five sit at
        # zero across the whole sweep, and drawing them stacks three flat lines on
        # one another, none of them readable; naming them says the same thing.
        admissible = (n_traps - g.traps_vetoed.to_numpy()) / n_traps if n_traps else None
        if admissible is not None and admissible.max() > 0:
            axes[0].plot(taus, admissible, drawstyle="steps-post", marker=marker,
                         markersize=4.5, linewidth=1.7, color=colour, linestyle=dash,
                         markeredgecolor=COLORS["ink"], markeredgewidth=0.4)
        else:
            flat.append(SCENARIO_LABELS[study].replace(chr(10), " "))

        # panel b: the admissible set is piecewise constant, so one refit per
        # distinct set covers the sweep; the locked value stands elsewhere
        aucs = []
        for _, row in g.iterrows():
            same = g[g.n_vetoed.eq(row.n_vetoed)].tau_v
            hit = refit[refit.study.eq(study) & refit.tau_v.isin(same)]
            aucs.append(float(hit.auc.iloc[0]) if len(hit) else locked[study])
        assert len(aucs) == len(taus)
        axes[1].plot(taus, aucs, drawstyle="steps-post", marker=marker, markersize=4.5,
                     linewidth=1.7, color=colour, linestyle=dash,
                     markeredgecolor=COLORS["ink"], markeredgewidth=0.4,
                     label=SCENARIO_LABELS[study].replace(chr(10), " "))

    for ax, title in ((axes[0], "a  Designated traps left admissible"),
                      (axes[1], "b  PSEUL-Select pooled out-of-fold AUC")):
        ax.axvline(0.85, color=COLORS["muted"], linewidth=1.0, linestyle=(0, (3, 3)),
                   zorder=0)
        ax.set_title(title, loc="left", fontsize=10.2, fontweight=500)
        ax.set_xlabel(r"semantic-veto threshold  $\tau_V$")
        ax.set_xlim(0.575, 0.975)
        ax.set_xticks([0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
        _clean_axes(ax)
    axes[0].set_ylabel("fraction of designated traps")
    axes[0].set_ylim(-0.10, 1.30)
    axes[0].set_yticks([0.0, 0.5, 1.0])
    axes[1].set_ylabel("ROC AUC")
    axes[1].set_ylim(0.545, 0.925)
    for ax in axes:
        ax.text(0.858, ax.get_ylim()[1], "locked", fontsize=8.6, color=COLORS["muted"],
                style="italic", ha="left", va="top")
    axes[0].text(0.583, 1.26, "zero across the whole sweep:" + chr(10) + ", ".join(flat),
                 fontsize=8.6, color=COLORS["muted"], style="italic", ha="left", va="top",
                 linespacing=1.3)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), handletextpad=0.3, columnspacing=1.4)
    fig.subplots_adjust(bottom=0.30, top=0.90, left=0.075, right=0.99)
    _save(fig, "fig05_veto_threshold_sensitivity")


def figure_structural_tradeoff() -> None:
    pooled, traps, _ = _load_combined()
    # this retired figure carries its own jitter and label maps for the
    # original four series, so it is pinned to them rather than following
    # METHOD_COLORS, which now also holds the staged baseline
    methods = ["All Features", "PSEUL-Select",
               "PSEUL-Auto (leakage metadata null)",
               "Oracle Trap Exclusion + SHAP"]
    # Eight points across two panels do not need a third of the page. The data
    # sit in the top third of the y-range and cluster at either end of x, so the
    # canvas was mostly empty; a wider aspect keeps every point legible and
    # returns roughly an inch of column height to the text.
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 2.9), gridspec_kw={"wspace": 0.25})
    jitter = {
        "All Features": (0.012, 0.0008),
        "PSEUL-Select": (-0.010, 0.0008),
        "PSEUL-Auto (leakage metadata null)": (-0.012, -0.0008),
        "Oracle Trap Exclusion + SHAP": (0.010, -0.0008),
    }
    markers = {
        "All Features": "o",
        "PSEUL-Select": "s",
        "PSEUL-Auto (leakage metadata null)": "D",
        "Oracle Trap Exclusion + SHAP": "^",
    }
    label_offsets = {
        "All Features": (7, 8),
        "PSEUL-Select": (7, 8),
        "PSEUL-Auto (leakage metadata null)": (7, -13),
        "Oracle Trap Exclusion + SHAP": (7, -13),
    }
    for ax, study, title in zip(axes, ["nhanes", "brfss"], ["a  NHANES definitional overlap", "b  BRFSS skip-pattern leakage"]):
        for method in methods:
            auc = float(pooled[(pooled.study == study) & (pooled.method == method)]["auc"].iloc[0])
            trap = float(
                traps[(traps.study == study) & (traps.method == method)]["mean_trap_selection_frequency"].iloc[0]
            )
            display_x = trap + jitter[method][0]
            display_y = auc + jitter[method][1]
            ax.scatter(
                display_x,
                display_y,
                s=70,
                marker=markers[method],
                color=METHOD_COLORS[method],
                edgecolor=COLORS["ink"],
                linewidth=0.5,
                zorder=3,
            )
            ax.annotate(
                SHORT_METHOD[method],
                (display_x, display_y),
                xytext=label_offsets[method],
                textcoords="offset points",
                fontsize=7.2,
                color=COLORS["ink"],
            )
        ax.set_title(title, loc="left", fontweight=500)
        ax.set_xlabel("Designated-trap selection frequency")
        ax.set_ylabel("Pooled ROC AUC")
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(0.76, 1.03)
        _clean_axes(ax)
    _save(fig, "fig05_structural_leakage_tradeoff")


def figure_landmark_scenarios() -> None:
    pooled, _, _ = _load_combined()
    methods = list(METHOD_COLORS)
    admission_metrics = pd.read_csv(
        OUTPUT_ROOT / "diabetes130_admission" / "full" / "nested_fold_metrics.csv"
    ).groupby("method", as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"wspace": 0.30})
    x = np.arange(len(methods))
    width = 0.34
    admission = pooled[pooled.study.eq("diabetes130_admission")].set_index("method").reindex(methods)["auc"]
    discharge = pooled[pooled.study.eq("diabetes130_discharge")].set_index("method").reindex(methods)["auc"]
    axes[0].bar(x - width / 2, admission, width, color=COLORS["rose"], label="Admission landmark")
    axes[0].bar(x + width / 2, discharge, width, color=COLORS["mint"], label="Discharge landmark")
    axes[0].set_title("a  Landmark-specific scenario estimates", loc="left", fontweight=500)
    axes[0].set_ylabel("Pooled ROC AUC")
    axes[0].set_xticks(x, [SHORT_METHOD[m] for m in methods], rotation=22, ha="right")
    axes[0].set_ylim(0.50, 0.68)
    axes[0].legend(frameon=False, fontsize=8)
    _clean_axes(axes[0])

    unavailable = admission_metrics.set_index("method").reindex(methods)["prediction_unavailable_selected"]
    axes[1].barh(
        np.arange(len(methods)),
        unavailable,
        color=[METHOD_COLORS[method] for method in methods],
        edgecolor="none",
    )
    axes[1].set_title("b  Admission-time availability violations", loc="left", fontweight=500)
    axes[1].set_xlabel("Mean unavailable features selected")
    axes[1].set_yticks(np.arange(len(methods)), [SHORT_METHOD[m] for m in methods])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(11.5, float(unavailable.max()) + 0.5))
    axes[1].grid(axis="x", color=COLORS["grid"], linewidth=0.7)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    for index, value in enumerate(unavailable):
        axes[1].text(value + 0.15, index, f"{value:.1f}", va="center", fontsize=8)
    _save(fig, "fig06_landmark_scenarios")


def main() -> None:
    _style()
    figure_study_workflow()
    figure_pseul_framework()
    figure_enforcement_cost()
    figure_traps_and_stability()
    figure_veto_threshold_sensitivity()
    figure_structural_tradeoff()
    figure_landmark_scenarios()
    captions = {
        "Fig. 1": "Leakage-resistant nested evaluation workflow. Five shared outer folds generate out-of-fold predictions; three inner folds estimate PSEUL signals within each outer-training partition. After the development analysis is frozen, the selector and classifier are refitted on the full development partition and evaluated once on the reserved internal test. Dashed rose arrows denote the isolated test hand-off.",
        "Fig. 2": "PSEUL scoring and decision framework. Predictive utility, SHAP stability, evidence relevance, and author-rated utility form the additive audit score. Prediction-landmark availability and semantic veto provide separate hard controls; the leakage penalty and redundancy determine the selected subset, while the utility gate and score floor are defined but were inactive in every reported run. The dashed Auto branch reruns the same pipeline after setting LD, DO, and TP to zero while retaining all other inputs.",
        # UNITSTOTAL is the proxy that was vetoed; ANNUALCLAIMAMOUNT is the one that
        # survived. The previous wording had the pair the wrong way round, and this
        # file ships with the reproduction package.
        "Fig. 3": "Pooled out-of-fold discrimination and designated-trap retention across five scenarios, with the staged two-gate baseline plotted alongside PSEUL-Select. PSEUL-Select removes every designated trap except the moderate-risk adherence proxy ANNUALCLAIMAMOUNT, which its supplied rating places below the veto threshold.",
        "Fig. 4": "Selection stability across outer folds, measured by mean pairwise Jaccard similarity.",
        "Fig. 5": "Performance-leakage trade-off in the two structural stress tests. Apparent discrimination approaches perfection when definitional or skip-pattern traps are retained.",
        "Fig. 6": "Diabetes 130 landmark-specific scenario audit. Both scenarios use the same top-k ceiling, but eligibility differs; contrasts are descriptive and do not isolate a causal effect of changing the landmark.",
    }
    (FIGURE_DIR / "captions.json").write_text(json.dumps(captions, indent=2), encoding="utf-8")
    print(f"figures -> {FIGURE_DIR}")


if __name__ == "__main__":
    main()
