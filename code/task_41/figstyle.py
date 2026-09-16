# ==============================================================================
# AI ASSISTANCE DISCLAIMER
# In accordance with academic guidelines, the author acknowledges the use of
# generative artificial intelligence assistance (DeepMind's Antigravity AI assistant)
# for writing assistance, code development, data formatting, and proofreading the
# manuscript. Some analytical derivations, mathematical proofs, numerical simulations,
# and physical interpretations still need rigorous verification by the author.
# ==============================================================================
"""Shared figure style. One system across both write-ups.

Categorical hues are the validated default palette, assigned in fixed order and
never cycled: slots 1-3 clear the all-pairs CVD and normal-vision floors, so any
scatter or small-multiple panel uses at most those three.  Panels with up to five
line/bar series use the adjacent-pair set.  Several slots sit below 3:1 contrast
on a white surface, so every panel with two or more series carries a legend and,
where there is room, a direct label - identity is never carried by colour alone.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated categorical slots (light surface)
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
CAT3 = [C1, C2, C3]                 # all-pairs safe
CAT5 = [C1, C2, C3, C4, C5]         # adjacent-pairs safe
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
GRID = "#e3e2de"
SEQ = "viridis"                     # single-hue-ish, perceptually uniform
NULLC = "#9a9995"

TIER = {"Core": C2, "Sub-core": C1, "Halo": C3}

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5,
    "axes.titlesize": 9, "axes.labelsize": 8.5, "axes.titleweight": "normal",
    "axes.edgecolor": INK2, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.frameon": False, "legend.fontsize": 7.2, "legend.handlelength": 1.6,
    "lines.linewidth": 1.6, "lines.markersize": 3.6,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "mathtext.fontset": "dejavuserif",
})


def tidy(ax, grid="y"):
    """Recessive axes: no top/right spine, a single light grid behind the marks."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(True, axis=grid, alpha=0.55, zorder=0)
        ax.set_axisbelow(True)
    return ax


def panel(ax, letter, title=""):
    ax.set_title(f"({letter})", loc="left", color=INK, pad=6)
    return ax
