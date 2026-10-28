"""Shared matplotlib style for the paper figures (static PDF/PNG)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 200,
})


def save(fig, path_noext):
    fig.tight_layout()
    fig.savefig(path_noext + ".pdf")
    fig.savefig(path_noext + ".png")
    plt.close(fig)
