"""
generate_plots.py
=================
Generates all paper-quality plots for the NoC-aware pruning paper.

Plots produced
--------------
  plot1b_wctg_cost_vs_drop      — WCTG comm. cost vs accuracy drop
  plot2b_wctg_cost_vs_neurons   — Comm. cost vs neuron count
  plot3_flops_vs_drop           — FLOPs vs accuracy drop
  plot4_cost_vs_edges           — Comm. cost vs edge count
  plot_budget_sweep_comm_cost   — Comm. cost vs accuracy budget (sweep)
  plot_budget_sweep_two_panel   — Two-panel: comm. cost + neurons vs budget

Each plot is saved as:
  <name>.pdf   — vector, via matplotlib savefig
  <name>.png   — high-res raster (PNG_SCALE × native), via PyMuPDF

Dependencies
------------
  pip install matplotlib pymupdf

Usage
-----
  python generate_plots.py

The trajectory CSV is expected at:
  ./trajectory_LeNet300_100.csv

Outputs are written to OUT_DIR (default: current directory).
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ── Configuration ──────────────────────────────────────────────────────────────
TRAJECTORY_CSV = "trajectory_LeNet300_100.csv"
OUT_DIR        = "."
PNG_SCALE      = 3      # PyMuPDF rasterisation scale (3× = ~220 dpi equivalent)

# ── Global matplotlib style ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth":  0.8,
    "grid.linewidth":  0.4,
    "grid.alpha":      0.4,
    "figure.dpi":      180,
})

# ── Branch styling ─────────────────────────────────────────────────────────────
BRANCHES = {
    "A: IterCC Neuron + Bridge-Link": dict(
        label="A: Heuristic CommCost + Bridge-Link",
        color="#4C3BCE", lw=2.2, ls="-", zorder=6),
    "B: IterCC Neuron + L2 edge": dict(
        label="B: Heuristic CommCost + L2",
        color="#1A9E74", lw=2.2, ls="-", zorder=5),
    "C: IterCC Neuron + L1 edge": dict(
        label="C: Heuristic CommCost + L1",
        color="#52B788", lw=1.6, ls="--", zorder=4),
    "D: Taylor Neuron + L2 edge": dict(
        label="D: Taylor + L2",
        color="#D55E00", lw=2.2, ls="-", zorder=3),
    "E: Taylor Neuron + L1 edge": dict(
        label="E: Taylor + L1",
        color="#E08020", lw=1.6, ls="--", zorder=2),
    "G: Taylor Neuron + Bridge-Link": dict(
        label="G: Taylor + Bridge-Link",
        color="#999999", lw=1.6, ls=":", zorder=1),
}

# ── WCTG anchor points (from summary tables) ───────────────────────────────────
ANCHORS_1B = {
    "A: IterCC Neuron + Bridge-Link": [
        (0.0, 66976, "Baseline"), (0.0056, 36633, "0.5%"), (0.0096, 1530, "Final"),
    ],
    "B: IterCC Neuron + L2 edge": [
        (0.0, 66976, "Baseline"), (0.0066, 47024, "0.5%"), (0.0096, 1258, "Final"),
    ],
    "C: IterCC Neuron + L1 edge": [
        (0.0, 66976, "Baseline"), (0.0066, 46961, "0.5%"), (0.0096, 1268, "Final"),
    ],
    "D: Taylor Neuron + L2 edge": [
        (0.0, 66976, "Baseline"), (0.0061, 40977, "0.5%"), (0.0101, 1313, "Final"),
    ],
    "E: Taylor Neuron + L1 edge": [
        (0.0, 66976, "Baseline"), (0.0061, 40977, "0.5%"), (0.0101, 1298, "Final"),
    ],
    "G: Taylor Neuron + Bridge-Link": [
        (0.0, 66976, "Baseline"), (0.0057, 30826, "0.5%"), (0.0010, 15238, "Final"),
    ],
}

SNAP_ANCHORS = {
    "A: IterCC Neuron + Bridge-Link": (1162, 36633),
    "B: IterCC Neuron + L2 edge":     (1173, 47024),
    "C: IterCC Neuron + L1 edge":     (1173, 46961),
    "D: Taylor Neuron + L2 edge":     (1137, 40977),
    "E: Taylor Neuron + L1 edge":     (1137, 40977),
    "G: Taylor Neuron + Bridge-Link": (1118, 30826),
}

FINAL_ANCHORS_2B = {
    "A: IterCC Neuron + Bridge-Link": (921,  1530),
    "B: IterCC Neuron + L2 edge":     (1173, 1258),
    "C: IterCC Neuron + L1 edge":     (1173, 1268),
    "D: Taylor Neuron + L2 edge":     (1137, 1313),
    "E: Taylor Neuron + L1 edge":     (1137, 1298),
    "G: Taylor Neuron + Bridge-Link": (1017, 15238),
}

# Final table accuracy drop per branch (used to clip plot3 trajectories)
FINAL_DROP = {
    "A: IterCC Neuron + Bridge-Link": 0.0096,
    "B: IterCC Neuron + L2 edge":     0.0096,
    "C: IterCC Neuron + L1 edge":     0.0096,
    "D: Taylor Neuron + L2 edge":     0.0101,
    "E: Taylor Neuron + L1 edge":     0.0101,
    "G: Taylor Neuron + Bridge-Link": 0.0010,
}

# Budget sweep data (LeNet 500-300 final table)
BUDGETS    = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
COMM_A     = [141968, 2343, 1979, 1539, 1109,  627]
COMM_B     = [141886, 1867, 1264, 1061,  957,  689]
COMM_D     = [ 90173, 2004, 1272, 1077,  781,  578]
NEURONS_A  = [1511,  921,  915,  913,  906,  903]
NEURONS_B  = [1511, 1508, 1502, 1502, 1502, 1502]
NEURONS_D  = [1330, 1330, 1330, 1330, 1330, 1330]

COL_A = "#4C3BCE"
COL_B = "#1A9E74"
COL_D = "#D55E00"

TOL_LINE    = 0.01
SNAP_LINE   = 0.005
PROXY_SCALE = 66976.0 / 5088.0   # scale proxy to WCTG units


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_trajectory(path: str) -> dict:
    """Load trajectory CSV into a dict keyed by branch name."""
    data: dict = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            data[row["branch"]].append({
                "step":          int(row["step"]),
                "accuracy_drop": float(row["accuracy_drop"]),
                "neurons":       int(row["neurons"]),
                "edges":         int(row["edges"]),
                "cross_group_w_sum": float(row["cross_group_w_sum"]),
                "flops":         int(row["flops"]),
            })
    return data


def export(fig: plt.Figure, name: str) -> None:
    """Save figure as PDF (matplotlib) and PNG (PyMuPDF rasterisation)."""
    pdf_path = os.path.join(OUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUT_DIR, f"{name}.png")

    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    print(f"PDF → {pdf_path}")

    try:
        import fitz
        doc  = fitz.open(pdf_path)
        page = doc[0]
        mat  = fitz.Matrix(PNG_SCALE, PNG_SCALE)
        pix  = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(png_path)
        print(f"PNG → {png_path}  ({PNG_SCALE}× scale, {pix.width}×{pix.height} px)")
        doc.close()
    except ImportError:
        print("PyMuPDF not found — install with:  pip install pymupdf")
        print("Falling back to matplotlib PNG export.")
        fig.savefig(png_path, bbox_inches="tight", dpi=220)
        print(f"PNG → {png_path}  (matplotlib fallback)")

    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1b — WCTG comm. cost vs accuracy drop
# ══════════════════════════════════════════════════════════════════════════════

def plot1b(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    # Faint proxy trajectories (cross-group Σ|w|)
    for bname, cfg in BRANCHES.items():
        rows = [r for r in data.get(bname, []) if r["accuracy_drop"] <= TOL_LINE + 0.001]
        if rows:
            ax.plot([r["accuracy_drop"] for r in rows],
                    [r["cross_group_w_sum"] * PROXY_SCALE for r in rows],
                    color=cfg["color"], lw=cfg["lw"], ls=cfg["ls"],
                    alpha=0.25, zorder=cfg["zorder"])

    # Solid WCTG n2n anchor lines + markers
    for bname, cfg in BRANCHES.items():
        anchors  = ANCHORS_1B.get(bname, [])
        non_base = [(d, c, lab) for d, c, lab in anchors if lab != "Baseline"]
        all_xs   = [0.0] + [d for d, c, lab in non_base]
        all_ys   = [66976] + [c for d, c, lab in non_base]
        ax.plot(all_xs, all_ys, color=cfg["color"], lw=cfg["lw"] + 0.4,
                ls=cfg["ls"], alpha=0.95, zorder=cfg["zorder"] + 5,
                label=cfg["label"])
        for drop, cost, lab in non_base:
            ax.scatter([drop], [cost], color=cfg["color"], s=52,
                       marker="D" if lab == "Final" else "s",
                       zorder=cfg["zorder"] + 15, edgecolors="white", linewidths=0.7)

    ax.scatter([0.0], [66976], color="black", s=55, marker="o",
               zorder=20, edgecolors="white", linewidths=0.7)

    ax.axvline(TOL_LINE,  color="#CC0000", lw=1.0, ls="--", alpha=0.7)
    ax.axvline(SNAP_LINE, color="#CC6600", lw=0.8, ls=":",  alpha=0.5)

    ax.set_xlabel("Accuracy drop from baseline", labelpad=4)
    ax.set_ylabel("Comm. cost  (WCTG n2n  /  cross-group Σ|w| proxy)", labelpad=4)
    ax.set_title(
        "WCTG comm. cost vs Accuracy drop  —  LeNet 784-300-100-10 | MNIST\n"
        "solid lines = WCTG n2n (SA-mapped);  faint lines = cross-group Σ|w| (proxy)",
        pad=6)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlim(-0.002, 0.020)
    ax.set_ylim(0, 72000)
    ax.grid(True, which="major")

    ax.annotate("1% tolerance",  xy=(TOL_LINE, 68500),
                xytext=(TOL_LINE + 0.0004, 68500), fontsize=7.5, color="#CC0000")
    ax.annotate("0.5% snapshot", xy=(SNAP_LINE, 63000),
                xytext=(SNAP_LINE + 0.0004, 63000), fontsize=7.5, color="#CC6600")
    ax.text(0.985, 0.97, "Lower-left is better", transform=ax.transAxes,
            fontsize=7.5, color="#555555", ha="right", va="top", style="italic")

    xlim_min, xlim_max = -0.002, 0.020
    x_anchor = (0.009 - xlim_min) / (xlim_max - xlim_min)

    leg1 = ax.legend(loc="upper left", bbox_to_anchor=(x_anchor, 0.88),
                     framealpha=0.93, edgecolor="#cccccc",
                     borderpad=0.6, labelspacing=0.4)
    ax.add_artist(leg1)

    marker_legend = [
        Line2D([0],[0], marker="s", color="gray", lw=0, ms=6,
               markeredgecolor="white", label="WCTG @ 0.5% drop"),
        Line2D([0],[0], marker="D", color="gray", lw=0, ms=6,
               markeredgecolor="white", label="WCTG final (stopping point)"),
    ]
    ax.legend(handles=marker_legend, loc="upper left",
              bbox_to_anchor=(x_anchor, 0.30),
              framealpha=0.88, edgecolor="#cccccc", fontsize=8, borderpad=0.6)

    plt.tight_layout(pad=0.8)
    export(fig, "plot1b_wctg_cost_vs_drop")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2b — Comm. cost vs neuron count
# ══════════════════════════════════════════════════════════════════════════════

def plot2b(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.4))

    for bname, cfg in BRANCHES.items():
        rows = [r for r in data.get(bname, []) if r["accuracy_drop"] <= TOL_LINE + 0.001]
        if not rows:
            continue
        rows.sort(key=lambda r: -r["neurons"])
        xs = [r["neurons"] for r in rows]
        ys = [r["cross_group_w_sum"] for r in rows]
        ax.plot(xs, ys, color=cfg["color"], lw=cfg["lw"], ls=cfg["ls"],
                alpha=0.85, zorder=cfg["zorder"], label=cfg["label"])
        ax.scatter([xs[-1]], [ys[-1]], color=cfg["color"], s=40,
                   zorder=cfg["zorder"] + 10, edgecolors="white", linewidths=0.6)

    ax.set_xlabel("Hidden neurons remaining  (fewer →)", labelpad=4)
    ax.set_ylabel("Comm. cost proxy  (Σ|w| cross-group edges)", labelpad=4)
    ax.set_title("Comm. cost vs Neuron count — neuron selection quality\n"
                 "LeNet 784-300-100-10 | MNIST", pad=6)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlim(410, 100)
    ax.set_ylim(0, 5500)
    ax.grid(True, which="major")
    ax.legend(loc="upper right", framealpha=0.92, edgecolor="#cccccc",
              bbox_to_anchor=(1.0, 1.0))
    ax.text(0.97, 0.03, "Lower-right is better\n(fewer neurons, lower cost)",
            transform=ax.transAxes, fontsize=7.5, color="#555555",
            ha="right", va="bottom", style="italic")

    plt.tight_layout(pad=0.8)
    export(fig, "plot2b_wctg_cost_vs_neurons")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — FLOPs vs accuracy drop (worst accepted, rolling max)
# ══════════════════════════════════════════════════════════════════════════════

def plot3(data: dict) -> None:
    Y_MAX = 560000

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    end_x_A, end_y_A = None, None

    for bname, cfg in BRANCHES.items():
        max_drop = FINAL_DROP.get(bname, TOL_LINE)
        rows = [r for r in data.get(bname, []) if r["accuracy_drop"] <= max_drop + 0.0001]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["step"])

        # Rolling maximum drop — monotonically non-decreasing x
        running_max = 0.0
        xs, ys = [], []
        for r in rows:
            running_max = max(running_max, r["accuracy_drop"])
            xs.append(running_max)
            ys.append(r["flops"])

        # Deduplicate consecutive identical points
        cx, cy = [xs[0]], [ys[0]]
        for i in range(1, len(xs)):
            if xs[i] != cx[-1] or ys[i] != cy[-1]:
                cx.append(xs[i])
                cy.append(ys[i])

        ax.plot(cx, cy, color=cfg["color"], lw=cfg["lw"], ls=cfg["ls"],
                alpha=0.85, zorder=cfg["zorder"], label=cfg["label"])
        ax.scatter([cx[-1]], [cy[-1]], color=cfg["color"], s=40,
                   zorder=cfg["zorder"] + 10, edgecolors="white", linewidths=0.6)

        if bname == "A: IterCC Neuron + Bridge-Link":
            end_x_A, end_y_A = cx[-1], cy[-1]

    ax.axvline(TOL_LINE,  color="#CC0000", lw=1.0, ls="--", alpha=0.7)
    ax.axvline(SNAP_LINE, color="#CC6600", lw=0.8, ls=":",  alpha=0.5)

    ax.set_xlabel("Accuracy drop from baseline  (worst accepted)", labelpad=4)
    ax.set_ylabel("FLOPs  (2 × Σ MAC per forward pass)", labelpad=4)
    ax.set_title("Compute efficiency: FLOPs vs Accuracy drop\n"
                 "LeNet 784-300-100-10 | MNIST", pad=6)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:,.0f}K"))
    ax.set_xlim(-0.001, 0.012)
    ax.set_ylim(0, Y_MAX)
    ax.grid(True, which="major")

    y_frac = 240000 / Y_MAX
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, y_frac),
              framealpha=0.92, edgecolor="#cccccc")
    ax.text(0.97, 0.03, "Lower-left is better", transform=ax.transAxes,
            fontsize=7.5, color="#555555", ha="right", va="bottom", style="italic")

    if end_x_A is not None:
        ax.text(end_x_A, end_y_A, "\n(compute-optimal)",
                fontsize=7.5,
                color=BRANCHES["A: IterCC Neuron + Bridge-Link"]["color"],
                ha="center", va="top", transform=ax.transData)

    plt.tight_layout(pad=0.8)
    export(fig, "plot3_flops_vs_drop")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Comm. cost vs edge count
# ══════════════════════════════════════════════════════════════════════════════

def plot4(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    for bname, cfg in BRANCHES.items():
        rows = [r for r in data.get(bname, []) if r["accuracy_drop"] <= TOL_LINE + 0.001]
        if not rows:
            continue
        rows.sort(key=lambda r: -r["edges"])
        xs = [r["edges"] for r in rows]
        ys = [r["cross_group_w_sum"] for r in rows]
        ax.plot(xs, ys, color=cfg["color"], lw=cfg["lw"], ls=cfg["ls"],
                alpha=0.85, zorder=cfg["zorder"], label=cfg["label"])
        ax.scatter([xs[-1]], [ys[-1]], color=cfg["color"], s=40,
                   zorder=cfg["zorder"] + 10, edgecolors="white", linewidths=0.6)

    ax.set_xlabel("Edges remaining  (fewer →)", labelpad=4)
    ax.set_ylabel("Comm. cost proxy  (Σ|w| cross-group edges)", labelpad=4)
    ax.set_title("Comm. cost vs Edge count — edge method quality\n"
                 "LeNet 784-300-100-10 | MNIST", pad=6)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlim(210000, 0)
    ax.set_ylim(0, 5500)
    ax.grid(True, which="major")
    ax.legend(loc="upper right", framealpha=0.92, edgecolor="#cccccc")
    ax.text(0.97, 0.03, "Lower-right is better\n(fewer edges, lower cost)",
            transform=ax.transAxes, fontsize=7.5, color="#555555",
            ha="right", va="bottom", style="italic")

    plt.tight_layout(pad=0.8)
    export(fig, "plot4_cost_vs_edges")


# ══════════════════════════════════════════════════════════════════════════════
# Plot budget sweep — Comm. cost vs accuracy budget
# ══════════════════════════════════════════════════════════════════════════════

def plot_budget_sweep_comm_cost() -> None:
    # Crossover interpolation (B vs D)
    x1, y1_B, y1_D = 2.0, math.log(COMM_B[3]), math.log(COMM_D[3])
    x2, y2_B, y2_D = 3.0, math.log(COMM_B[4]), math.log(COMM_D[4])
    t       = (y1_D - y1_B) / ((y2_B - y1_B) - (y2_D - y1_D))
    x_cross = x1 + t * (x2 - x1)
    y_cross = math.exp(y1_B + t * (y2_B - y1_B))

    label_A = ("A: Heuristic CommCost + Bridge-Link\n"
               "neurons: " + ", ".join(str(n) for n in NEURONS_A))
    label_B = ("B: Heuristic CommCost + L2\n"
               "neurons: " + ", ".join(str(n) for n in NEURONS_B))
    label_D = ("D: Taylor + L2\n"
               "neurons: " + ", ".join(str(n) for n in NEURONS_D) +
               "\nΔ(D−B) ≈ −172 neurons")

    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    ax.semilogy(BUDGETS, COMM_A, color=COL_A, lw=2.2, ls="-",
                marker="o", ms=6, label=label_A)
    ax.semilogy(BUDGETS, COMM_B, color=COL_B, lw=2.2, ls="-",
                marker="s", ms=6, label=label_B)
    ax.semilogy(BUDGETS, COMM_D, color=COL_D, lw=2.2, ls="-",
                marker="D", ms=6, label=label_D)

    ax.axvline(x_cross, color="#888888", lw=0.9, ls="--", alpha=0.7)
    ax.annotate(f"Crossover\n≈{x_cross:.1f}% budget",
                xy=(x_cross, y_cross),
                xytext=(x_cross + 0.35, y_cross * 2.8),
                fontsize=8, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))
    ax.axvspan(0,       x_cross, alpha=0.06, color=COL_B)
    ax.axvspan(x_cross, 5.5,     alpha=0.06, color=COL_D)
    ax.text(1.0, 650, "B wins\n(Heuristic CommCost)",
            fontsize=7.5, color=COL_B, ha="center", style="italic")
    ax.text(4.2, 650, "D wins\n(Taylor)",
            fontsize=7.5, color=COL_D, ha="center", style="italic")

    ax.set_xlabel("Accuracy budget (% drop from baseline)", labelpad=4)
    ax.set_ylabel("WCTG n2n comm. cost  (log scale)", labelpad=4)
    ax.set_title("Comm. cost vs Accuracy budget — budget sweep\n"
                 "LeNet 784-500-300-10 | MNIST", pad=6)
    ax.set_xlim(0.2, 5.5)
    ax.set_xticks(BUDGETS)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}" if x >= 1000 else f"{x:.0f}"))
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.15)

    ax.text(0.97, 0.97, "Lower is better", transform=ax.transAxes,
            fontsize=7.5, color="#555555", ha="right", va="top", style="italic")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=1, framealpha=0.92, edgecolor="#cccccc", fontsize=8.2)

    plt.tight_layout(pad=0.8)
    plt.subplots_adjust(bottom=0.32)
    export(fig, "plot_budget_sweep_comm_cost")


# ══════════════════════════════════════════════════════════════════════════════
# Plot budget sweep — Two-panel: comm. cost + neurons vs budget
# ══════════════════════════════════════════════════════════════════════════════

def plot_budget_sweep_two_panel() -> None:
    x1, y1_B, y1_D = 2.0, math.log(COMM_B[3]), math.log(COMM_D[3])
    x2, y2_B, y2_D = 3.0, math.log(COMM_B[4]), math.log(COMM_D[4])
    t       = (y1_D - y1_B) / ((y2_B - y1_B) - (y2_D - y1_D))
    x_cross = x1 + t * (x2 - x1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.8, 6.0),
                                    sharex=True,
                                    gridspec_kw={"hspace": 0.10})

    # Panel 1 — comm. cost (log scale)
    ax1.semilogy(BUDGETS, COMM_A, color=COL_A, lw=2.0, ls="-",
                 marker="o", ms=5.5, label="A: Heuristic CommCost + Bridge-Link")
    ax1.semilogy(BUDGETS, COMM_B, color=COL_B, lw=2.0, ls="-",
                 marker="s", ms=5.5, label="B: Heuristic CommCost + L2")
    ax1.semilogy(BUDGETS, COMM_D, color=COL_D, lw=2.0, ls="-",
                 marker="D", ms=5.5, label="D: Taylor + L2")
    ax1.axvline(x_cross, color="#888888", lw=0.9, ls="--", alpha=0.7)
    ax1.axvspan(0,       x_cross, alpha=0.06, color=COL_B)
    ax1.axvspan(x_cross, 5.5,     alpha=0.06, color=COL_D)
    ax1.set_ylabel("Comm. cost  (log scale)", labelpad=4)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}" if x >= 1000 else f"{x:.0f}"))
    ax1.grid(True, which="major")
    ax1.grid(True, which="minor", alpha=0.15)
    ax1.legend(loc="upper right", framealpha=0.92, edgecolor="#cccccc", fontsize=8.5)
    ax1.annotate(f"Crossover ≈{x_cross:.1f}%",
                 xy=(x_cross, 3000), xytext=(x_cross + 0.4, 8000),
                 fontsize=8, color="#555555",
                 arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))

    # Panel 2 — neurons remaining
    ax2.plot(BUDGETS, NEURONS_A, color=COL_A, lw=2.0, ls="-",
             marker="o", ms=5.5, label="A: Heuristic CommCost + Bridge-Link")
    ax2.plot(BUDGETS, NEURONS_B, color=COL_B, lw=2.0, ls="-",
             marker="s", ms=5.5, label="B: Heuristic CommCost + L2")
    ax2.plot(BUDGETS, NEURONS_D, color=COL_D, lw=2.0, ls="-",
             marker="D", ms=5.5, label="D: Taylor + L2")
    ax2.axvline(x_cross, color="#888888", lw=0.9, ls="--", alpha=0.7)
    ax2.axvspan(0,       x_cross, alpha=0.06, color=COL_B)
    ax2.axvspan(x_cross, 5.5,     alpha=0.06, color=COL_D)
    ax2.annotate("", xy=(5.0, NEURONS_D[5]), xytext=(5.0, NEURONS_B[5]),
                 arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.0))
    ax2.text(5.05, (NEURONS_D[5] + NEURONS_B[5]) / 2,
             f"Δ={NEURONS_B[5]-NEURONS_D[5]}",
             fontsize=8, color="#333333", va="center")
    ax2.set_xlabel("Accuracy budget (% drop from baseline)", labelpad=4)
    ax2.set_ylabel("Hidden neurons remaining", labelpad=4)
    ax2.set_xlim(0.2, 5.5)
    ax2.set_xticks(BUDGETS)
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax2.grid(True, which="major")
    ax2.set_ylim(800, 1650)

    fig.suptitle("Budget sweep: Comm. cost and Neuron count vs Accuracy budget\n"
                 "LeNet 784-500-300-10 | MNIST", fontsize=11, y=1.01)

    plt.tight_layout(pad=0.8)
    export(fig, "plot_budget_sweep_two_panel")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Loading trajectory from: {TRAJECTORY_CSV}")
    data = load_trajectory(TRAJECTORY_CSV)

    print("\n--- Plot 1b: WCTG comm. cost vs accuracy drop ---")
    plot1b(data)

    print("\n--- Plot 2b: Comm. cost vs neuron count ---")
    plot2b(data)

    print("\n--- Plot 3: FLOPs vs accuracy drop ---")
    plot3(data)

    print("\n--- Plot 4: Comm. cost vs edge count ---")
    plot4(data)

    print("\n--- Plot budget sweep: comm. cost ---")
    plot_budget_sweep_comm_cost()

    print("\n--- Plot budget sweep: two-panel ---")
    plot_budget_sweep_two_panel()

    print("\nAll plots generated.")
