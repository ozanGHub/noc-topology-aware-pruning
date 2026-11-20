from typing import Optional, Tuple
import os
import re

import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn

import matplotlib.cm as cm
import matplotlib.patches as mpatches


Node = Tuple[int, int]  # (node_layer, idx)
VIS_DIR = "visuals1"
_SAVE_COUNTER = 0  # simple in-memory counter

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _next_prefix(width: int = 3) -> str:
    global _SAVE_COUNTER
    _SAVE_COUNTER += 1
    return f"{_SAVE_COUNTER:0{width}d}"


def _title_to_filename(title: str, *, prefix: Optional[str] = None) -> str:
    slug = re.sub(r"[^\w\s-]", "", title).strip().lower()
    slug = re.sub(r"\s+", "_", slug)
    if prefix:
        return f"{VIS_DIR}/{prefix}_{slug}.png"
    return f"{VIS_DIR}/{slug}.png"


def _finish_plot(
    title: str,
    show: bool,
    *,
    numbered: bool = True,
    width: int = 3,
) -> None:
    _ensure_dir(VIS_DIR)

    prefix = _next_prefix(width) if numbered else None
    path = _title_to_filename(title, prefix=prefix)

    plt.savefig(path, bbox_inches="tight", dpi=150)
    print(f"[visualize] Saved → {path}")

    if show:
        plt.show()

    plt.close()


# ------------------------------------------------------------------
# Public functions
# ------------------------------------------------------------------

def visualize_model(
    model: nn.Module,
    *,
    show_labels: bool = False,
    eps: float = 1e-12,
    title: str = "Neural Network Visualization (Pruning Aware)",
    show: bool = False,
):
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("No nn.Linear layers found in model.")

    layers = [linears[0].in_features] + [l.out_features for l in linears]
    total_neurons = sum(layers)

    G_active = nx.DiGraph()
    G_pruned = nx.DiGraph()

    all_nodes = list(range(total_neurons))
    G_active.add_nodes_from(all_nodes)
    G_pruned.add_nodes_from(all_nodes)

    offsets = []
    s = 0
    for lsize in layers:
        offsets.append(s)
        s += lsize

    for layer_idx, layer in enumerate(linears):
        W = layer.weight.detach().cpu().numpy()
        in_offset = offsets[layer_idx]
        out_offset = offsets[layer_idx + 1]
        for out_i in range(W.shape[0]):
            for in_j in range(W.shape[1]):
                src = in_offset + in_j
                dst = out_offset + out_i
                if abs(W[out_i, in_j]) > eps:
                    G_active.add_edge(src, dst)
                else:
                    G_pruned.add_edge(src, dst)

    pos = {}
    current_index = 0
    horizontal_spacing = 4
    max_neurons = max(layers)

    for layer_idx, layer_size in enumerate(layers):
        vertical_spacing = max(10 / max_neurons, 1.5)
        center_offset = (max_neurons - layer_size) / 2
        for neuron_idx in range(layer_size):
            pos[current_index] = (
                layer_idx * horizontal_spacing,
                -(neuron_idx + center_offset) * vertical_spacing,
            )
            current_index += 1

    plt.figure(figsize=(12, 8))

    nx.draw_networkx_edges(
        G_active, pos, edge_color="gray", arrows=True,
        arrowsize=10, width=1.0, alpha=0.8,
    )
    nx.draw_networkx_edges(
        G_pruned, pos, edge_color="red", arrows=True,
        arrowsize=10, width=1.0, alpha=0.6,
    )

    node_colors = [
        "lightblue" if n < layers[0]
        else "brown" if n == total_neurons - 1
        else "lightgreen"
        for n in range(total_neurons)
    ]

    nx.draw_networkx_nodes(
        G_active, pos, nodelist=all_nodes,
        node_size=700, node_color=node_colors, edgecolors="white",
    )

    if show_labels:
        nx.draw_networkx_labels(G_active, pos, {i: str(i) for i in all_nodes}, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    _finish_plot(title, show)


def visualize_graph(
    G: nx.DiGraph,
    *,
    groups=None,
    show_labels: bool = True,
    title: str = "NN Graph (Debug View, Tuple Nodes)",
    show: bool = False,
):
    if "layers" not in G.graph:
        raise ValueError("Graph has no 'layers' metadata.")

    layers = G.graph["layers"]
    pos = {}
    horizontal_spacing = 4
    max_neurons = max(layers)

    for layer_idx, layer_size in enumerate(layers):
        vertical_spacing = max(10 / max_neurons, 1.5)
        center_offset = (max_neurons - layer_size) / 2
        for neuron_idx in range(layer_size):
            pos[(layer_idx, neuron_idx)] = (
                layer_idx * horizontal_spacing,
                -(neuron_idx + center_offset) * vertical_spacing,
            )

    plt.figure(figsize=(12, 8))

    nx.draw_networkx_edges(
        G, pos, edge_color="gray", arrows=True,
        arrowsize=10, width=1.0, alpha=0.6,
    )

    if groups is not None:
        node_to_gid = {node: gid for gid, group in enumerate(groups) for node in group}
        cmap = cm.get_cmap("viridis", max(len(groups), 1))
        node_colors = [
            cmap(node_to_gid[node]) if node in node_to_gid else (0.8, 0.8, 0.8, 1.0)
            for node in G.nodes
        ]
    else:
        last = len(layers) - 1
        node_colors = [
            "#0D47A1" if n[0] == 0
            else "#4E342E" if n[0] == last
            else "#1B5E20"
            for n in G.nodes
        ]

    nx.draw_networkx_nodes(
        G, pos, nodelist=list(G.nodes),
        node_size=1400, node_color=node_colors, edgecolors="white",
    )

    if show_labels:
        nx.draw_networkx_labels(
            G, pos,
            labels={n: f"{n[0]},{n[1]}" for n in G.nodes},
            font_size=12, font_color="white", font_weight="bold",
        )

    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    _finish_plot(title, show)


def visualize_wctg(
    W: nx.DiGraph,
    groups=None,
    *,
    show_labels: bool = True,
    title: str = "WCTG",
    with_edge_weights: bool = True,
    show: bool = False,
):
    if W.number_of_nodes() == 0:
        print("visualize_wctg: empty graph.")
        return

    # ── tuneable constants ────────────────────────────────────────────────────
    COLS_PER_ROW  = 5      # member tokens per row inside a panel
    MEMBER_FONT   = 6.5    # pt – member token font size
    NODE_SIZE     = 1_200  # scatter marker size (pts²)
    H_SCALE       = 2.8    # horizontal stretch between layers
    V_SCALE       = 2.0    # multipartite layout vertical scale
    PANEL_PAD_X   = 0.22   # data-unit gap between node edge and panel box
    # ─────────────────────────────────────────────────────────────────────────

    def _wrap_members(members, cols=COLS_PER_ROW):
        """Format neuron tuples into a compact multi-line grid string."""
        tokens = [f"({m[0]},{m[1]})" for m in sorted(members)]
        rows = [tokens[i: i + cols] for i in range(0, len(tokens), cols)]
        return "\n".join(" ".join(r) for r in rows)

    # ── layout ────────────────────────────────────────────────────────────────
    pos = nx.multipartite_layout(W, subset_key="layer", align="vertical", scale=V_SCALE)

    # Stretch x-coordinates so layers are further apart
    layer_ids = sorted({W.nodes[n].get("layer", 0) for n in W.nodes()})
    n_layers  = max(len(layer_ids), 1)
    x_vals    = sorted({p[0] for p in pos.values()})
    if len(x_vals) > 1:
        x_min, x_max = x_vals[0], x_vals[-1]
        for nd, (x, y) in pos.items():
            new_x = (x - x_min) / (x_max - x_min) * H_SCALE * (n_layers - 1)
            pos[nd] = (new_x, y)

    # ── figure auto-sizing ────────────────────────────────────────────────────
    max_per_layer = max(
        sum(1 for n in W.nodes() if W.nodes[n].get("layer", 0) == lid)
        for lid in layer_ids
    ) if layer_ids else 1

    fig_w = max(14, n_layers * 5)
    fig_h = max(10, max_per_layer * 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ── edges ─────────────────────────────────────────────────────────────────
    has_weights = with_edge_weights and any("weight" in W[u][v] for u, v in W.edges())
    if has_weights:
        weights = [W[u][v].get("weight", 1) for u, v in W.edges()]
        wmin, wmax = min(weights), max(weights)
        widths = [
            0.8 + 3.2 * (w - wmin) / (wmax - wmin) if wmax != wmin else 1.8
            for w in weights
        ]
    else:
        widths = [1.5] * W.number_of_edges()

    nx.draw_networkx_edges(
        W, pos, ax=ax, arrows=True, arrowsize=14, alpha=0.7,
        width=widths,
        node_size=NODE_SIZE, connectionstyle="arc3,rad=0.12",
    )
    nx.draw_networkx_nodes(
        W, pos, ax=ax, node_size=NODE_SIZE,
        node_color="steelblue", edgecolors="white", linewidths=1.5,
    )

    if show_labels:
        nx.draw_networkx_labels(W, pos, ax=ax, font_size=14,
                                font_color="white", font_weight="bold")

    # ── edge-weight labels ────────────────────────────────────────────────────
    if has_weights:
        edge_labels = {
            (u, v): round(W[u][v]["weight"], 2)
            for u, v in W.edges() if "weight" in W[u][v]
        }
        if edge_labels:
            nx.draw_networkx_edge_labels(W, pos, edge_labels=edge_labels, ax=ax,
                                         font_size=14, label_pos=0.5)

    # ── member-group panels (alternating left / right) ────────────────────────
    if groups is not None:
        all_x = [p[0] for p in pos.values()]
        x_data_span = max(max(all_x) - min(all_x), 1.0)

        # Approximate node radius in data units
        node_radius_pts  = (NODE_SIZE ** 0.5) / 2.0
        inch_per_data_x  = fig_w / (x_data_span + 1e-9)
        node_radius_data = (node_radius_pts / 72.0) / inch_per_data_x

        for node_id, (nx_pos, ny_pos) in pos.items():
            if node_id >= len(groups):
                continue
            members = groups[node_id]
            if not members:
                continue

            text = _wrap_members(members)

            # Even node_id → panel to the RIGHT; odd → LEFT
            if node_id % 2 == 0:
                ha    = "left"
                box_x = nx_pos + node_radius_data + PANEL_PAD_X
            else:
                ha    = "right"
                box_x = nx_pos - node_radius_data - PANEL_PAD_X

            ax.text(
                box_x, ny_pos, text,
                ha=ha, va="center",
                fontsize=MEMBER_FONT,
                fontfamily="monospace",
                color="#1a1a2e",
                linespacing=1.4,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    fc="#eef2ff",
                    ec="#6e8efb",
                    alpha=0.88,
                    linewidth=0.8,
                ),
                zorder=5,
            )

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    _finish_plot(title, show)


def visualize_mesh(
    G: nx.Graph,
    *,
    title: str = "Mesh Graph",
    highlight_nodes: list = None,
    highlight_edges: list = None,
    node_size: int = 1400,
    show_labels: bool = True,
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=(max(6, G.graph.get("cols", 8)),
                                    max(5, G.graph.get("rows", 6))))

    pos = {(r, c): (c, -r) for r, c in G.nodes()}
    is_directed = isinstance(G, nx.DiGraph)
    node_to_core_id = {node: i + 1 for i, node in enumerate(sorted(G.nodes()))}

    highlight_nodes = set(map(tuple, highlight_nodes)) if highlight_nodes else set()
    highlight_edges = set(map(tuple, map(sorted, highlight_edges))) if highlight_edges else set()

    normal_nodes  = [n for n in G.nodes() if n not in highlight_nodes]
    special_nodes = [n for n in G.nodes() if n in highlight_nodes]

    def is_highlighted_edge(u, v):
        return tuple(sorted([u, v])) in highlight_edges

    normal_edges  = [(u, v) for u, v in G.edges() if not is_highlighted_edge(u, v)]
    special_edges = [(u, v) for u, v in G.edges() if is_highlighted_edge(u, v)]

    nx.draw_networkx_edges(
        G, pos, ax=ax, edgelist=normal_edges, edge_color="#aac4dd", width=1.5,
        **( dict(arrows=True, arrowsize=12, node_size=node_size) if is_directed else {} ),
    )
    if special_edges:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=special_edges, edge_color="#e05c5c", width=2.5,
            **( dict(arrows=True, arrowsize=14, node_size=node_size) if is_directed else {} ),
        )

    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=normal_nodes, node_size=node_size,
        node_color="steelblue", edgecolors="white", linewidths=1.2,
    )
    if special_nodes:
        nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=special_nodes, node_size=node_size,
            node_color="#e05c5c", edgecolors="white", linewidths=1.2,
        )

    if show_labels:
        nx.draw_networkx_labels(
            G, pos, labels={n: f"Core {node_to_core_id[n]}" for n in G.nodes()},
            ax=ax, font_size=10, font_color="white",
        )

    if highlight_nodes or highlight_edges:
        ax.legend(handles=[
            mpatches.Patch(color="steelblue", label="Normal"),
            mpatches.Patch(color="#e05c5c",   label="Highlighted"),
        ], loc="upper right", framealpha=0.8)

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    _finish_plot(title, show)


def visualize_mapping(
    G: nx.Graph,
    mapping: list,
    *,
    title: str = "Task Mapping on Mesh",
    node_size: int = 1800,
    show: bool = False,
    groups: list = None,
    wctg: nx.DiGraph = None,
    spg: dict = None,
    total_cost: float = None,
    metrics: dict = None,
    structural_wctg: nx.DiGraph = None,
    show_wctg_overlay: bool = False,
):
    """
    Render the final SA task-to-core mapping on the mesh.

    Two layers:
      1. Light blue straight lines — physical mesh topology (always)
      2. Red curved arrows — drawn as explicit quadratic Bezier PathPatches
         (only when show_wctg_overlay=True).

    Optional parameters:
        groups             : neuron member sets per task (NoCMapper.groups).
                             Neuron-group tables below nodes are NOT rendered.
        wctg               : Weighted Communication Task Graph (NoCMapper.wctg).
                             Used for routing / label computation.
        structural_wctg    : Structural WCTG whose edge weights drive arrow
                             thickness, colour intensity, and alpha.  When
                             provided it replaces wctg for visual scaling;
                             the label still shows the cost contribution
                             derived from wctg (or structural weight if wctg
                             is absent).  Falls back to wctg when None.
        spg                : all-pairs shortest-path dict on mesh cores
                             (NoCMapper.spg, keyed by (core_idx_i, core_idx_j)).
                             When provided, arrow labels show the actual
                             communication cost contribution =
                             WCTG_weight × hop_distance,
                             instead of just the WCTG weight.
        show_wctg_overlay  : If True, draw the WCTG communication arrows
                             (Bezier curves / straight lines with cost labels)
                             on top of the mesh.  Defaults to False (mesh only).
    """
    import numpy as np
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    cols = G.graph.get("cols", 4)
    rows = G.graph.get("rows", 4)

    fig, ax = plt.subplots(figsize=(max(10, cols * 2.8),
                                    max(8,  rows * 2.8)))

    sorted_nodes    = sorted(G.nodes())
    core_id_to_node = {i: node for i, node in enumerate(sorted_nodes)}
    node_to_core_id = {node: i + 1 for i, node in enumerate(sorted_nodes)}
    # 0-indexed core index (used for SPG lookup)
    node_to_core_idx = {node: i for i, node in enumerate(sorted_nodes)}
    node_to_task    = {core_id_to_node[core_id]: task_id for task_id, core_id in mapping}
    task_to_node    = {task_id: core_id_to_node[core_id] for task_id, core_id in mapping}

    occupied_nodes = [n for n in G.nodes() if n in node_to_task]
    idle_nodes     = [n for n in G.nodes() if n not in node_to_task]

    pos = {(r, c): (c, -r) for r, c in G.nodes()}

    def bezier_point(p0, c, p1, t):
        return (1-t)**2 * p0 + 2*(1-t)*t * c + t**2 * p1

    def bezier_tangent(p0, c, p1, t):
        return 2*(1-t)*(c - p0) + 2*t*(p1 - c)

    # ------------------------------------------------------------------
    # Layer 1: Physical mesh — light blue straight lines
    # ------------------------------------------------------------------
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color="#c8d8e8", width=1.5, arrows=False,
    )

    # ------------------------------------------------------------------
    # Layer 2: Explicit quadratic Bezier curves for WCTG communication
    # Only drawn when show_wctg_overlay=True.
    # ------------------------------------------------------------------
    # The graph used to drive visual style (thickness, colour, alpha) is
    # structural_wctg when provided, otherwise fall back to wctg.
    _visual_graph = structural_wctg if structural_wctg is not None else wctg

    if show_wctg_overlay and (wctg is not None or structural_wctg is not None):
        # Collect all positive visual weights for normalisation
        all_w = [d.get("weight", 0) for _, _, d in _visual_graph.edges(data=True)
                 if d.get("weight", 0) > 0]
        wmax = max(all_w) if all_w else 1
        rad  = 0.3

        # Iterate over structural_wctg edges if available, else wctg edges.
        # We need labels from wctg (cost contribution), so build a quick lookup.
        _label_graph = wctg  # may be None if only structural_wctg passed

        for u, v, data in _visual_graph.edges(data=True):
            # Visual weight comes from the style graph
            w = data.get("weight", 0)
            if w == 0:
                continue
            cu = task_to_node.get(u)
            cv = task_to_node.get(v)
            if cu is None or cv is None or cu == cv:
                continue

            p0 = np.array(pos[cu])
            p1 = np.array(pos[cv])

            intensity = 0.25 + 0.75 * (w / wmax)
            color     = (1.0, 1.0 - intensity, 1.0 - intensity)
            lw        = 1.0 + 4.5 * (w / wmax)
            alpha     = 0.5  + 0.5  * (w / wmax)

            # Label weight: prefer wctg edge weight for cost labels,
            # fall back to structural weight when wctg is absent.
            if _label_graph is not None and _label_graph.has_edge(u, v):
                w_label = _label_graph[u][v].get("weight", w)
            else:
                w_label = w

            # Determine if cores are mesh neighbours (Manhattan distance == 1)
            manhattan = abs(cu[0] - cv[0]) + abs(cu[1] - cv[1])
            is_neighbour = (manhattan == 1)

            # Build label string
            if spg is not None:
                ci   = node_to_core_idx[cu]
                cj   = node_to_core_idx[cv]
                hops = spg.get((ci, cj), spg.get((cj, ci), None))
                if hops is not None:
                    label_val = int(w_label * hops)
                    label_str = f"{label_val}\n({int(w_label)}×{int(hops)})"
                else:
                    label_str = str(int(w_label))
            else:
                label_str = str(int(w_label))

            if is_neighbour:
                # ── Straight line overlapping mesh edge ──────────────────
                ax.annotate(
                    "", xy=(p1[0], p1[1]), xytext=(p0[0], p0[1]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color,
                        lw=lw,
                        alpha=alpha,
                        connectionstyle="arc3,rad=0.0",   # straight
                    ),
                    zorder=4,
                )
                # Label at t=0.25 along the straight line (same logic as
                # Bezier branch) — exactly on the arrow, no perp offset
                lpos = p0 + 0.25 * (p1 - p0)
                ax.text(lpos[0], lpos[1], label_str,
                        ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color="darkred", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.12",
                                  fc="white", ec="lightcoral", alpha=0.95))
            else:
                # ── Curved Bezier for non-adjacent cores ─────────────────
                diff   = p1 - p0
                length = np.linalg.norm(diff) + 1e-9
                perp   = np.array([-diff[1], diff[0]]) / length
                C      = (p0 + p1) / 2 + rad * perp * length

                verts = [p0, C, p1]
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                patch = PathPatch(Path(verts, codes),
                                  facecolor="none", edgecolor=color,
                                  linewidth=lw, alpha=alpha, zorder=4)
                ax.add_patch(patch)

                # Arrowhead at t=0.98
                t_arr  = 0.98
                tip    = bezier_point(p0, C, p1, t_arr)
                tang   = bezier_tangent(p0, C, p1, t_arr)
                tang  /= (np.linalg.norm(tang) + 1e-9)
                perp_h = np.array([-tang[1], tang[0]])
                arrow_tri = plt.Polygon(
                    [tip + tang * 0.08,
                     tip - perp_h * 0.05,
                     tip + perp_h * 0.05],
                    closed=True, facecolor=color, edgecolor=color,
                    alpha=alpha, zorder=5)
                ax.add_patch(arrow_tri)

                # Label at t=0.20 on the Bezier curve
                lpos = bezier_point(p0, C, p1, 0.20)
                ax.text(lpos[0], lpos[1], label_str,
                        ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color="darkred", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.12",
                                  fc="white", ec="lightcoral", alpha=0.95))

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=idle_nodes, node_size=node_size,
        node_color="#d0dce8", edgecolors="white", linewidths=1.2,
    )
    if occupied_nodes:
        cmap = plt.cm.get_cmap("tab20", max(len(occupied_nodes), 1))
        nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=occupied_nodes, node_size=node_size,
            node_color=[cmap(node_to_task[n]) for n in occupied_nodes],
            edgecolors="white", linewidths=1.8,
        )

    for node in occupied_nodes:
        x, y    = pos[node]
        task_id = node_to_task[node]
        ax.text(x, y, f"T{task_id + 1}",
                ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=6)

    # ------------------------------------------------------------------
    # Core ID + neuron members below each node
    # Compute node radius in data coordinates via display transform so
    # labels are always anchored directly below the circle regardless
    # of figure size or DPI.
    # ------------------------------------------------------------------
    fig.canvas.draw()

    # Convert node_size (pts^2) to a data-unit radius:
    # node radius in points = sqrt(node_size) / 2
    # 1 point = 1/72 inch; use the axes bbox to get inches-per-data-unit
    ax_width_inch  = ax.get_window_extent().width  / fig.dpi
    ax_height_inch = ax.get_window_extent().height / fig.dpi
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_data_range = xlim[1] - xlim[0]
    y_data_range = ylim[1] - ylim[0]
    inch_per_xunit = ax_width_inch  / x_data_range if x_data_range else 1
    inch_per_yunit = ax_height_inch / abs(y_data_range) if y_data_range else 1
    # Use the smaller of x/y so the radius fits in both directions
    inch_per_unit  = min(inch_per_xunit, inch_per_yunit)
    radius_pts     = (node_size ** 0.5) / 2          # node radius in points
    radius_data    = (radius_pts / 72) / inch_per_unit  # convert to data units

    for node in G.nodes():
        x, y    = pos[node]
        label_y = y - radius_data - 0.06
        ax.text(x, label_y, f"Core {node_to_core_id[node]}",
                ha="center", va="top", fontsize=8,
                color="black", zorder=6)

    # ------------------------------------------------------------------
    # Title — left-aligned with small indent, no total cost line
    # ------------------------------------------------------------------
    ax.set_title(f"  {title}",
                 fontsize=13, fontweight="bold",
                 loc="left", pad=10)

    # ------------------------------------------------------------------
    # Legend — top-right
    # ------------------------------------------------------------------
    legend_handles = [mpatches.Patch(color="#c8d8e8", label="Mesh link")]
    if idle_nodes:
        legend_handles.append(mpatches.Patch(color="#d0dce8", label="Idle core"))
    if show_wctg_overlay and (wctg is not None or structural_wctg is not None):
        label_desc = ("Cost contribution = = Σ_{(i,j) in str_wctg} weight(i,j) × hops(i,j)"
                      if spg is not None else "Structural WCTG traffic weight" if structural_wctg is not None else "WCTG traffic weight (curved)")
        legend_handles.append(mpatches.Patch(color="#e05c5c", label=label_desc))
    legend = ax.legend(handles=legend_handles, loc="upper right",
                       framealpha=0.9, fontsize=9)

    # ------------------------------------------------------------------
    # Metrics panel — top-left, below the title.
    # Includes Total Comm. Cost as first row when provided.
    # ------------------------------------------------------------------
    if metrics is not None or total_cost is not None:
        def _fmt(val, base, is_pct=False):
            if val is None or base is None:
                return "N/A"
            cur_s = f"{val*100:.2f}%" if is_pct else f"{val:,.1f}"
            if base == 0:
                return cur_s
            delta = (val - base) / abs(base) * 100
            sign  = "+" if delta >= 0 else ""
            return f"{cur_s}  ({sign}{delta:.1f}%)"

        baseline = (metrics or {}).get("baseline", {})
        current  = (metrics or {}).get("current",  {})

        text_lines = ["■ Summary", ""]

        # Struct. Cost — read directly from the metrics dict (sp_cost key),
        # which make_metrics populates from stages["sp_cost"] = compute_structural_cost(...)
        cur_sp_cost  = current.get("sp_cost")
        base_sp_cost = baseline.get("sp_cost")
        if cur_sp_cost is not None:
            if base_sp_cost and base_sp_cost != 0:
                delta = (cur_sp_cost - base_sp_cost) / abs(base_sp_cost) * 100
                sign  = "+" if delta >= 0 else ""
                scost_str = f"{cur_sp_cost:,.1f}  ({sign}{delta:.1f}%)"
            else:
                scost_str = f"{cur_sp_cost:,.1f}"
            text_lines.append(f"{'Str. Comm. Cost':<12} {scost_str}")

        # Total Comm. Cost row (from noc.cost, not metrics dict)
        if total_cost is not None:
            base_cost = baseline.get("cost")
            if base_cost and base_cost != 0:
                delta = (total_cost - base_cost) / abs(base_cost) * 100
                sign  = "+" if delta >= 0 else ""
                cost_str = f"{total_cost:,.1f}  ({sign}{delta:.1f}%)"
            else:
                cost_str = f"{total_cost:,.1f}"
            text_lines.append(f"{'Comm. Cost':<12} {cost_str}")

        if metrics is not None:
            def _fmt_ms(val, base):
                """3 decimal places, ms before delta: 0.15 ms  (-46.4%)"""
                if val is None or base is None:
                    return "N/A"
                cur_s = f"{val:.3f} ms"
                if base == 0:
                    return cur_s
                delta = (val - base) / abs(base) * 100
                sign  = "+" if delta >= 0 else ""
                return f"{cur_s}  ({sign}{delta:.1f}%)"

            rows = [
                ("Accuracy",   _fmt(current.get("accuracy"),  baseline.get("accuracy"),  is_pct=True)),
                ("Edges",      _fmt(current.get("edges"),     baseline.get("edges"))),
                ("Neurons",    _fmt(current.get("neurons"),   baseline.get("neurons"))),
                ("Batch Inf.", _fmt_ms(current.get("batch_ms"), baseline.get("batch_ms"))),
            ]
            for label, val_str in rows:
                text_lines.append(f"{label:<12} {val_str}")

        panel_text = chr(10).join(text_lines)

        ax.text(0.01, 0.99, panel_text,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=8, family="monospace",
                color="#1a1a1a", zorder=10,
                bbox=dict(boxstyle="round,pad=0.5",
                          fc="#f0f4ff", ec="#7090c0",
                          alpha=0.92, linewidth=1.2))

    ax.axis("off")
    plt.tight_layout()
    _finish_plot(title, show)
