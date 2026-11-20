from typing import List, Tuple

import networkx as nx


def bin_pack_best_fit(
    nodes: List[Tuple[int, int]],
    weights: List[int],
    capacity: float,
) -> List[List[Tuple[int, int]]]:
    if len(nodes) != len(weights):
        raise ValueError("nodes and weights must have the same length")

    groups: List[List[Tuple[int, int]]] = []
    remaining: List[float] = []

    for node, w in zip(nodes, weights):
        best_idx = -1
        best_rem = None

        for i, rem in enumerate(remaining):
            if w <= rem:
                new_rem = rem - w
                if best_rem is None or new_rem < best_rem:
                    best_rem = new_rem
                    best_idx = i

        if best_idx == -1:
            groups.append([node])
            remaining.append(capacity - w if w <= capacity else 0.0)
        else:
            groups[best_idx].append(node)
            remaining[best_idx] -= w

    return groups


def indegree_grouping(
    G: nx.DiGraph,
    N: int,
    capacity: float,
    *,
    decreasing: bool = True,   # Best-Fit Decreasing if True
) -> List[List[Tuple[int, int]]]:
    if "layers" not in G.graph:
        raise ValueError("Graph must contain layer metadata (use construct_nn_graph_tuple).")

    layers = G.graph["layers"]
    L = len(layers)

    per_layer_groups: List[List[List[Tuple[int, int]]]] = []

    for layer in range(L):
        nodes = sorted([n for n in G.nodes if n[0] == layer], key=lambda x: x[1])

        weights = []
        for node in nodes:
            weights.append(1 if layer == 0 else int(G.in_degree(node)))

        if decreasing:
            items = sorted(zip(nodes, weights), key=lambda t: t[1], reverse=True)
            nodes2 = [n for n, _ in items]
            weights2 = [w for _, w in items]
        else:
            nodes2, weights2 = nodes, weights

        groups = bin_pack_best_fit(nodes2, weights2, capacity)
        per_layer_groups.append(groups)

    groups_global: List[List[Tuple[int, int]]] = [g for layer_groups in per_layer_groups for g in layer_groups]

    if len(groups_global) > N:
        raise RuntimeError(f"Infeasible: produced {len(groups_global)} groups (> N={N}).")

    while len(groups_global) < N:
        max_idx = max(range(len(groups_global)), key=lambda k: len(groups_global[k])) if groups_global else -1
        if max_idx == -1:
            groups_global.append([])
            continue

        largest = groups_global[max_idx]
        if len(largest) <= 1:
            groups_global.append([])
            continue

        half = len(largest) // 2
        new_group = [largest.pop() for _ in range(half)]
        groups_global.append(new_group)

    return groups_global