from typing import List, Tuple

import networkx as nx


Node = Tuple[int, int]          # (layer_idx, neuron_idx)
Groups = List[List[Node]]      # list of grouped neurons

def build_wctg(G: nx.DiGraph, groups: Groups) -> nx.DiGraph:
    """
    Build WCTG as a NetworkX DiGraph.

    - Group layer = layer of first neuron in group.
    - Only edges from layer ℓ to ℓ+1 are allowed.
    - Edge weight = number of unique source neurons in group i
      that have >=1 outgoing connection to any neuron in group j.
    - Groups on the last layer naturally have no outgoing edges.
    """
    wctg = nx.DiGraph()

    if not groups:
        return wctg

    group_to_layer = [grp[0][0] for grp in groups]
    group_N = len(groups)

    for i in range(group_N):
        wctg.add_node(i, layer=group_to_layer[i], size=len(groups[i]), members=groups[i])

    for i in range(group_N):          # <-- was range(group_N - 1)
        src_layer = group_to_layer[i]
        next_layer = src_layer + 1

        next_group_indices = [
            j for j in range(group_N)
            if group_to_layer[j] == next_layer
        ]

        src_group = groups[i]

        for j in next_group_indices:
            dst_set = set(groups[j])
            talking = 0

            for u in src_group:
                if not G.has_node(u):
                    continue
                for v in G.successors(u):
                    if v in dst_set:
                        talking += 1
                        break   # count each source neuron at most once

            if talking > 0:
                wctg.add_edge(i, j, weight=talking)

    return wctg

def build_mesh(rows: int, cols: int, directed: bool = False) -> nx.Graph:
    """
    Build a 2-D grid mesh graph with nodes keyed as (row, col) tuples.

    Parameters
    ----------
    rows : int   Number of rows.
    cols : int   Number of columns.
    directed : bool
        If True returns a DiGraph where edges flow right and down only.
        If False (default) returns an undirected Graph.

    Returns
    -------
    nx.Graph or nx.DiGraph
    """
    G = nx.DiGraph() if directed else nx.Graph()

    # Add all nodes with positional metadata
    for r in range(rows):
        for c in range(cols):
            G.add_node((r, c), row=r, col=c)

    # Add edges to cardinal neighbours
    for r in range(rows):
        for c in range(cols):
            if directed:
                # right and down only
                neighbours = [(r, c + 1), (r + 1, c)]
            else:
                neighbours = [(r, c + 1), (r + 1, c)]   # undirected: same pairs, no duplicates

            for nr, nc in neighbours:
                if 0 <= nr < rows and 0 <= nc < cols:
                    G.add_edge((r, c), (nr, nc))

    return G

import numpy as np


def get_wctg_matrix(W: nx.DiGraph) -> tuple[np.ndarray, list]:
    """
    Convert a WCTG DiGraph to a weighted adjacency matrix.

    Returns
    -------
    matrix : np.ndarray, shape (N, N)
        matrix[i][j] = edge weight if edge i→j exists, else 0.
    node_ids : list
        Ordered list of node ids corresponding to matrix rows/columns.
    """
    node_ids = sorted(W.nodes())
    idx = {n: i for i, n in enumerate(node_ids)}
    N = len(node_ids)

    matrix = np.zeros((N, N), dtype=float)

    for u, v, data in W.edges(data=True):
        matrix[idx[u], idx[v]] = data.get("weight", 1.0)

    return matrix, node_ids


def get_spg_matrix(G: nx.Graph) -> tuple[np.ndarray, list]:
    """
    Run Floyd-Warshall on a mesh graph and return the all-pairs shortest path matrix.

    Edge weight defaults to 1 if not set.

    Returns
    -------
    dist : np.ndarray, shape (N, N)
        dist[i][j] = shortest path length from node i to node j.
        np.inf means unreachable.
    node_ids : list
        Ordered list of (row, col) node ids corresponding to matrix rows/columns.
    """
    node_ids = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(node_ids)}
    N = len(node_ids)

    dist = np.full((N, N), np.inf)
    np.fill_diagonal(dist, 0.0)

    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        dist[idx[u], idx[v]] = w
        if not isinstance(G, nx.DiGraph):   # undirected: both directions
            dist[idx[v], idx[u]] = w

    # Floyd-Warshall
    for k in range(N):
        for i in range(N):
            if dist[i, k] == np.inf:        # minor optimisation: skip dead rows
                continue
            for j in range(N):
                through_k = dist[i, k] + dist[k, j]
                if through_k < dist[i, j]:
                    dist[i, j] = through_k

    return dist, node_ids


# ===========================================================================
# Cross Group Communication Cost Metric  — Structural Sparsity Cost
# ===========================================================================
#
# Motivation
# ----------
# The standard WCTG×hops cost counts every graph edge (active connection
# between two neurons in different groups) regardless of whether the
# corresponding weight is zero.  After unstructured L1 edge pruning many edges
# in the PrunableNetwork graph still exist as structural entries but carry
# a zero weight — they contribute nothing to the forward pass yet inflate
# the communication cost.  This may make L1-pruned models appear cheaper than
# they really are on real hardware.
#
# Alternative Cost Calculation
# ---
# Define a second WCTG that only counts connections whose underlying weight
# |w| > eps, then compute cost = Σ sparsity_wctg(i,j) × hops(i,j).
# This correctly reflects the cross group traffic that will actually traverse the NoC.
#
# Usage
# -----
#   from mapping_helpers import build_wctg_structural, compute_structural_cost
#
#   sp_wctg = build_wctg_structural(net.graph, groups, model, eps=1e-8)
#   sp_cost = compute_structural_cost(sp_wctg, spg_matrix, spg_index)
# ===========================================================================

import torch
import torch.nn as nn


def build_wctg_structural(
    G: nx.DiGraph,
    groups: "Groups",
    model: "nn.Module" = None,
    eps: float = 1e-8,
) -> nx.DiGraph:
    """
    Build a WCTG that counts only non-zero weight connections.

    Identical logic to build_wctg() except that edge weights are read
    directly from the PrunableNetwork graph G (each edge carries a
    'weight' attribute set by construct_nn_graph). Only edges with
    |weight| > eps are counted — zero-weight edges from unstructured
    L1 pruning are excluded.

    This reflects real NoC traffic: an L1-pruned zero weight
    carries no data even though the graph edge still exists structurally.

    Args:
        G       : PrunableNetwork graph (nx.DiGraph). Edges must have a
                  'weight' attribute (set by construct_nn_graph).
        groups  : list of node groups from NoCMapper.
        model   : unused, kept for API compatibility.
        eps     : threshold below which a weight is considered zero.

    Returns:
        nx.DiGraph — sparse WCTG counting only non-zero-weight traffic.
    """
    wctg = nx.DiGraph()
    if not groups:
        return wctg

    group_to_layer = [grp[0][0] for grp in groups]
    group_N = len(groups)

    for i in range(group_N):
        wctg.add_node(i, layer=group_to_layer[i],
                      size=len(groups[i]), members=groups[i])

    for i in range(group_N):
        src_layer  = group_to_layer[i]
        next_layer = src_layer + 1

        next_group_indices = [
            j for j in range(group_N)
            if group_to_layer[j] == next_layer
        ]

        src_group = groups[i]

        for j in next_group_indices:
            dst_set = set(groups[j])
            talking = 0

            for u in src_group:
                if not G.has_node(u):
                    continue

                # Count u if it has at least one non-zero-weight edge to group j
                found = False
                for v in G.successors(u):
                    if v not in dst_set:
                        continue
                    # Read the actual weight stored on this graph edge
                    edge_weight = G[u][v].get("weight", 0.0)
                    if abs(edge_weight) > eps:
                        found = True
                        break  # one non-zero connection is enough

                if found:
                    talking += 1

            if talking > 0:
                wctg.add_edge(i, j, weight=talking)

    return wctg


def compute_structural_cost(
    sp_wctg: nx.DiGraph,
    spg: dict,
) -> float:
    """
    Compute Cross Group communication cost using the structural sparsity WCTG.
    This captures cross-group communication cost based on the structural topology 
    of which groups are connected, ignoring fine-grained weight magnitudes.
	Builds a structural version of the WCTG — one that abstracts away individual neurons 
    and instead works at the level of groups/tiles (from noc.groups). It then computes 
    cost on the shortest-path graph (noc.spg).
    cost = Σ_{(i,j) in sp_wctg} weight(i,j) × hops(i,j)

    Uses noc.spg which is a dict {(core_i, core_j): hops} with integer
    core index keys — matching the integer node ids in sp_wctg directly.
    This avoids the (row,col) vs integer index mismatch of the matrix API.

    Args:
        sp_wctg : DiGraph from build_wctg_structural(). Nodes are integer
                  core indices (0..P-1), matching noc.spg keys.
        spg     : dict {(core_i, core_j): hops} from noc.spg.

    Returns:
        float — structural sparsity communication cost.
    """
    total = 0.0
    for u, v, data in sp_wctg.edges(data=True):
        hops   = spg.get((u, v), 0.0)
        weight = data.get("weight", 1.0)
        total += weight * hops
    return float(total)
