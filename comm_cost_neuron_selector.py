# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  NOT USED IN FINAL COMPARISON — SEE iterative_prune_neurons_sa INSTEAD    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 
# comm_cost_neuron_selector.py
#
# Two-stage neuron selection for communication-cost-aware structural pruning.
#
# Stage 1 — in-degree filter  (fast, uses existing heuristic)
#   Nominates a candidate shortlist using select_neurons_high_indegree_low_magnitude.
#   Tighten low_mag_percentile (e.g. 25.0) and degree_quantile (e.g. 0.75) to
#   keep the shortlist small (10–30 neurons) and Stage-2 fast.
#
# Stage 2 — SA sensitivity ranking  (accurate, runs only over the shortlist)
#   For each candidate, temporarily removes it from a *graph-only* copy,
#   runs grouping (fixed from the current state) + WCTG + SA (sa_runs times),
#   and records the mean cost reduction.  Candidates are sorted descending.
#
# Speed optimisations vs. the naive version
# ------------------------------------------
#   OPT-1  Tight Stage-1 filter
#          sa_low_mag_percentile=25.0 + sa_degree_quantile=0.75 keeps the
#          shortlist to ~10-30 neurons instead of ~370, giving a 15-30x speedup.
#
#   OPT-2  Batch removal (batch_size > 1)
#          Instead of re-scoring after every single removal, the outer loop
#          removes the top-k from one ranking pass, then re-scores.  With
#          batch_size=5 the number of full SA scoring rounds drops by ~5x.
#          Accuracy trade-off: slight, because neuron interactions within a
#          small batch are ignored.  Acceptable when neurons are independently
#          prunable (low mutual dependence), which is typical after L1 pre-pruning.
#
#   OPT-3  Mesh and SPG matrix caching
#          The mesh topology and SPG matrix are identical for every probe call
#          within a scoring round.  They are now cached in a module-level dict
#          keyed by (mesh_rows, mesh_cols), eliminating redundant rebuilds.
#
#   OPT-4  SA runs reduced to 1 for candidate screening
#          sa_runs=1 is sufficient to rank candidates reliably when the
#          shortlist is already small and homogeneous.  Use sa_runs=3 only
#          if you observe noisy rankings across repeated runs.
#
# Recommended fast settings (configure in main file):
#   sa_low_mag_percentile = 25.0
#   sa_degree_quantile    = 0.75
#   sa_runs               = 1
#   sa_batch_size         = 5

from __future__ import annotations

import heapq
import statistics
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from pruning import select_neurons_high_indegree_low_magnitude
from mapping_helpers import build_wctg, build_mesh, get_spg_matrix, get_wctg_matrix
from mapping_sa import map_nodes
from grouping_heuristic import indegree_grouping


# ---------------------------------------------------------------------------
# Module-level mesh/SPG cache (OPT-3)
# Keyed by (mesh_rows, mesh_cols).  Populated on first use, reused thereafter.
# ---------------------------------------------------------------------------
_MESH_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _graph_remove_neuron(G: nx.DiGraph, node: Tuple[int, int]) -> nx.DiGraph:
    """
    Return a copy of *G* with *node* and all its incident edges removed.
    Node indices are NOT remapped — the fixed grouping still references old IDs.
    """
    H = G.copy()
    if H.has_node(node):
        H.remove_node(node)
    return H


def _merge_groups_to_fit(groups: list, P: int) -> list:
    """
    If len(groups) > P, pairwise-merge the smallest groups until len <= P.
    map_nodes requires P >= N (number of task groups).
    """
    if len(groups) <= P:
        return groups

    heap = [(len(g), i, g) for i, g in enumerate(groups)]
    heapq.heapify(heap)
    counter = len(groups)

    while len(heap) > P:
        _, _, g1 = heapq.heappop(heap)
        _, _, g2 = heapq.heappop(heap)
        merged = g1 + g2
        heapq.heappush(heap, (len(merged), counter, merged))
        counter += 1

    return [g for _, _, g in heap]


def _fix_grouping(G: nx.DiGraph, N: int, grouping_cap: float) -> list:
    """
    Compute the fixed grouping for the current graph using NoCMapper.N as the
    group budget.  Falls back to N evenly-sized chunks if indegree_grouping
    raises RuntimeError (infeasible packing).
    """
    try:
        return indegree_grouping(G, N, grouping_cap)
    except RuntimeError:
        all_nodes = list(G.nodes())
        chunk = max(1, len(all_nodes) // N)
        groups = [all_nodes[i:i + chunk] for i in range(0, len(all_nodes), chunk)]
        if len(groups) > N:
            tail = groups[N - 1:]
            groups = groups[:N - 1] + [[n for g in tail for n in g]]
        return groups


def _sa_cost_on_graph(
    G: nx.DiGraph,
    groups: list,
    P: int,
    mesh_rows: int,
    mesh_cols: int,
    sa_runs: int,
) -> float:
    """
    Build WCTG from *G* using *groups*, run SA *sa_runs* times, return mean cost.

    The mesh and SPG matrix are read from the module-level cache (OPT-3).
    They are built on first use and never rebuilt for the same topology.
    """
    live_nodes = set(G.nodes())
    filtered = [[n for n in g if n in live_nodes] for g in groups]
    filtered = [g for g in filtered if g]
    filtered = _merge_groups_to_fit(filtered, P)

    # Mesh / SPG cache (OPT-3)
    cache_key = (mesh_rows, mesh_cols)
    if cache_key not in _MESH_CACHE:
        mesh = build_mesh(mesh_rows, mesh_cols)
        spg_matrix, _ = get_spg_matrix(mesh)
        _MESH_CACHE[cache_key] = spg_matrix
    spg_matrix = _MESH_CACHE[cache_key]

    wctg = build_wctg(G, filtered)
    wctg_matrix, _ = get_wctg_matrix(wctg)

    costs = [map_nodes(wctg_matrix, spg_matrix, P)["cost"] for _ in range(sa_runs)]
    return statistics.mean(costs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_neurons_sa_ranked(
    G: nx.DiGraph,
    *,
    # ---- Stage-1 (in-degree filter) parameters ----
    low_mag_percentile: float = 25.0,
    degree_thresholds: Optional[Dict[int, int]] = None,
    degree_quantile: Optional[float] = None,
    min_in_degree: int = 1,
    # ---- Stage-2 (SA sensitivity) parameters ----
    P: int,
    N: int,
    mesh_rows: int,
    mesh_cols: int,
    grouping_cap: float = 10_000.0,
    sa_runs: int = 1,
    # ---- Behaviour ----
    debug: bool = True,
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Two-stage neuron selector.  Returns candidates sorted by estimated
    SA cost reduction, highest first.

    Speed is controlled by Stage-1 filter tightness and sa_runs.
    See module header for recommended fast settings.

    Args:
        G:                  Current NN graph from PrunableNetwork.graph.
        low_mag_percentile: Bottom X% by mean |incoming weight| qualify.
                            Lower = fewer candidates = faster.  25.0 recommended.
        degree_thresholds:  Per-layer fixed in-degree threshold overrides.
        degree_quantile:    Neurons above this in-degree quantile qualify.
                            Higher = fewer candidates = faster.  0.75 recommended.
        min_in_degree:      Neurons with fewer incoming edges are excluded.
        P:                  Mesh processing elements (NoCMapper.P).
        N:                  Max task groups (NoCMapper.N).  Must satisfy N <= P.
        mesh_rows:          Mesh rows (NoCMapper._mesh_rows).
        mesh_cols:          Mesh cols (NoCMapper._mesh_cols).
        grouping_cap:       Bin-pack weight cap (NoCMapper._grouping_cap).
        sa_runs:            SA repetitions per candidate.  1 recommended for speed.
        debug:              Verbose per-candidate logging.

    Returns:
        List of (node, mean_cost_reduction) sorted descending.
        Positive reduction = removing this neuron lowers cost (good).
        Negative reduction = removing it raises cost (stop pruning).
    """
    # ------------------------------------------------------------------
    # Stage 1: in-degree filter → candidate shortlist
    # ------------------------------------------------------------------
    candidates = select_neurons_high_indegree_low_magnitude(
        G,
        low_mag_percentile=low_mag_percentile,
        degree_thresholds=degree_thresholds,
        degree_quantile=degree_quantile,
        min_in_degree=min_in_degree,
        debug=debug,
    )

    if not candidates:
        if debug:
            print("[sa_ranked] Stage-1 returned no candidates.")
        return []

    if debug:
        print(f"\n[sa_ranked] Stage-1 shortlist: {len(candidates)} candidate(s).")
        print(f"[sa_ranked] Computing baseline SA cost ({sa_runs} SA run(s) per probe)...")

    # ------------------------------------------------------------------
    # Fix grouping once on the unmodified graph (reused for all probes)
    # ------------------------------------------------------------------
    fixed_groups = _fix_grouping(G, N, grouping_cap)

    # ------------------------------------------------------------------
    # Baseline SA cost
    # ------------------------------------------------------------------
    baseline_cost = _sa_cost_on_graph(G, fixed_groups, P, mesh_rows, mesh_cols, sa_runs)

    if debug:
        print(f"[sa_ranked] Baseline mean cost: {baseline_cost:.2f}")

    # ------------------------------------------------------------------
    # Stage 2: probe each candidate
    # ------------------------------------------------------------------
    scored: List[Tuple[Tuple[int, int], float]] = []

    for i, node in enumerate(candidates):
        G_probe    = _graph_remove_neuron(G, node)
        probe_cost = _sa_cost_on_graph(G_probe, fixed_groups, P, mesh_rows, mesh_cols, sa_runs)
        reduction  = baseline_cost - probe_cost

        if debug:
            print(
                f"[sa_ranked]  [{i+1}/{len(candidates)}] node={node} | "
                f"probe={probe_cost:.2f} | Δcost={reduction:+.2f}"
            )

        scored.append((node, reduction))

    scored.sort(key=lambda t: t[1], reverse=True)

    if debug:
        print("\n[sa_ranked] Ranking (node → Δcost):")
        for node, red in scored:
            print(f"  {'OK' if red > 0 else '--'} {node}  Δcost={red:+.2f}")

    return scored


# ---------------------------------------------------------------------------
# PrunableNetwork integration — iterative batch pruning loop
# ---------------------------------------------------------------------------

def iterative_prune_neurons_sa(
    net,
    noc,
    *,
    # Stage-1 parameters
    low_mag_percentile: float = 25.0,
    degree_thresholds: Optional[Dict[int, int]] = None,
    degree_quantile: Optional[float] = None,
    min_in_degree: int = 1,
    # Stage-2 parameters
    sa_runs: int = 1,
    # Batch removal (OPT-2)
    batch_size: int = 5,
    # Stopping criteria
    min_cost_reduction: float = 0.0,
    max_neurons: Optional[int] = None,
    # Behaviour
    debug: bool = True,
) -> Tuple[int, List[Tuple[Tuple[int, int], float]]]:
    """
    Iterative batch neuron pruning driven by SA cost sensitivity.

    Each outer (scoring) iteration:
      1. Scores all Stage-1 candidates via SA probe (select_neurons_sa_ranked).
      2. Removes the top `batch_size` candidates whose reduction > min_cost_reduction.
      3. Stops the batch early if a candidate's reduction <= min_cost_reduction.
      4. Rebuilds the graph (via net.prune_neurons) and re-scores.

    This amortises the expensive SA scoring over `batch_size` removals, giving
    a ~batch_size speedup over single-removal iteration.

    Args:
        net:                 PrunableNetwork to prune (mutated in place).
        noc:                 NoCMapper — supplies P, N, mesh dims, grouping_cap.
        low_mag_percentile:  Stage-1 filter percentile.  25.0 recommended.
        degree_thresholds:   Stage-1 per-layer degree overrides.
        degree_quantile:     Stage-1 in-degree quantile.  0.75 recommended.
        min_in_degree:       Stage-1 minimum in-degree to be a candidate.
        sa_runs:             SA reps per candidate probe.  1 recommended.
        batch_size:          Neurons removed per scoring round before re-scoring.
                             5 is a good balance; 10 for maximum speed.
        min_cost_reduction:  Stop when best candidate's reduction <= this value.
        max_neurons:         Hard cap on total neurons removed.
        debug:               Verbose logging.

    Returns:
        (n_pruned, history)
        n_pruned: total neurons removed across all iterations.
        history:  list of (node, estimated_cost_reduction) in removal order.
    """
    n_pruned  = 0
    history:   List[Tuple[Tuple[int, int], float]] = []
    iteration = 0

    while True:
        if max_neurons is not None and n_pruned >= max_neurons:
            if debug:
                print(f"[iterative] Reached max_neurons={max_neurons}. Stopping.")
            break

        iteration += 1
        G = net.graph

        if debug:
            print(f"\n[iterative] === Scoring round {iteration} ===")
            print(f"[iterative] Graph: {G.number_of_nodes()} nodes, "
                  f"{G.number_of_edges()} edges | removed so far: {n_pruned}")

        ranked = select_neurons_sa_ranked(
            G,
            low_mag_percentile=low_mag_percentile,
            degree_thresholds=degree_thresholds,
            degree_quantile=degree_quantile,
            min_in_degree=min_in_degree,
            P=noc.P,
            N=noc.N,
            mesh_rows=noc._mesh_rows,
            mesh_cols=noc._mesh_cols,
            grouping_cap=noc._grouping_cap,
            sa_runs=sa_runs,
            debug=debug,
        )

        if not ranked:
            if debug:
                print("[iterative] No candidates from Stage-1. Stopping.")
            break

        # Stop if even the top candidate is not beneficial
        if ranked[0][1] <= min_cost_reduction:
            if debug:
                print(
                    f"[iterative] Best candidate {ranked[0][0]} gives "
                    f"Δcost={ranked[0][1]:+.2f} <= threshold {min_cost_reduction}. Stopping."
                )
            break

        # Batch: remove top-k beneficial candidates from this round's ranking
        removed_this_round = 0
        for node, reduction in ranked[:batch_size]:
            if reduction <= min_cost_reduction:
                if debug:
                    print(
                        f"[iterative]   Batch stopping early — "
                        f"{node} Δcost={reduction:+.2f} <= threshold"
                    )
                break

            if max_neurons is not None and n_pruned >= max_neurons:
                break

            if debug:
                print(f"[iterative]   Removing {node}  Δcost={reduction:+.2f}")

            net.prune_neurons([node])
            history.append((node, reduction))
            n_pruned += 1
            removed_this_round += 1

        if removed_this_round == 0:
            if debug:
                print("[iterative] No neurons removed this round. Stopping.")
            break

    if debug:
        print(f"\n[iterative] Done. {n_pruned} neuron(s) removed "
              f"across {iteration} scoring round(s).")
        if history:
            print(f"[iterative] Total estimated cost reduction: "
                  f"{sum(r for _, r in history):+.2f}")

    return n_pruned, history
