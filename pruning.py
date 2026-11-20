# pruning_lib.py

from __future__ import annotations

import copy
from typing import List, Tuple, Optional, Dict, Any, Sequence
import copy

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import networkx as nx


# -----------------------------
# Dotted path helpers
# -----------------------------
def _get_module_by_name(root: nn.Module, dotted: str) -> nn.Module:
    cur: nn.Module = root
    for part in dotted.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def _get_parent_and_attr(root: nn.Module, dotted: str):
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]


from typing import List, Tuple, Optional, Sequence, Dict
import copy
import torch
import torch.nn as nn

# -----------------------------
# 1) Neuron pruning (structural)
#    accepts tuples (node_layer, idx) directly
# -----------------------------
def prune_neurons(
    model: nn.Module,
    src: Sequence[Tuple[int, int]],     # (node_layer, idx) from your graph
    *,
    linear_names: Optional[List[str]] = None,
    inplace: bool = False,
) -> nn.Module:
    """
    Prune neurons specified by (node_layer, neuron_idx), where node_layer matches
    construct_nn_graph's node indexing:
      - node_layer=0 is the input activation layer (not prunable here)
      - node_layer=k (k>=1) are activations produced by linear_names[k-1]
    Structural pruning removes selected neurons from the producing Linear's output
    and removes the corresponding input columns in the next Linear.

    Therefore this supports pruning only for:
      1 <= node_layer <= (len(linear_names) - 1)
    i.e., hidden activation layers that have both a producing Linear and a next Linear.

    Args:
        model: PyTorch module
        src: sequence of (node_layer, neuron_idx)
        linear_names: optional precomputed forward-order list of Linear module names.
        inplace: if True, mutate model; else deep-copy.

    Returns:
        pruned model
    """
    if not src:
        return model if inplace else copy.deepcopy(model)

    m = model if inplace else copy.deepcopy(model)

    # Determine forward-order Linear names (your previous pattern)
    if linear_names is None:
        linear_names = [name for name, mod in m.named_modules() if isinstance(mod, nn.Linear)]

    if len(linear_names) < 2:
        raise ValueError(
            f"Need at least 2 nn.Linear layers to prune structurally, found {len(linear_names)}."
        )

    # Group neuron indices by node_layer (graph layer index)
    per_node_layer: Dict[int, List[int]] = {}
    for node_layer, neuron_idx in src:
        per_node_layer.setdefault(node_layer, []).append(neuron_idx)

    # Apply pruning per node_layer
    # node_layer must be in [1, len(linear_names)-1]
    max_prunable_node_layer = len(linear_names) - 1

    for node_layer, neuron_indices in per_node_layer.items():
        if node_layer == 0:
            raise ValueError(
                "node_layer=0 corresponds to input features; "
                "structural neuron pruning as implemented here does not prune inputs."
            )
        if not (1 <= node_layer <= max_prunable_node_layer):
            raise ValueError(
                f"node_layer={node_layer} is not structurally prunable with this function. "
                f"Valid node_layer range is [1, {max_prunable_node_layer}] given "
                f"{len(linear_names)} Linear layers."
            )

        # Map graph node_layer -> Linear indices
        prune_linear_idx = node_layer - 1
        next_linear_idx = node_layer

        prune_layer_name = linear_names[prune_linear_idx]
        next_layer_name = linear_names[next_linear_idx]

        prune_layer = _get_module_by_name(m, prune_layer_name)
        next_layer = _get_module_by_name(m, next_layer_name)

        if not isinstance(prune_layer, nn.Linear):
            raise ValueError(f"{prune_layer_name} must be nn.Linear")
        if not isinstance(next_layer, nn.Linear):
            raise ValueError(f"{next_layer_name} must be nn.Linear")

        # dedupe + bounds check
        out_features = prune_layer.out_features
        uniq = sorted(set(neuron_indices))
        if uniq and (uniq[0] < 0 or uniq[-1] >= out_features):
            raise IndexError(
                f"Neuron index out of range for {prune_layer_name}: "
                f"valid [0, {out_features-1}], got {uniq[0]}..{uniq[-1]}"
            )

        mask = torch.ones(out_features, dtype=torch.bool, device=prune_layer.weight.device)
        mask[uniq] = False

        kept = int(mask.sum().item())
        if kept <= 0:
            raise ValueError(f"Cannot prune all neurons in {prune_layer_name}.")

        new_prune = nn.Linear(
            prune_layer.in_features,
            kept,
            bias=prune_layer.bias is not None,
        ).to(prune_layer.weight.device)

        new_next = nn.Linear(
            kept,
            next_layer.out_features,
            bias=next_layer.bias is not None,
        ).to(prune_layer.weight.device)

        with torch.no_grad():
            new_prune.weight.copy_(prune_layer.weight[mask])
            if prune_layer.bias is not None:
                new_prune.bias.copy_(prune_layer.bias[mask])

            new_next.weight.copy_(next_layer.weight[:, mask])
            if next_layer.bias is not None:
                new_next.bias.copy_(next_layer.bias)

        # Swap modules back into the model
        p1, a1 = _get_parent_and_attr(m, prune_layer_name)
        if a1.isdigit():
            p1[int(a1)] = new_prune
        else:
            setattr(p1, a1, new_prune)

        p2, a2 = _get_parent_and_attr(m, next_layer_name)
        if a2.isdigit():
            p2[int(a2)] = new_next
        else:
            setattr(p2, a2, new_next)

    return m


def prune_link(
    model: nn.Module,
    *,
    src: Tuple[int, int],   # (node_layer, idx)
    dst: Tuple[int, int],   # (node_layer, idx)
    linear_names: Optional[Sequence[str]] = None,
    inplace: bool = False,
    persistent: bool = True,
) -> nn.Module:
    """
    Prune a single link specified by tuple nodes.

    src=(k, j) means neuron j in node-layer k
    dst=(k+1, i) means neuron i in node-layer k+1

    This prunes the weight in Linear layer k:
        W[i, j] = 0  (and optionally bakes it in)

    Args:
        model: nn.Module
        src, dst: tuple node ids
        linear_names: optional list of Linear module names in forward order.
                      If None, inferred from model.named_modules() order.
        inplace: mutate model or return a copy
        persistent: if True, prune.remove() to bake mask into weights

    Returns:
        pruned model (same object if inplace=True)
    """
    (src_layer, src_idx) = src
    (dst_layer, tgt_idx) = dst

    if dst_layer != src_layer + 1:
        raise ValueError(f"Tuple edge must go k->k+1. Got {src_layer}->{dst_layer}.")

    m = model if inplace else copy.deepcopy(model)

    # Determine forward-order Linear names
    if linear_names is None:
        linear_names = [name for name, mod in m.named_modules() if isinstance(mod, nn.Linear)]

    if not (0 <= src_layer < len(linear_names)):
        raise ValueError(
            f"src_layer={src_layer} implies Linear index {src_layer}, "
            f"but only {len(linear_names)} Linear layers were found."
        )

    target_layer_name = linear_names[src_layer]
    layer = _get_module_by_name(m, target_layer_name)

    if not isinstance(layer, nn.Linear):
        raise ValueError(f"{target_layer_name} must be nn.Linear")

    if not (0 <= src_idx < layer.in_features):
        raise IndexError(f"src_idx={src_idx} out of range for {target_layer_name}.in_features={layer.in_features}")
    if not (0 <= tgt_idx < layer.out_features):
        raise IndexError(f"tgt_idx={tgt_idx} out of range for {target_layer_name}.out_features={layer.out_features}")

    # Build/update mask
    if hasattr(layer, "weight_mask"):
        mask = layer.weight_mask.detach().clone()
    else:
        mask = torch.ones_like(layer.weight)
        #mask = (layer.weight.detach().abs() > 0).float()

    mask[tgt_idx, src_idx] = 0.0

    # Avoid stacking pruning wrappers
    if hasattr(layer, "weight_orig"):
        prune.remove(layer, "weight")

    prune.custom_from_mask(layer, "weight", mask)

    if persistent:
        prune.remove(layer, "weight")

    return m

# -----------------------------
# 3) Save / Load
# -----------------------------
def save_model(path: str, model: nn.Module):
    torch.save(model.state_dict(), path)


def load_model(
    path: str,
    model_class,
    *,
    device="cuda",
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Module:

    device = torch.device(device)
    state_dict = torch.load(path, map_location=device)
    model_kwargs = model_kwargs or {}

    model = model_class(**model_kwargs).to(device)

    for key, tensor in state_dict.items():
        if not key.endswith(".weight") or tensor.ndim != 2:
            continue

        module_name = key[:-7]
        module = _get_module_by_name(model, module_name)
        if not isinstance(module, nn.Linear):
            continue

        out_f, in_f = tensor.shape
        if module.in_features != in_f or module.out_features != out_f:
            new_lin = nn.Linear(in_f, out_f, bias=(module.bias is not None)).to(device)
            parent, attr = _get_parent_and_attr(model, module_name)
            if attr.isdigit():
                parent[int(attr)] = new_lin
            else:
                setattr(parent, attr, new_lin)

    model.load_state_dict(state_dict, strict=True)
    return model


def l1_prune_links_global(
    model: nn.Module,
    amount: float = 0.25,
    *,
    inplace: bool = True,
    persistent: bool = True,
) -> nn.Module:
    """
    Global L1 unstructured pruning over all nn.Linear weights.

    amount=0.25 means prune 25% of all Linear weights (links) globally, by smallest |w|.

    If persistent=True, bakes pruning into weights immediately via prune.remove,
    so save/load is plain state_dict and visualization can mark pruned links by w==0.
    """
    if not (0.0 <= amount <= 1.0):
        raise ValueError("amount must be in [0, 1].")

    m = model if inplace else copy.deepcopy(model)

    # Collect parameters to prune
    params_to_prune = []
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            # if previously pruned and still wrapped, make it plain before re-pruning
            if hasattr(mod, "weight_orig"):
                prune.remove(mod, "weight")
            params_to_prune.append((mod, "weight"))

    if not params_to_prune:
        raise ValueError("No nn.Linear layers found to prune.")

    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )

    if persistent:
        for mod, pname in params_to_prune:
            if hasattr(mod, "weight_orig"):
                prune.remove(mod, pname)

    return m

def select_neurons_high_indegree_low_magnitude(
    G: nx.DiGraph,
    *,
    eps: float = 1e-12,
    low_mag_percentile: float = 25.0,
    degree_thresholds: Optional[Dict[int, int]] = None,
    degree_quantile: Optional[float] = None,
    min_in_degree: int = 1,
    debug: bool = True,
) -> List[Tuple[int, int]]:
    """
    Select neurons (node_layer, idx) with high in-degree AND low mean |incoming weight|.
    Only considers prunable hidden layers (node_layer 1..L-2).
    """
    if degree_thresholds is not None and degree_quantile is not None:
        raise ValueError("Use either degree_thresholds OR degree_quantile, not both.")

    layers = G.graph["layers"]
    n_layers = len(layers)
    selected = []

    for node_layer in range(1, n_layers - 1):
        nodes = [n for n, d in G.nodes(data=True) if d["node_layer"] == node_layer]
        if not nodes:
            continue

        in_deg = np.array([G.in_degree(n) for n in nodes], dtype=np.int64)
        mag = np.array([
            np.mean([abs(d["weight"]) for _, _, d in G.in_edges(n, data=True)]) if G.in_degree(n) > 0 else 0.0
            for n in nodes
        ])

        valid = in_deg >= min_in_degree

        if debug:
            print(f"\n[select_neurons] Layer {node_layer} — {len(nodes)} neuron(s)")
            print(f"  in_degree : {in_deg.tolist()}")
            print(f"  magnitude : {np.round(mag, 4).tolist()}")
            print(f"  valid (in_deg >= {min_in_degree}): {valid.tolist()}")

        if not np.any(valid):
            if debug:
                print(f"  => All filtered by min_in_degree. Skipping.")
            continue

        # low magnitude
        mag_thr = np.percentile(mag[valid], low_mag_percentile)
        low_mag_mask = (mag < mag_thr) & valid

        # high in-degree
        if degree_thresholds is not None and node_layer in degree_thresholds:
            deg_thr = degree_thresholds[node_layer]
            high_deg_mask = (in_deg > deg_thr) & valid
        elif degree_quantile is not None:
            deg_thr = np.quantile(in_deg[valid].astype(np.float64), degree_quantile)
            high_deg_mask = (in_deg.astype(np.float64) > deg_thr) & valid
        else:
            deg_thr = int(in_deg[valid].max())
            high_deg_mask = (in_deg >= deg_thr) & valid

        if debug:
            print(f"  mag_thr (p{low_mag_percentile}): {mag_thr:.4f}")
            print(f"  low_mag_mask  : {low_mag_mask.tolist()}")
            print(f"  deg_thr       : {deg_thr}")
            print(f"  high_deg_mask : {high_deg_mask.tolist()}")
            print(f"  intersection  : {(high_deg_mask & low_mag_mask).tolist()}")

        for i, node in enumerate(nodes):
            if high_deg_mask[i] and low_mag_mask[i]:
                selected.append(node)

    print(f"[select_neurons] Selected {len(selected)} neuron(s): {[str(n) for n in selected]}")
    return selected

def prune_high_indegree_low_magnitude(
    model: nn.Module,
    G: nx.DiGraph,
    *,
    eps: float = 1e-12,
    low_mag_percentile: float = 25.0,
    degree_thresholds: Optional[Dict[int, int]] = None,
    degree_quantile: Optional[float] = None,
    min_in_degree: int = 1,
    mode: str = "neurons",
    inplace: bool = False,
) -> nn.Module:
    if mode not in ("neurons", "links"):
        raise ValueError(f"mode must be 'neurons' or 'links', got '{mode}'.")

    linear_names = G.graph["linear_names"]

    candidates = select_neurons_high_indegree_low_magnitude(
        G,
        eps=eps,
        low_mag_percentile=low_mag_percentile,
        degree_thresholds=degree_thresholds,
        degree_quantile=degree_quantile,
        min_in_degree=min_in_degree,
    )

    if not candidates:
        return model if inplace else copy.deepcopy(model)

    if mode == "neurons":
        return prune_neurons(
            model,
            candidates,
            linear_names=linear_names,
            inplace=inplace,
        )

    else:
        m = model if inplace else copy.deepcopy(model)
        for (node_layer, neuron_idx) in candidates:
            for src, _, _ in G.in_edges((node_layer, neuron_idx), data=True):
                m = prune_link(
                    m,
                    src=src,
                    dst=(node_layer, neuron_idx),
                    linear_names=linear_names,
                    inplace=True,
                    persistent=True,
                )
        return m
    
# =============================================================================
# Iterative CommCost neuron pruning with fresh grouping per step.
#
# ----------------------
# The original prune_high_indegree_low_magnitude scores ALL neurons in one
# pass on graph G₀, then removes them all at once.  The grouping used for
# scoring is G₀'s grouping.  After removal the graph is rebuilt as G₁ with
# a completely different grouping — so the SA mapping that measures the final
# cost runs on G₁'s grouping, not G₀'s.  The pruning decisions were made
# against a metric that no longer matches what is measured.
# This does not reflect the full capability of Comm. cost aware pruning approach.
#
# What this function does instead
# --------------------------------
# Each iteration:
#   1. Re-run indegree_grouping on the CURRENT graph  (fast, no SA needed)
#   2. Score every hidden neuron by CommCost using THAT grouping
#   3. Remove only the single best candidate
#   4. Rebuild the graph (net.prune_neurons + net.resync_graph)
#   5. Repeat until no candidate scores above min_score or max_neurons hit
#
# The grouping used for scoring is always the grouping that will actually
# be seen by the next SA mapping, so the metric stays aligned.
#
# Cost
# ----
# Per iteration: one indegree_grouping call (milliseconds) + one scoring
# pass (microseconds).  No SA runs, no model weight cloning.
# Typical total: a few seconds for 50–200 neurons removed.
#
# Required imports (add to top of pruning.py if not already present):
#   from grouping_heuristic import indegree_grouping
# =============================================================================

from grouping_heuristic import indegree_grouping   # add to pruning.py imports


def _commcost_score_neuron(
    G: "nx.DiGraph",
    node: "Tuple[int, int]",
    node_to_group: "Dict",
    group_to_nodes: "Dict",
) -> float:
    """
    Compute the communication cost contribution of removing `node`.

    Score = sum of |weight| for each cross-group edge touching `node`,
            counting both incoming edges (from other groups into node)
            and outgoing edges (from node into other groups in next layer).

    Removing a neuron eliminates both its incoming and outgoing cross-group
    edges from the WCTG.  The original one-sided score (in-edges only)
    underestimates the cost of high-fanout neurons in wide networks (e.g.
    1024-1024) where outgoing weight magnitude dominates.
    """
    my_group = node_to_group.get(node)
    score = 0.0
    for src, _, data in G.in_edges(node, data=True):
        src_group = node_to_group.get(src)
        if src_group is not None and src_group != my_group:
            score += abs(data.get("weight", 0.0))
    for _, dst, data in G.out_edges(node, data=True):       #
        dst_group = node_to_group.get(dst)                  #
        if dst_group is not None and dst_group != my_group: #
            score += abs(data.get("weight", 0.0))           #
    return score


def iterative_commcost_neuron_pruning(
    net,                          # PrunableNetwork instance (mutated in place)
    *,
    N: int,
    grouping_cap: float = 10_000.0,
    low_mag_percentile: float = 75.0,
    degree_quantile: Optional[float] = 0.25,
    degree_thresholds: Optional[Dict[int, int]] = None,
    min_in_degree: int = 0,
    min_score: float = 0.0,
    max_neurons: Optional[int] = None,
    debug: bool = True,
) -> Tuple[int, List[Tuple[Tuple[int, int], float]]]:
    """
    Iterative CommCost-aware neuron pruning with fresh grouping per step.

    Unlike the one-shot prune_high_indegree_low_magnitude, this function
    re-computes the grouping after every single neuron removal, so the
    CommCost score is always evaluated against the grouping that the SA
    mapper will actually use.

    Args:
        net:               PrunableNetwork to prune (mutated in place).
        N:                 Max task groups — must match NoCMapper.N.
        grouping_cap:      Bin-pack weight cap — must match NoCMapper._grouping_cap.
        low_mag_percentile: Stage-1 magnitude filter (same as original heuristic).
        degree_quantile:   Stage-1 in-degree quantile filter (same as original).
        degree_thresholds: Stage-1 per-layer fixed threshold (alternative to quantile).
        min_in_degree:     Minimum in-degree to be a candidate (same as original).
        min_score:         Stop when the best candidate's CommCost score <= this.
                           Do NOT rely on this alone — almost every neuron has at
                           least one nonzero cross-group edge, so the score is rarely
                           zero and the loop will destroy the model if max_neurons is
                           not also set.  Leave at 0.0 and control budget via
                           max_neurons instead.
        max_neurons:       PRIMARY stopping criterion.  Set this to the number of
                           neurons the one-shot CommCost heuristic would remove
                           (i.e. l1_neuron_count - commcost_neuron_count) so IterativeCC
                           removes exactly the same count but chooses WHICH neurons
                           to remove using the fresh-grouping score each iteration.
                           Must not be None when min_score=0.0.
        debug:             Print per-iteration diagnostics.

    Returns:
        (n_pruned, history)
        n_pruned : total neurons removed.
        history  : list of (node, commcost_score) in removal order.
    """
    n_pruned = 0
    history: List[Tuple[Tuple[int, int], float]] = []
    iteration = 0

    while True:
        if max_neurons is not None and n_pruned >= max_neurons:
            if debug:
                print(f"[iterative_commcost] Reached max_neurons={max_neurons}. Stopping.")
            break

        iteration += 1
        G = net.graph   # always fetch fresh after each structural change

        if debug:
            print(f"\n[iterative_commcost] === Iteration {iteration} ===")
            print(f"[iterative_commcost] Graph: {G.number_of_nodes()} nodes, "
                  f"{G.number_of_edges()} edges | removed so far: {n_pruned}")

        # ------------------------------------------------------------------
        # Step 1: Re-run grouping on the current graph.
        # This is the key difference from the original one-shot version.
        # indegree_grouping is fast (pure Python bin-packing, no SA).
        # ------------------------------------------------------------------
        try:
            groups = indegree_grouping(G, N, grouping_cap)
        except RuntimeError:
            if debug:
                print("[iterative_commcost] indegree_grouping infeasible. Stopping.")
            break

        # Build lookup maps for fast per-neuron scoring
        node_to_group: Dict = {}
        group_to_nodes: Dict = {}
        for gid, members in enumerate(groups):
            group_to_nodes[gid] = members
            for node in members:
                node_to_group[node] = gid

        # ------------------------------------------------------------------
        # Step 2: Stage-1 filter — same as original heuristic.
        # Select candidates by high in-degree AND low magnitude.
        # ------------------------------------------------------------------
        candidates = select_neurons_high_indegree_low_magnitude(
            G,
            low_mag_percentile=low_mag_percentile,
            degree_thresholds=degree_thresholds,
            degree_quantile=degree_quantile,
            min_in_degree=min_in_degree,
            debug=False,   # suppress per-layer noise; outer loop is already verbose
        )

        if not candidates:
            if debug:
                print("[iterative_commcost] No candidates from Stage-1 filter. Stopping.")
            break

        # ------------------------------------------------------------------
        # Step 3: Score each candidate by cross-group communication cost.
        # ------------------------------------------------------------------
        scored = []
        for node in candidates:
            score = _commcost_score_neuron(G, node, node_to_group, group_to_nodes)
            scored.append((node, score))

        # Sort descending by cross-group cost contribution
        scored.sort(key=lambda t: t[1], reverse=True)

        best_node, best_score = scored[0]

        if debug:
            print(f"[iterative_commcost] {len(candidates)} candidate(s). "
                  f"Best: {best_node}  cross-group cost={best_score:.6f}")
            if len(scored) > 1:
                print(f"[iterative_commcost] Top-5 candidates:")
                for node, sc in scored[:5]:
                    grp = node_to_group.get(node, "?")
                    print(f"    {node}  score={sc:.6f}  group={grp}")

        # ------------------------------------------------------------------
        # Step 4: Stopping check.
        # ------------------------------------------------------------------
        if best_score <= min_score:
            if debug:
                print(
                    f"[iterative_commcost] Best score {best_score:.6f} <= "
                    f"min_score {min_score}. No cross-group benefit. Stopping."
                )
            break

        # ------------------------------------------------------------------
        # Step 5: Remove the best candidate and rebuild the graph.
        # net.prune_neurons handles both model weight surgery and graph rebuild.
        # ------------------------------------------------------------------
        if debug:
            print(f"[iterative_commcost] Removing {best_node}  "
                  f"(cross-group cost contribution={best_score:.6f})")

        net.prune_neurons([best_node])
        history.append((best_node, best_score))
        n_pruned += 1

    if debug:
        print(f"\n[iterative_commcost] Done. {n_pruned} neuron(s) removed "
              f"across {iteration - 1} successful iteration(s).")
        if history:
            print(f"[iterative_commcost] Removal log (node → cross-group score):")
            for node, sc in history:
                print(f"    {node}  score={sc:.6f}")

    return n_pruned, history
