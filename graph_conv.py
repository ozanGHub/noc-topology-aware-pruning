"""
graph_conv.py
=============
Graph construction for mixed Conv2d + Linear networks.

Extends graph.py's construct_nn_graph to handle convolutional layers by
treating each output channel as a graph node. Edge weight between channel
i (source) and channel/neuron j (destination) is the L1 norm of the
connecting kernel or weight:

  Conv2d  :  w(i→j) = ||W[j, i, :, :]||_1   (sum of absolute kernel values)
  Linear  :  w(i→j) = ||W[j, i*sp : (i+1)*sp]||_1
             where sp = in_features // n_source_channels
             (= 1 for linear-after-linear, = spatial_size for linear-after-conv)

This keeps the graph at channel granularity throughout, which matches the
NoC cost model: one channel = one logical communication endpoint.

Graph metadata stored in G.graph
---------------------------------
  "layer_names"            : list[str] — module names (Conv2d + Linear) in order
  "layer_types"            : list[str] — 'conv' or 'linear' per transition
  "layers"                 : list[int] — node layer sizes (input + each output)
  "spatial_sizes"          : list[int] — spatial block size per transition
  "linear_names"           : list[str] — names of Linear layers only
  "node_layer_to_linear_idx": dict[int, int] — graph node_layer → 0-based linear index
"""

from __future__ import annotations
import os
from typing import Tuple

import networkx as nx
import numpy as np
import torch.nn as nn


Node = Tuple[int, int]   # (node_layer, channel_or_neuron_idx)


# ----------------------------------------------------------------------
# Speed switch (default OFF -> original behaviour).  Same semantics as in graph.py.
#   GRAPH_FAST=1   per-channel-pair weights via one tensor reduction per layer and the
#                  adjacency dicts filled directly (instead of a Python loop with a
#                  .item() + add_edge per channel pair).  Same nodes, edge insertion
#                  order and edge weights as the original.
#   GRAPH_CHECK=1  (verification only, slow; implies GRAPH_FAST) also builds the original
#                  graph on every call and asserts: identical nodes+attrs, identical edges
#                  and in-edges in the same order, identical G.graph, and edge weights
#                  equal (reports BIT-IDENTICAL, or the max relative difference if the
#                  float32 reduction order differs; asserts <= 1e-6).
# ----------------------------------------------------------------------
GRAPH_CHECK: bool = os.environ.get("GRAPH_CHECK") == "1"
GRAPH_FAST:  bool = GRAPH_CHECK or os.environ.get("GRAPH_FAST") == "1"
_graph_check_calls = 0


def _fill_edges_direct(G: nx.DiGraph, layer_idx: int, M: "np.ndarray", eps: float) -> None:
    """Add edges src=(layer_idx, in_i) -> dst=(layer_idx+1, out_j) for M[out_j, in_i] > eps,
    in the original insertion order (out_j outer, in_i inner), filling networkx's adjacency
    dicts directly (one shared attribute dict per edge, like add_edge)."""
    rr, cc = np.nonzero(M > eps)                       # row-major = original loop order
    ws = M[rr, cc].tolist()
    rl, cl = rr.tolist(), cc.tolist()
    if not (hasattr(G, "_succ") and hasattr(G, "_pred")):
        G.add_edges_from(((layer_idx, c), (layer_idx + 1, r), {"weight": w})
                         for r, c, w in zip(rl, cl, ws))
        return
    datas    = [{"weight": w} for w in ws]
    src_keys = [(layer_idx, j) for j in range(M.shape[1])]
    dst_keys = [(layer_idx + 1, i) for i in range(M.shape[0])]
    succ, pred = G._succ, G._pred
    starts = np.searchsorted(rr, np.arange(M.shape[0] + 1)).tolist()
    for r in range(M.shape[0]):                        # predecessors: ascending in_i
        a, b = starts[r], starts[r + 1]
        if a != b:
            pred[dst_keys[r]].update(zip([src_keys[c] for c in cl[a:b]], datas[a:b]))
    order  = np.argsort(cc, kind="stable")
    starts = np.searchsorted(cc[order], np.arange(M.shape[1] + 1)).tolist()
    ol     = order.tolist()
    for c in range(M.shape[1]):                        # successors: ascending out_j
        a, b = starts[c], starts[c + 1]
        if a != b:
            idx = ol[a:b]
            succ[src_keys[c]].update(zip([dst_keys[rl[k]] for k in idx],
                                         [datas[k] for k in idx]))


def construct_nn_graph_conv(model: nn.Module, *, eps: float = 0.0) -> nx.DiGraph:
    global _graph_check_calls
    if not GRAPH_FAST:
        return _construct_nn_graph_conv_impl(model, eps=eps, fast=False)
    fast = _construct_nn_graph_conv_impl(model, eps=eps, fast=True)
    if GRAPH_CHECK:
        slow = _construct_nn_graph_conv_impl(model, eps=eps, fast=False)
        fe, se = list(fast.edges(data=True)), list(slow.edges(data=True))
        structure = (list(fast.nodes(data=True)) == list(slow.nodes(data=True))
                     and [(u, v) for u, v, _ in fe] == [(u, v) for u, v, _ in se]
                     and [(u, v) for u, v, _ in fast.in_edges(data=True)]
                         == [(u, v) for u, v, _ in slow.in_edges(data=True)]
                     and fast.graph == slow.graph)
        fw = np.array([d["weight"] for *_, d in fe], dtype=np.float64)
        sw = np.array([d["weight"] for *_, d in se], dtype=np.float64)
        bit = bool(structure and np.array_equal(fw, sw))
        rel = float(np.max(np.abs(fw - sw) / np.maximum(np.abs(sw), 1e-30))) if structure and fw.size else float("nan")
        ok = structure and (bit or rel <= 1e-6)
        _graph_check_calls += 1
        if _graph_check_calls <= 3 or _graph_check_calls % 25 == 0 or not ok or (structure and not bit):
            print(f"[graph-check] call {_graph_check_calls}: {fast.number_of_nodes():,} nodes, "
                  f"{fast.number_of_edges():,} edges -> structure "
                  f"{'IDENTICAL' if structure else 'MISMATCH'}, weights "
                  f"{'BIT-IDENTICAL' if bit else f'max rel diff {rel:.2e}'}")
        assert ok, "vectorised conv graph differs from the original construct_nn_graph_conv"
    return fast


def _construct_nn_graph_conv_impl(model: nn.Module, *, eps: float = 0.0,
                                  fast: bool = False) -> nx.DiGraph:
    """
    Build a channel-level directed graph for a CNN with Conv2d and Linear layers.

    All Conv2d and Linear modules encountered in named_modules() order are
    treated as "transition" layers; pooling, activation, BatchNorm and other
    modules are ignored (they don't create new graph nodes).

    Args:
        model : nn.Module with at least one Conv2d or Linear layer.
        eps   : edges with |weight| <= eps are omitted.

    Returns:
        nx.DiGraph with node tuples (node_layer, idx) and edge attribute "weight".
    """
    ordered: list[tuple[str, nn.Module]] = [
        (name, m) for name, m in model.named_modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    if not ordered:
        raise ValueError("No Conv2d or Linear layers found in model.")

    # --- Determine input size (channel count) ---
    first_m = ordered[0][1]
    input_size = first_m.in_channels if isinstance(first_m, nn.Conv2d) else first_m.in_features

    # --- Walk layers to compute sizes, types, and spatial_sizes ---
    layer_sizes   = [input_size]   # index 0 = input
    layer_types   = []             # one per transition (len = len(ordered))
    layer_names   = []
    spatial_sizes = []             # one per transition

    prev_is_conv  = isinstance(first_m, nn.Conv2d)
    prev_out      = (first_m.out_channels if prev_is_conv else first_m.out_features)

    for name, m in ordered:
        layer_names.append(name)
        if isinstance(m, nn.Conv2d):
            layer_sizes.append(m.out_channels)
            layer_types.append("conv")
            spatial_sizes.append(1)            # kernel spatial folded at edge weight level
            prev_is_conv = True
            prev_out     = m.out_channels
        else:  # Linear
            layer_sizes.append(m.out_features)
            layer_types.append("linear")
            # spatial block: how many input weights per source channel
            sp = (m.in_features // prev_out) if prev_is_conv else 1
            spatial_sizes.append(sp)
            prev_is_conv = False
            prev_out     = m.out_features

    # --- Build graph ---
    G = nx.DiGraph()

    for node_layer, size in enumerate(layer_sizes):
        for j in range(size):
            G.add_node((node_layer, j), node_layer=node_layer, idx=j)

    for layer_idx, (name, m) in enumerate(ordered):
        src_size = layer_sizes[layer_idx]          # number of source nodes
        sp       = spatial_sizes[layer_idx]

        if fast:
            Wt = m.weight.detach().cpu()
            if isinstance(m, nn.Conv2d):                       # [C_out, C_in, kH, kW]
                S = Wt.abs().sum(dim=(2, 3))
            else:                                              # [out, in] -> blocks of sp
                S = Wt.abs().reshape(Wt.shape[0], src_size, sp).sum(dim=2)
            _fill_edges_direct(G, layer_idx, S.numpy().astype(np.float64), eps)
            continue

        if isinstance(m, nn.Conv2d):
            W = m.weight.detach().cpu()            # [C_out, C_in, kH, kW]
            for out_j in range(W.shape[0]):
                for in_i in range(W.shape[1]):
                    w_scalar = float(W[out_j, in_i].abs().sum().item())
                    if w_scalar > eps:
                        G.add_edge((layer_idx, in_i), (layer_idx + 1, out_j),
                                   weight=w_scalar)
        else:  # Linear
            W = m.weight.detach().cpu()            # [out, in]
            for out_j in range(W.shape[0]):
                for in_i in range(src_size):
                    w_block  = W[out_j, in_i * sp : (in_i + 1) * sp]
                    w_scalar = float(w_block.abs().sum().item())
                    if w_scalar > eps:
                        G.add_edge((layer_idx, in_i), (layer_idx + 1, out_j),
                                   weight=w_scalar)

    # --- Metadata ---
    linear_names = [name for name, m in ordered if isinstance(m, nn.Linear)]

    # Map from graph node_layer (1-based output layer) to 0-based linear index
    node_layer_to_linear_idx: dict[int, int] = {}
    lin_count = 0
    for layer_idx, (name, m) in enumerate(ordered):
        if isinstance(m, nn.Linear):
            node_layer_to_linear_idx[layer_idx + 1] = lin_count
            lin_count += 1

    G.graph["layer_names"]             = layer_names
    G.graph["layer_types"]             = layer_types
    G.graph["layers"]                  = layer_sizes
    G.graph["spatial_sizes"]           = spatial_sizes
    G.graph["linear_names"]            = linear_names
    G.graph["node_layer_to_linear_idx"] = node_layer_to_linear_idx

    return G
