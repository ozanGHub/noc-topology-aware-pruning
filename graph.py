import os
from typing import Iterable, Set, Tuple

import networkx as nx
import numpy as np
import torch.nn as nn


Node = Tuple[int, int]  # (node_layer, idx)

# ----------------------------------------------------------------------
# Speed switch (default OFF -> original behaviour).
#   GRAPH_FAST=1   build the graph with numpy (np.nonzero + add_edges_from) instead of a
#                  Python loop over every weight.  Same nodes, node attributes, edge
#                  insertion order and (signed) edge weights as the original builder.
#   GRAPH_CHECK=1  (verification only, slow; implies GRAPH_FAST) also runs the original
#                  builder on every call and asserts that both graphs are identical
#                  (nodes+attrs, edges+weights in order, in-edges in order, G.graph).
# ----------------------------------------------------------------------
GRAPH_CHECK: bool = os.environ.get("GRAPH_CHECK") == "1"
GRAPH_FAST:  bool = GRAPH_CHECK or os.environ.get("GRAPH_FAST") == "1"
_graph_check_calls = 0


def _construct_nn_graph_slow(model: nn.Module, *, eps: float = 1e-12) -> nx.DiGraph:
    linear_names, linears = [], []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            linear_names.append(name)
            linears.append(m)
    if not linears:
        raise ValueError("No nn.Linear layers found in model.")

    layers = [linears[0].in_features] + [l.out_features for l in linears]
    G = nx.DiGraph()

    # nodes: (node_layer, idx)
    for node_layer, sz in enumerate(layers):
        for j in range(sz):
            G.add_node((node_layer, j), node_layer=node_layer, idx=j)

    # edges
    for linear_idx, layer in enumerate(linears):
        W = layer.weight.detach().cpu().numpy()  # [out, in]
        for out_i in range(W.shape[0]):
            for in_j in range(W.shape[1]):
                w = float(W[out_i, in_j])
                if abs(w) > eps:
                    src = (linear_idx, in_j)
                    dst = (linear_idx + 1, out_i)
                    G.add_edge(src, dst, weight=w)

    G.graph["linear_names"] = linear_names
    G.graph["layers"] = layers
    G.graph["name_to_linear_idx"] = {n: i for i, n in enumerate(linear_names)}
    return G


def _construct_nn_graph_fast(model: nn.Module, *, eps: float = 1e-12) -> nx.DiGraph:
    linear_names, linears = [], []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            linear_names.append(name)
            linears.append(m)
    if not linears:
        raise ValueError("No nn.Linear layers found in model.")

    layers = [linears[0].in_features] + [l.out_features for l in linears]
    G = nx.DiGraph()

    # nodes: (node_layer, idx) -- same insertion order and attributes as the original
    G.add_nodes_from(((node_layer, j), {"node_layer": node_layer, "idx": j})
                     for node_layer, sz in enumerate(layers) for j in range(sz))

    # edges: np.nonzero is row-major = (out_i outer, in_j inner), the original loop order.
    # networkx's per-edge add_edge is the real cost (~2.3x slower), so the adjacency dicts
    # are filled directly, in the same order the original loop would insert (successors of
    # a source: out_i ascending; predecessors of a target: in_j ascending), sharing one
    # attribute dict per edge exactly like add_edge does.  Falls back to add_edges_from if
    # the networkx internals are not available.
    direct = hasattr(G, "_succ") and hasattr(G, "_pred")
    for linear_idx, layer in enumerate(linears):
        W  = layer.weight.detach().cpu().numpy().astype(np.float64)   # [out, in]
        rr, cc = np.nonzero(np.abs(W) > eps)
        ws = W[rr, cc].tolist()
        rl, cl = rr.tolist(), cc.tolist()
        if not direct:
            G.add_edges_from(((linear_idx, c), (linear_idx + 1, r), {"weight": w})
                             for r, c, w in zip(rl, cl, ws))
            continue
        datas    = [{"weight": w} for w in ws]
        src_keys = [(linear_idx, j) for j in range(W.shape[1])]
        dst_keys = [(linear_idx + 1, i) for i in range(W.shape[0])]
        succ, pred = G._succ, G._pred
        # predecessors: per target row, sources in encounter (ascending in_j) order
        starts = np.searchsorted(rr, np.arange(W.shape[0] + 1)).tolist()
        for r in range(W.shape[0]):
            a, b = starts[r], starts[r + 1]
            if a != b:
                pred[dst_keys[r]].update(zip([src_keys[c] for c in cl[a:b]], datas[a:b]))
        # successors: per source column, targets in ascending out_i order (stable sort)
        order  = np.argsort(cc, kind="stable")
        starts = np.searchsorted(cc[order], np.arange(W.shape[1] + 1)).tolist()
        ol     = order.tolist()
        for c in range(W.shape[1]):
            a, b = starts[c], starts[c + 1]
            if a != b:
                idx = ol[a:b]
                succ[src_keys[c]].update(zip([dst_keys[rl[k]] for k in idx],
                                             [datas[k] for k in idx]))

    G.graph["linear_names"] = linear_names
    G.graph["layers"] = layers
    G.graph["name_to_linear_idx"] = {n: i for i, n in enumerate(linear_names)}
    return G


def construct_nn_graph(model: nn.Module, *, eps: float = 1e-12) -> nx.DiGraph:
    global _graph_check_calls
    if not GRAPH_FAST:
        return _construct_nn_graph_slow(model, eps=eps)
    fast = _construct_nn_graph_fast(model, eps=eps)
    if GRAPH_CHECK:
        slow = _construct_nn_graph_slow(model, eps=eps)
        ok = (list(fast.nodes(data=True)) == list(slow.nodes(data=True))
              and list(fast.edges(data=True)) == list(slow.edges(data=True))
              and list(fast.in_edges(data=True)) == list(slow.in_edges(data=True))
              and fast.graph == slow.graph)
        _graph_check_calls += 1
        if _graph_check_calls <= 3 or _graph_check_calls % 25 == 0 or not ok:
            print(f"[graph-check] call {_graph_check_calls}: {fast.number_of_nodes():,} nodes, "
                  f"{fast.number_of_edges():,} edges -> {'IDENTICAL' if ok else 'MISMATCH'}")
        assert ok, "vectorised graph differs from the original construct_nn_graph"
    return fast


def graph_prune_link(
    G: nx.DiGraph,
    *,
    src: tuple[int, int],   # (node_layer, idx)
    dst: tuple[int, int],   # (node_layer, idx)
    inplace: bool = False,
    record: bool = True,
) -> nx.DiGraph:
    """
    Prune (remove) one edge from the graph only.

    - Nodes are (node_layer, idx)
    - Enforces k -> k+1 topology
    - Optionally records the prune in G.graph["pruned_edges"] so it can be applied to the model later
      even if the edge didn't exist at the time of pruning.

    Returns:
      updated graph (same object if inplace=True, else a copy)
    """
    if "linear_names" not in G.graph:
        raise ValueError("Graph missing metadata. Build using construct_nn_graph_tuple().")

    (src_layer, _) = src
    (dst_layer, _) = dst
    if dst_layer != src_layer + 1:
        raise ValueError(f"Edges must go k->k+1. Got {src_layer}->{dst_layer}.")

    H = G if inplace else G.copy()

    # Remove if present (edge becomes truly missing)
    if H.has_edge(src, dst):
        H.remove_edge(src, dst)

    # Record prune intent (so we can apply later to the model)
    if record:
        pruned = H.graph.get("pruned_edges", set())
        # store as tuple of tuples; ensure hashable
        pruned.add((src, dst))
        H.graph["pruned_edges"] = pruned

    return H


def filter_lonely_neurons(G: nx.DiGraph) -> Set[Node]:
    """
    Identify 'lonely' neurons in a sequential NN graph.

    Lonely = union of:
      1) Nodes in input or hidden layers (layer < last) with NO outgoing edges (out_degree == 0)
      2) Nodes in output layer (layer == last) with NO incoming edges (in_degree == 0)

    Returns:
        A set of nodes (node_layer, idx) to remove.
    """
    if "layers" not in G.graph:
        raise ValueError("G.graph['layers'] missing. Construct graph with construct_nn_graph first.")

    layers = G.graph["layers"]
    if not layers:
        return set()

    last_layer = len(layers) - 1
    lonely: Set[Node] = set()

    for (node_layer, idx) in G.nodes:
        if node_layer < last_layer:
            if G.out_degree((node_layer, idx)) == 0:
                lonely.add((node_layer, idx))
        else:  # output layer
            if G.in_degree((node_layer, idx)) == 0:
                lonely.add((node_layer, idx))

    return lonely


def remove_lonely_neurons(G: nx.DiGraph, *, inplace: bool = False) -> nx.DiGraph:
    """
    Remove lonely neurons as defined in filter_lonely_neurons().

    Args:
        G: directed graph produced by construct_nn_graph
        inplace: if True, mutate G; else operate on a copy

    Returns:
        Graph with lonely nodes removed.
    """
    H = G if inplace else G.copy()
    to_remove = filter_lonely_neurons(H)
    H.remove_nodes_from(to_remove)
    return H

def graph_prune_neuron(
    G: nx.DiGraph,
    node: Node,
    *,
    inplace: bool = False,
    record: bool = True,
) -> nx.DiGraph:
    """
    Prune a single neuron (and all its incident edges) from the graph.

    Args:
        G:        Directed graph produced by construct_nn_graph.
        node:     The neuron to remove, as a (node_layer, idx) tuple.
        inplace:  If True, mutate G in place; otherwise operate on a copy.
        record:   If True, append the node to G.graph["pruned_neurons"] so the
                  prune can be replayed against the original model later.

    Returns:
        Updated graph (same object if inplace=True, else a copy).

    Raises:
        ValueError: If the graph is missing required metadata or the node layer
                    is out of range.
        KeyError:   If the node does not exist in the graph.
    """
    if "layers" not in G.graph:
        raise ValueError("Graph missing metadata. Build using construct_nn_graph().")

    node_layer, idx = node
    n_layers = len(G.graph["layers"])

    if not (0 <= node_layer < n_layers):
        raise ValueError(
            f"node_layer {node_layer} is out of range for a graph with {n_layers} layers."
        )

    if node not in G:
        raise KeyError(f"Node {node} does not exist in the graph.")

    H = G if inplace else G.copy()

    H.remove_node(node)  # networkx removes all incident edges automatically

    if record:
        pruned = H.graph.get("pruned_neurons", set())
        pruned.add(node)
        H.graph["pruned_neurons"] = pruned

    return H

def graph_prune_neurons(
    G: nx.DiGraph,
    nodes: Iterable[Node],
    *,
    inplace: bool = False,
    record: bool = True,
) -> nx.DiGraph:
    """
    Prune multiple neurons (and all their incident edges) from the graph.

    This is preferred over calling graph_prune_neuron() in a loop because it
    validates and removes all nodes in a single pass, avoiding redundant copies.

    Args:
        G:        Directed graph produced by construct_nn_graph.
        nodes:    Iterable of (node_layer, idx) tuples to remove.
        inplace:  If True, mutate G in place; otherwise operate on a copy.
        record:   If True, extend G.graph["pruned_neurons"] with the removed
                  nodes so the prune can be replayed against the original model.

    Returns:
        Updated graph (same object if inplace=True, else a copy).

    Raises:
        ValueError: If the graph is missing required metadata, or any node has
                    an out-of-range layer index.
        KeyError:   If any of the requested nodes do not exist in the graph.
    """
    if "layers" not in G.graph:
        raise ValueError("Graph missing metadata. Build using construct_nn_graph().")

    node_set = set(nodes)  # deduplicate and materialise the iterable once
    if not node_set:
        return G if inplace else G.copy()

    n_layers = len(G.graph["layers"])

    # Validate everything before touching the graph so we never apply a
    # partial prune when the input contains bad nodes.
    missing, out_of_range = [], []
    for node in node_set:
        node_layer, _ = node
        if not (0 <= node_layer < n_layers):
            out_of_range.append(node)
        elif node not in G:
            missing.append(node)

    if out_of_range:
        raise ValueError(
            f"The following nodes have an out-of-range layer index "
            f"(graph has {n_layers} layers): {out_of_range}"
        )
    if missing:
        raise KeyError(f"The following nodes do not exist in the graph: {missing}")

    H = G if inplace else G.copy()

    H.remove_nodes_from(node_set)  # networkx removes all incident edges automatically

    if record:
        pruned = H.graph.get("pruned_neurons", set())
        pruned.update(node_set)
        H.graph["pruned_neurons"] = pruned

    return H