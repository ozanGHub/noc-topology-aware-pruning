"""
prunable_network_conv.py
========================
PrunableNetworkConv — a drop-in replacement for PrunableNetwork that
supports mixed Conv2d + Linear architectures.

Key differences from PrunableNetwork
--------------------------------------
1. Uses graph_conv.construct_nn_graph_conv instead of graph.construct_nn_graph.
   Graph nodes represent output *channels* (Conv2d) or *neurons* (Linear) at
   channel granularity throughout.

2. prune_neurons / remove_lonely_neurons translate conv-graph node_layers
   to the linear-only indices that pruning.prune_neurons expects, and silently
   skip Conv2d channel nodes (physical Conv2d channel removal is not supported
   — conv channels contribute to the NoC graph but are left structurally intact).

3. commit_pruning also removes pending prune masks on Conv2d layers.

Everything else (prune_links_l1_global, prune_link, save, load, visualize,
resync_graph, commit_pruning) is inherited or adapted from PrunableNetwork.
"""

from __future__ import annotations
from typing import Dict, Iterable, Optional, Set, Tuple

import networkx as nx
import torch.nn as nn
import torch.nn.utils.prune as prune

from graph_conv import construct_nn_graph_conv
from graph import (
    Node,
    graph_prune_link,
    filter_lonely_neurons,
)
from pruning import (
    l1_prune_links_global,
    prune_link,
    prune_neurons,
    save_model,
    load_model,
)
from visualize import visualize_graph


class PrunableNetworkConv:
    """
    Keeps an nn.Module (Conv2d + Linear) and its channel-level nx.DiGraph in sync.

    Neuron pruning only applies to Linear-layer nodes; Conv2d channel nodes
    participate in the graph / NoC cost model but are never physically removed.
    """

    def __init__(self, model: nn.Module, *, eps: float = 0.0) -> None:
        self._model: nn.Module  = model
        self._eps:   float      = eps
        self._graph: nx.DiGraph = construct_nn_graph_conv(model, eps=eps)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph.copy()

    @property
    def eps(self) -> float:
        return self._eps

    @property
    def linear_names(self) -> list[str]:
        return self._graph.graph["linear_names"]

    @property
    def layers(self) -> list[int]:
        return self._graph.graph["layers"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_conv_node(self, node: Node) -> bool:
        """True if this graph node is a Conv2d output channel."""
        layer_types = self._graph.graph.get("layer_types", [])
        nl = node[0]
        if nl == 0 or nl > len(layer_types):
            return False
        return layer_types[nl - 1] == "conv"

    def _translate_linear_nodes(
        self, nodes: Iterable[Node]
    ) -> list[Node]:
        """
        Convert conv-graph node_layers to the linear-only indices that
        pruning.prune_neurons expects.

        pruning.prune_neurons interprets node_layer k as:
          prune_layer  = linear_names[k-1]
          next_layer   = linear_names[k]
        so we map conv-graph node_layer → (linear_idx + 1).

        Conv2d channel nodes and input/output nodes are silently dropped.
        """
        nl_to_li = self._graph.graph.get("node_layer_to_linear_idx", {})
        n_linears = len(self.linear_names)
        translated = []
        for (nl, idx) in nodes:
            if nl not in nl_to_li:
                continue                     # conv node or input/output — skip
            li = nl_to_li[nl]
            translated_nl = li + 1
            # prunable range: [1, n_linears - 1]  (output layer excluded)
            if 1 <= translated_nl <= n_linears - 1:
                translated.append((translated_nl, idx))
        return translated

    # ------------------------------------------------------------------
    # Link operations
    # ------------------------------------------------------------------

    def prune_link(self, src: Node, dst: Node, *, persistent: bool = True):
        self._model = prune_link(self._model, src=src, dst=dst, persistent=persistent)
        graph_prune_link(self._graph, src=src, dst=dst, inplace=True)
        return self

    def prune_links_l1_global(self, amount: float, *, persistent: bool = False):
        l1_prune_links_global(
            self._model, amount=amount, inplace=True, persistent=persistent
        )
        self._graph = construct_nn_graph_conv(self._model, eps=self._eps)
        return self

    # ------------------------------------------------------------------
    # Neuron operations
    # ------------------------------------------------------------------

    def prune_neuron(self, node: Node):
        return self.prune_neurons([node])

    def prune_neurons(self, nodes: Iterable[Node]):
        node_set = set(nodes)
        if not node_set:
            return self

        # Separate conv nodes (skip) from linear nodes (translate + prune)
        linear_nodes = [n for n in node_set if not self._is_conv_node(n)]
        if not linear_nodes:
            return self

        translated = self._translate_linear_nodes(linear_nodes)
        if not translated:
            return self

        self._model = prune_neurons(
            self._model,
            src=sorted(translated),
            linear_names=self.linear_names,
        )
        self._graph = construct_nn_graph_conv(self._model, eps=self._eps)
        return self

    def remove_lonely_neurons(self) -> Set[Node]:
        layers    = self._graph.graph["layers"]
        last_layer = len(layers) - 1
        lonely    = filter_lonely_neurons(self._graph)

        # Only prune linear nodes that are neither input nor output
        prunable = {
            node for node in lonely
            if 0 < node[0] < last_layer and not self._is_conv_node(node)
        }

        if prunable:
            translated = self._translate_linear_nodes(prunable)
            if translated:
                self._model = prune_neurons(
                    self._model,
                    src=sorted(translated),
                    linear_names=self.linear_names,
                )
                self._graph = construct_nn_graph_conv(self._model, eps=self._eps)

        return lonely

    # ------------------------------------------------------------------
    # Graph sync
    # ------------------------------------------------------------------

    def commit_pruning(self):
        """Bake all pending prune masks (Linear and Conv2d) and rebuild graph."""
        for mod in self._model.modules():
            if isinstance(mod, (nn.Linear, nn.Conv2d)) and hasattr(mod, "weight_orig"):
                prune.remove(mod, "weight")
        self._graph = construct_nn_graph_conv(self._model, eps=self._eps)
        return self

    def resync_graph(self):
        self._graph = construct_nn_graph_conv(self._model, eps=self._eps)
        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        save_model(self._model, path)
        return self

    def load(self, path: str):
        self._model = load_model(path)
        self._graph = construct_nn_graph_conv(self._model, eps=self._eps)
        return self

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(self, title: str = "PrunableNetworkConv", **kwargs):
        visualize_graph(self._graph, title="CNN: " + title, **kwargs)
        return self

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PrunableNetworkConv("
            f"layers={self.layers}, "
            f"nodes={self._graph.number_of_nodes()}, "
            f"edges={self._graph.number_of_edges()})"
        )
