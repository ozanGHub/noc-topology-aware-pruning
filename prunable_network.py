from __future__ import annotations
from typing import Dict, Iterable, Optional, Set, Tuple

import networkx as nx
import torch.nn as nn
import torch.nn.utils.prune as prune

from graph import (
    Node,
    construct_nn_graph,
    graph_prune_link,
    graph_prune_neuron,
    graph_prune_neurons,
    filter_lonely_neurons,
    remove_lonely_neurons,
)
from pruning import (
   l1_prune_links_global,
    prune_link,
    prune_neurons,
    select_neurons_high_indegree_low_magnitude,
    save_model,
    load_model,
)
from visualize import visualize_graph


class PrunableNetwork:
    """
    Keeps an ``nn.Module`` and its corresponding ``nx.DiGraph`` in sync.

    Every mutation method updates **both** the model and the graph in a
    single call, so it is impossible to leave the two representations in an
    inconsistent state.  The graph is never exposed directly; callers
    receive a read-only snapshot via the :py:attr:`graph` property to
    prevent silent external mutations from breaking the invariant.

    Typical workflow
    ----------------
    >>> net = PrunableNetwork(model)
    >>> net.prune_links_l1_global(amount=0.25)
    >>> net.prune_link(src=(2, 0), dst=(3, 0))
    >>> removed = net.remove_lonely_neurons()
    >>> net.visualize("After pruning")
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, model: nn.Module, *, eps: float = 0.0) -> None:
        """
        Args:
            model:  A ``torch.nn.Module`` containing at least one
                    ``nn.Linear`` layer.
            eps:    Weights whose absolute value is ≤ ``eps`` are treated
                    as zero when building the graph.  Use ``0.0`` to
                    represent all weights faithfully.
        """
        self._model: nn.Module = model
        self._eps: float = eps
        self._graph: nx.DiGraph = construct_nn_graph(model, eps=eps)

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def model(self) -> nn.Module:
        """The underlying ``nn.Module``.  Treat as read-only."""
        return self._model

    @property
    def graph(self) -> nx.DiGraph:
        """
        A **copy** of the internal graph.

        A copy is returned so that callers cannot accidentally mutate the
        graph without going through the atomic methods on this class.
        """
        return self._graph.copy()

    @property
    def eps(self) -> float:
        return self._eps

    @property
    def linear_names(self) -> list[str]:
        """Names of the ``nn.Linear`` layers, in order."""
        return self._graph.graph["linear_names"]

    @property
    def layers(self) -> list[int]:
        """Width (number of neurons) of each layer, in order."""
        return self._graph.graph["layers"]

    # ------------------------------------------------------------------
    # Link-level operations
    # ------------------------------------------------------------------

    def prune_link(
        self,
        src: Node,
        dst: Node,
        *,
        persistent: bool = True,
    ) -> "PrunableNetwork":
        """
        Zero out (and optionally register as a persistent mask) a single
        weight, then remove the corresponding edge from the graph.

        Args:
            src:        ``(node_layer, idx)`` of the source neuron.
            dst:        ``(node_layer, idx)`` of the destination neuron.
            persistent: If ``True``, register a forward-hook mask so the
                        weight stays zero even after an optimiser step.

        Returns:
            ``self`` — to allow method chaining.
        """
        self._model = prune_link(
            self._model, src=src, dst=dst, persistent=persistent
        )
        graph_prune_link(self._graph, src=src, dst=dst, inplace=True)
        return self

    def prune_links_l1_global(
        self,
        amount: float,
        *,
        persistent: bool = False,
    ) -> "PrunableNetwork":
        """
        Remove the globally weakest ``amount`` fraction of weights by L1
        magnitude, then rebuild the graph from scratch.

        The graph is fully reconstructed rather than patched edge-by-edge
        because L1 pruning can affect a large, unpredictable set of edges.

        Args:
            amount:     Fraction of weights to prune in ``[0, 1)``.
            persistent: Passed through to ``l1_prune_links_global``.

        Returns:
            ``self``
        """
        l1_prune_links_global(
            self._model, amount=amount, inplace=True, persistent=persistent
        )
        self._graph = construct_nn_graph(self._model, eps=self._eps)
        return self

    # ------------------------------------------------------------------
    # Neuron-level operations
    # ------------------------------------------------------------------

    def prune_neuron(self, node: Node) -> "PrunableNetwork":
        """
        Remove a single neuron — and all its incident edges — from both
        the model and the graph.

        Args:
            node:  ``(node_layer, idx)`` of the neuron to remove.

        Returns:
            ``self``
        """
        return self.prune_neurons([node])

    def prune_neurons(self, nodes: Iterable[Node]) -> "PrunableNetwork":
        node_set = set(nodes)
        if not node_set:
            return self

        # Validate first
        graph_prune_neurons(self._graph, node_set, inplace=False, record=False)

        # Apply to model
        self._model = prune_neurons(
            self._model,
            src=sorted(node_set),
            linear_names=self._graph.graph["linear_names"],
        )

        # Rebuild graph from scratch — structural pruning resizes weight matrices
        # so node indices in the old graph no longer match the model
        self._graph = construct_nn_graph(self._model, eps=self._eps)
        return self

    def remove_lonely_neurons(self) -> Set[Node]:
        layers = self._graph.graph["layers"]
        last_layer = len(layers) - 1

        lonely = filter_lonely_neurons(self._graph)

        # Input (node_layer=0) and output (node_layer=last) neurons cannot be
        # structurally removed — filter them out silently.
        prunable = {node for node in lonely if 0 < node[0] < last_layer}

        if prunable:
            self._model = prune_neurons(
                self._model,
                src=sorted(prunable),
                linear_names=self._graph.graph["linear_names"],
            )
            self._graph = construct_nn_graph(self._model, eps=self._eps)

        return lonely  # return the full set so the caller knows what was detected
    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> "PrunableNetwork":
        """
        Persist the model weights to *path*.

        Returns:
            ``self``
        """
        save_model(self._model, path)
        return self

    def load(self, path: str) -> "PrunableNetwork":
        """
        Load model weights from *path* and rebuild the graph.

        Returns:
            ``self``
        """
        self._model = load_model(path)
        self._graph = construct_nn_graph(self._model, eps=self._eps)
        return self

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(self, title: str = "PrunableNetwork", **kwargs) -> "PrunableNetwork":
        """
        Render the current graph via ``visualize_graph``.

        Any extra keyword arguments are forwarded to ``visualize_graph``
        (e.g. ``groups=...``).

        Returns:
            ``self``
        """
        visualize_graph(self._graph, title="NN: " + title, **kwargs)
        return self

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        layers = self.layers
        n_nodes = self._graph.number_of_nodes()
        n_edges = self._graph.number_of_edges()
        return (
            f"PrunableNetwork("
            f"layers={layers}, "
            f"nodes={n_nodes}, "
            f"edges={n_edges})"
        )
        
    def prune_high_indegree_low_magnitude(
        self,
        *,
        low_mag_percentile: float = 25.0,
        degree_thresholds: Optional[Dict[int, int]] = None,
        degree_quantile: Optional[float] = None,
        min_in_degree: int = 1,
    ) -> "PrunableNetwork":
        """
        Remove neurons that have both high in-degree and low mean incoming
        weight magnitude.  Selection is performed on the current graph, then
        dispatched to :py:meth:`prune_neurons` so model and graph stay in sync.

        Args:
            low_mag_percentile: Bottom X% of neurons by mean |incoming weight|
                                are considered low-magnitude.
            degree_thresholds:  Optional per-layer fixed in-degree thresholds.
            degree_quantile:    Optional quantile threshold for in-degree.
                                Mutually exclusive with ``degree_thresholds``.
            min_in_degree:      Neurons with fewer incoming edges than this
                                are excluded from consideration.

        Returns:
            ``self``
        """
        candidates = select_neurons_high_indegree_low_magnitude(
            self._graph,
            low_mag_percentile=low_mag_percentile,
            degree_thresholds=degree_thresholds,
            degree_quantile=degree_quantile,
            min_in_degree=min_in_degree,
        )
        return self.prune_neurons(candidates)
    
    def prune_links_heuristic(
        self,
        groups: list,
        *,
        amount: Optional[float] = None,
        importance_threshold: Optional[float] = None,
        debug: bool = False,
    ) -> "PrunableNetwork":
        """
        Heuristic bridge-link pruning based on L1 importance.

        Steps:
            1. Rank all edges by |weight| ascending (weakest = least important first).
            2. Identify bridge links — edges whose src and dst neurons belong to
            different groups (cross-group communication links).
            3. Traverse the importance list and prune bridge links, subject to the
            constraint that at least one link must remain between every pair of
            groups that currently communicate.

        Stopping criterion (one must be provided):
            amount:               Prune at most this fraction of all edges, e.g. 0.25.
            importance_threshold: Prune all bridge links with |weight| < threshold.

        Args:
            groups:               List of neuron sets as produced by NoCMapper.groups,
                                e.g. [{(1, 0), (1, 1)}, {(1, 2), (2, 0)}, ...].
            amount:               Fraction of total edges to prune in (0, 1).
            importance_threshold: Magnitude below which a bridge link is pruned.
            debug:                Print per-step decisions.

        Returns:
            ``self``
        """
        if amount is None and importance_threshold is None:
            raise ValueError("Provide either amount or importance_threshold.")
        if amount is not None and importance_threshold is not None:
            raise ValueError("Provide either amount or importance_threshold, not both.")
        if amount is not None and not (0.0 < amount < 1.0):
            raise ValueError("amount must be in (0, 1).")

        # Build node -> group_id lookup from the list of sets
        node_to_group: Dict[Node, int] = {
            node: group_id
            for group_id, group in enumerate(groups)
            for node in group
        }

        # 1. Collect and sort all edges by |weight| ascending
        edges = [
            (abs(d["weight"]), u, v)
            for u, v, d in self._graph.edges(data=True)
        ]
        edges.sort(key=lambda x: x[0])

        total_edges = len(edges)
        max_prunable = int(total_edges * amount) if amount is not None else total_edges

        if debug:
            print(f"\n[prune_links_heuristic] Total edges: {total_edges}")
            print(f"  Stopping criterion: "
                f"{'amount=' + str(amount) if amount else 'threshold=' + str(importance_threshold)}")

        # 2. Track how many links remain between each pair of communicating groups.
        group_pair_counts: Dict[Tuple[int, int], int] = {}
        for _, u, v in edges:
            gu = node_to_group.get(u)
            gv = node_to_group.get(v)
            if gu is None or gv is None or gu == gv:
                continue
            key = (min(gu, gv), max(gu, gv))
            group_pair_counts[key] = group_pair_counts.get(key, 0) + 1

        if debug:
            print(f"  Communicating group pairs: {dict(group_pair_counts)}")

        # 3. Traverse and prune
        pruned = 0
        for weight, u, v in edges:
            if pruned >= max_prunable:
                break
            if importance_threshold is not None and weight >= importance_threshold:
                break

            gu = node_to_group.get(u)
            gv = node_to_group.get(v)

            # Skip intra-group links — not candidates
            if gu is None or gv is None or gu == gv:
                if debug:
                    print(f"  SKIP (intra-group) {u} -> {v}  |w|={weight:.4f}")
                continue

            key = (min(gu, gv), max(gu, gv))

            # Guard: must keep at least one link between this group pair
            if group_pair_counts[key] <= 1:
                if debug:
                    print(f"  SKIP (last link between groups {key}) {u} -> {v}  |w|={weight:.4f}")
                continue

            # Safe to prune
            if debug:
                print(f"  PRUNE {u} -> {v}  |w|={weight:.4f}  groups {gu}->{gv}")

            self.prune_link(u, v)
            group_pair_counts[key] -= 1
            pruned += 1

        print(f"[prune_links_heuristic] Pruned {pruned}/{total_edges} bridge links.")
        return self

    def commit_pruning(self) -> "PrunableNetwork":
        """
        Bake all pending pruning masks permanently into the weight tensors
        and rebuild the graph.

        Call this once after iterative prune → fine-tune loops to materialise
        all zeroed weights before visualisation or further structural pruning.

        Returns:
            ``self``
        """
        for mod in self._model.modules():
            if isinstance(mod, nn.Linear) and hasattr(mod, "weight_orig"):
                prune.remove(mod, "weight")
        self._graph = construct_nn_graph(self._model, eps=self._eps)
        return self
    
    def resync_graph(self) -> "PrunableNetwork":
        #Rebuild the graph from the current model weights. Call after any
        #external operation (e.g. fine-tuning) that may have changed weights
        #without going through PrunableNetwork's mutation methods.
        self._graph = construct_nn_graph(self._model, eps=self._eps)
        return self