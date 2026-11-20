from __future__ import annotations

import math
from typing import Any
import networkx as nx

from grouping_heuristic import indegree_grouping
from mapping_helpers import build_wctg, build_mesh, get_spg_matrix, get_wctg_matrix
from mapping_sa import map_nodes

# Real deal as stosun requested below. SA is for quick testing.
# from mapping_ilp import map_nodes

from visualize import visualize_graph, visualize_wctg, visualize_mesh, visualize_mapping


class NoCMapper:
    """
    Encapsulates the full Network-on-Chip mapping pipeline for a pruned NN graph.

    Pipeline stages (each depends on the previous):

        1. **Group**   — bin-pack neurons into at most *N* groups via
                         in-degree heuristic.
        2. **Build**   — construct the Weighted Communication Task Graph
                         (WCTG) from the groups, and a 2-D mesh topology
                         from *P* and the mesh dimensions.
        3. **Map**     — derive cost matrices and run the SA to assign
                         task groups to mesh cores.

    All three stages are triggered together by :py:meth:`run`, which takes
    a graph snapshot from ``PrunableNetwork.graph`` and re-runs the entire
    pipeline atomically.  Intermediate results are stored as properties and
    are reset to ``None`` whenever ``run`` is called again, so stale state
    can never leak into a new mapping.

    Args:
        P:             Number of processing elements (mesh cores).
        N:             Maximum number of task groups.
        mesh_rows:     Rows in the 2-D mesh.  Defaults to ``floor(sqrt(P))``.
        mesh_cols:     Columns in the 2-D mesh.  Defaults to ``floor(sqrt(P))``.
        grouping_cap:  Weight cap passed to the in-degree grouping heuristic.

    Example
    -------
    >>> noc = NoCMapper(P=9, N=6)
    >>> noc.run(net.graph)
    >>> noc.visualize_all()
    >>> print(noc.mapping)
    >>> print(noc.cost)
    """

    def __init__(
        self,
        P: int,
        N: int,
        *,
        mesh_rows: int | None = None,
        mesh_cols: int | None = None,
        grouping_cap: float = 10_000.0,
    ) -> None:
        if P < 1:
            raise ValueError(f"P must be at least 1, got {P}.")
        if N < 1:
            raise ValueError(f"N must be at least 1, got {N}.")

        self._P = P
        self._N = N
        self._mesh_rows = mesh_rows if mesh_rows is not None else int(math.sqrt(P))
        self._mesh_cols = mesh_cols if mesh_cols is not None else int(math.sqrt(P))
        self._grouping_cap = grouping_cap

        self._reset()

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def P(self) -> int:
        return self._P

    @property
    def N(self) -> int:
        return self._N

    @property
    def groups(self) -> list | None:
        """Neuron groups produced by the grouping stage, or ``None`` before ``run``."""
        return self._groups

    @property
    def wctg(self) -> nx.DiGraph | None:
        """Weighted Communication Task Graph, or ``None`` before ``run``."""
        return self._wctg

    @property
    def structural_wctg(self) -> nx.DiGraph | None:
        """
        Structural WCTG whose edge weights are raw connection counts between
        groups (independent of synaptic weight magnitudes), or ``None`` before
        ``run``.  Used to drive arrow thickness / colour in visualize_mapping.
        """
        return self._structural_wctg

    @property
    def mesh(self) -> nx.Graph | None:
        """2-D mesh topology graph, or ``None`` before ``run``."""
        return self._mesh

    @property
    def mapping(self) -> list[tuple[int, int]] | None:
        """
        SA result as a list of ``(task_id, core_id)`` pairs,
        or ``None`` before ``run``.
        """
        return self._result["mapping"] if self._result else None

    @property
    def cost(self) -> float | None:
        """Optimal communication cost from the SA, or ``None`` before ``run``."""
        return self._result["cost"] if self._result else None

    @property
    def structural_cost(self) -> float | None:
        """
        Structural communication cost after mapping:
        Σ (structural_wctg_weight × hop_distance) for every mapped edge pair.
        This uses raw connection counts (not synaptic magnitudes) as weights.
        Returns None before run() is called.
        """
        if self._structural_wctg is None or self._result is None:
            return None

        sorted_nodes     = sorted(self._mesh.nodes())
        core_id_to_node  = {i: node for i, node in enumerate(sorted_nodes)}
        node_to_core_idx = {node: i for i, node in enumerate(sorted_nodes)}
        task_to_node     = {task_id: core_id_to_node[core_id]
                            for task_id, core_id in self._result["mapping"]}
        spg_dict         = self.spg  # builds the dict once

        total = 0.0
        for u, v, data in self._structural_wctg.edges(data=True):
            sw = data.get("weight", 0)
            if sw == 0:
                continue
            cu = task_to_node.get(u)
            cv = task_to_node.get(v)
            if cu is None or cv is None or cu == cv:
                continue
            ci   = node_to_core_idx[cu]
            cj   = node_to_core_idx[cv]
            hops = spg_dict.get((ci, cj), spg_dict.get((cj, ci), 0)) or 0
            total += sw * hops
        return total

    @property
    def result(self) -> dict[str, Any] | None:
        """Full SA result dict, or ``None`` before ``run``."""
        return self._result

    @property
    def spg(self) -> dict | None:
        """
        All-pairs shortest-path distances on the mesh as a dict
        keyed by (core_idx_i, core_idx_j) -> hop_count (int).
        core_idx is the 0-based index into sorted(mesh.nodes()).
        Returns None before run() is called.
        """
        if self._spg_matrix is None or self._spg_index is None:
            return None
        n = len(self._spg_index)
        return {
            (i, j): int(self._spg_matrix[i][j])
            for i in range(n)
            for j in range(n)
        }

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def run(self, graph: nx.DiGraph) -> "NoCMapper":
        """
        Execute the full grouping -> build -> map pipeline on *graph*.

        Any results from a previous ``run`` are discarded before the new
        pipeline starts, so there is no risk of mixing results from
        different graph states.

        Args:
            graph:  A graph snapshot, typically obtained via
                    ``PrunableNetwork.graph``.

        Returns:
            ``self`` — to allow method chaining.
        """
        self._reset()
        self._group(graph)
        self._build(graph)
        self._map()
        return self

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize_groups(self, *, title: str = "After grouping") -> "NoCMapper":
        """Render the NN graph with group colouring."""
        self._require("groups")
        visualize_graph(self._graph_snapshot, groups=self._groups, title=title)
        return self

    def visualize_wctg(self, *, title: str = "WCTG") -> "NoCMapper":
        """Render the Weighted Communication Task Graph."""
        self._require("wctg")
        visualize_wctg(self._wctg, self._groups, title=title)
        return self

    def visualize_mesh(self, title: str = "Mesh") -> "NoCMapper":
        """Render the 2-D mesh topology."""
        self._require("mesh")
        visualize_mesh(self._mesh, title=title)
        return self

    def visualize_mapping(self, *, title: str = "ILP Task Mapping",
                          show_wctg_overlay: bool = False,
                          metrics: dict = None) -> "NoCMapper":
        """Render the mapping with cost contributions (weight x hops) on arrows.

        Args:
            metrics: optional dict with keys "baseline" and "current", each
                     containing accuracy, cost, edges, neurons, batch_ms.
                     Example:
                         noc.visualize_mapping(metrics={
                             "baseline": {"accuracy": 0.98, "cost": 2000,
                                          "edges": 5000, "neurons": 800,
                                          "batch_ms": 1.2},
                             "current":  {"accuracy": 0.97, "cost": 1400,
                                          "edges": 3200, "neurons": 620,
                                          "batch_ms": 0.9},
                         })
        """
        self._require("result")
        visualize_mapping(
            self._mesh,
            self._result["mapping"],
            title=title,
            groups=self._groups,
            wctg=self._wctg,
            spg=self.spg,
            total_cost=self.cost,
            metrics=metrics,
            structural_wctg=self._structural_wctg,
            show_wctg_overlay=show_wctg_overlay,
        )
        return self

    def visualize_all(self, title, metrics: dict = None, **kwargs) -> "NoCMapper":
        """Run all three visualisations in pipeline order.

        Any extra keyword arguments (e.g. ``structural_wctg``) are accepted
        but ignored — structural_wctg is built internally and forwarded
        automatically from ``self._structural_wctg``.
        """
        return (
            #self.visualize_groups(title="Groups: " + title)
            self.visualize_wctg(title="WCTG: " + title)
                .visualize_mapping(title="Mapping: " + title, metrics=metrics, show_wctg_overlay=False)
            #self.visualize_mapping(title="Mapping: " + title, metrics=metrics, show_wctg_overlay=True)
        )



    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        ran = self._result is not None
        return (
            f"NoCMapper("
            f"P={self._P}, N={self._N}, "
            f"mesh={self._mesh_rows}x{self._mesh_cols}, "
            f"ran={ran})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        """Discard all pipeline state."""
        self._graph_snapshot: nx.DiGraph | None = None
        self._groups: list | None = None
        self._wctg: nx.DiGraph | None = None
        self._structural_wctg: nx.DiGraph | None = None
        self._mesh: nx.Graph | None = None
        self._result: dict[str, Any] | None = None
        self._spg_matrix = None   # saved from get_spg_matrix() for spg property
        self._spg_index  = None   # node ordering for SPG matrix rows/cols

    def _group(self, graph: nx.DiGraph) -> None:
        self._graph_snapshot = graph
        self._groups = indegree_grouping(graph, self._N, self._grouping_cap)

    def _build(self, graph: nx.DiGraph) -> None:
        self._wctg = build_wctg(graph, self._groups)
        self._mesh = build_mesh(self._mesh_rows, self._mesh_cols)

        # Build structural WCTG: edge weight = number of directed connections
        # between groups (raw edge count, independent of weight magnitudes).
        node_to_group: dict = {}
        for gid, members in enumerate(self._groups):
            for node in members:
                node_to_group[node] = gid

        swctg: nx.DiGraph = nx.DiGraph()
        swctg.add_nodes_from(range(len(self._groups)))
        for u, v in graph.edges():
            gu = node_to_group.get(u)
            gv = node_to_group.get(v)
            if gu is None or gv is None or gu == gv:
                continue
            if swctg.has_edge(gu, gv):
                swctg[gu][gv]["weight"] += 1
            else:
                swctg.add_edge(gu, gv, weight=1)
        # Copy node attributes (e.g. 'layer') from the regular WCTG so
        # multipartite layout works for both graphs.
        for n, attrs in self._wctg.nodes(data=True):
            if n in swctg.nodes:
                swctg.nodes[n].update(attrs)
        self._structural_wctg = swctg

    def _map(self) -> None:
        spg_matrix, spg_index  = get_spg_matrix(self._mesh)
        wctg_matrix, _         = get_wctg_matrix(self._wctg)
        self._spg_matrix       = spg_matrix   # save for spg property
        self._spg_index        = spg_index
        self._result           = map_nodes(wctg_matrix, spg_matrix, self._P)

    def _require(self, stage: str) -> None:
        """Raise a clear error if the pipeline has not been run yet."""
        sentinel = getattr(self, f"_{stage}")
        if sentinel is None:
            raise RuntimeError(
                f"'{stage}' is not available. Call run(graph) first."
            )
