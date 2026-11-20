"""
pruning_meta.py
===============
Metaheuristic neuron pruning strategies for NoC-aware DNN compression.

All public step functions share the same call signature as the heuristic
step functions in the main pipeline:

    fn(net, *, max_neurons, **kwargs) -> (n_removed, history)

This makes them drop-in replacements for ``iterative_commcost_neuron_pruning``
and the Taylor/L1 wrappers — no changes needed in the branch loop.

Implemented
-----------
  ga_neuron_step
      Plain GA: proxy fitness = cross-group edge weight sum after removal.

  ga_neuron_step_biased           (Branch O)
      Biased GA: proxy fitness is a weighted combination of
        (a) cross-group cost after removal  (like plain GA), and
        (b) the IterCC score of the chosen neurons  (domain knowledge).
      This encodes the heuristic's insight — neurons that individually
      contribute most to cross-group traffic are good candidates — while
      letting the GA optimise *which batch* to remove jointly.

  ga_neuron_step_guided_mutation  (Branch P)
      Guided-mutation GA: the mutation operator is replaced with an
      IterCC-aware operator that always swaps a gene for the highest-
      IterCC-scoring unused candidate rather than a random one.
      Additionally the initial population is warm-started from IterCC's
      greedy ranking so evolution begins near the heuristic's answer.

  ga_neuron_step_memetic          (Branch U)
      Memetic GA: after crossover + mutation creates each offspring, a
      Lamarckian local search refines the chromosome before it enters the
      next generation.  The local search uses IterCC scores to guide which
      candidates to try, and the proxy fitness to confirm each swap.
      This injects domain knowledge at the individual-refinement level,
      orthogonal to biased fitness (selection) and guided mutation (generation).

  ga_neuron_step_memetic_biased   (Branches V / W)
      Memetic + Biased GA: combines Branch U's Lamarckian local search with
      Branch Q's biased fitness.  The local search oracle is the biased
      fitness (α × proxy_norm − β × itercc_norm) rather than the pure proxy.
      Because the biased landscape has a different basin structure from the
      pure proxy, local search converges each individual to a different local
      optimum — preserving population diversity that Branch U lost.

Planned
-------
  pso_neuron_step  — Particle Swarm Optimisation
  aco_edge_step    — Ant Colony Optimisation

Shared proxy fitness
--------------------
All GA variants use ``ga_proxy_fitness`` as the base evaluation:
  1. Copy the current graph (networkx — no model copy).
  2. Remove candidate neurons from the copy.
  3. Re-run indegree_grouping.
  4. Return sum of |weight| for all surviving cross-group edges.
No fine-tuning, no SA mapping — < 1 ms per call on LeNet-300-100.

IterCC score helper
-------------------
``_compute_itercc_scores`` computes the Stage-2 score from
iterative_commcost_neuron_pruning for each candidate neuron:
  sum of |weight| for all edges (incoming + outgoing) that cross
  group boundaries under the *current* grouping.
Higher score = neuron contributes more to cross-group NoC traffic.
This is pre-computed once before the GA loop (grouping is fixed for
the duration of the search).

Usage example
-------------
    from pruning_meta import (
        ga_neuron_step,
        ga_neuron_step_biased,
        ga_neuron_step_guided_mutation,
    )

    _ga_base_kwargs = dict(N=36, grouping_cap=10_000.0,
                           low_mag_percentile=90.0, pop_size=30,
                           n_generations=20, tournament_k=3,
                           mutation_rate=0.2, elite_frac=0.1,
                           ga_seed=0, debug=False)

    _ga_biased_kwargs  = dict(**_ga_base_kwargs, alpha=0.7, beta=0.3)
    _ga_guided_kwargs  = dict(**_ga_base_kwargs, warm_start_frac=0.5)
"""

from __future__ import annotations

import random as _random
from typing import TYPE_CHECKING

import numpy as np
import torch.nn as nn

if TYPE_CHECKING:
    from prunable_network import PrunableNetwork


# ==================================================================
# Internal utilities
# ==================================================================

def _count_hidden_neurons(model: nn.Module) -> int:
    """Count total hidden-layer output neurons (excludes output layer)."""
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return sum(m.out_features for m in linears[:-1])


def _magnitude_filter(net: "PrunableNetwork",
                      hidden_nodes: list,
                      low_mag_percentile: float) -> list:
    """
    Stage-1 filter shared by all GA variants (mirrors IterCC Stage-1).

    Returns the subset of hidden_nodes whose mean |incoming weight| falls
    in the bottom ``low_mag_percentile`` percent.  If the filtered pool
    is empty, returns the full hidden_nodes list unchanged.

    Graph convention: neuron (li+1, j) has incoming weights from
    linears[li].weight[j, :]  (shape: out_features × in_features).
    """
    linears = [m for m in net.model.modules() if isinstance(m, nn.Linear)]

    mag_map: dict = {}
    for li, layer in enumerate(linears[:-1]):
        w = layer.weight.data
        for j in range(w.shape[0]):
            mag_map[(li + 1, j)] = float(w[j, :].abs().mean())

    mags = np.array([mag_map.get(n, 0.0) for n in hidden_nodes])

    if low_mag_percentile < 100.0 and len(mags) > 0:
        thresh     = float(np.percentile(mags, low_mag_percentile))
        candidates = [n for n, m in zip(hidden_nodes, mags) if m <= thresh]
        if candidates:
            return candidates

    return list(hidden_nodes)


def _compute_itercc_scores(graph, candidate_nodes: list, groups: list) -> list[float]:
    """
    Compute the IterCC Stage-2 cross-group edge weight score for each
    candidate neuron under the given grouping.

    For neuron u in group g:
        score(u) = Σ |w(pred, u)| for all preds NOT in g
                 + Σ |w(u, succ)| for all succs NOT in g

    This measures how much cross-group NoC traffic the neuron is
    responsible for.  Higher score = removing this neuron reduces
    cross-group traffic more.

    Parameters
    ----------
    graph          : current net.graph (networkx DiGraph with 'weight' edges)
    candidate_nodes: list of (layer_idx, neuron_idx) nodes to score
    groups         : list of node lists from indegree_grouping

    Returns
    -------
    list of float, same length as candidate_nodes
    """
    node_to_grp: dict = {}
    for gid, members in enumerate(groups):
        for nd in members:
            node_to_grp[nd] = gid

    scores: list[float] = []
    for node in candidate_nodes:
        node_grp = node_to_grp.get(node)
        if node_grp is None:
            scores.append(0.0)
            continue
        score = 0.0
        for pred in graph.predecessors(node):
            if node_to_grp.get(pred) != node_grp:
                score += abs(graph[pred][node].get("weight", 0.0))
        for succ in graph.successors(node):
            if node_to_grp.get(succ) != node_grp:
                score += abs(graph[node][succ].get("weight", 0.0))
        scores.append(score)

    return scores


def _ga_core(
    graph,
    candidate_nodes: list,
    max_neurons: int,
    N: int,
    grouping_cap: float,
    pop_size: int,
    n_generations: int,
    tournament_k: int,
    mutation_rate: float,
    elite_frac: int,
    rng: _random.Random,
    fitness_fn,          # callable(individual: list[int]) -> float
    mutate_fn,           # callable(individual: list[int]) -> list[int]
    initial_population: list[list[int]] | None,
    debug: bool,
    label: str = "GA",
    on_generation_end=None,  # optional callable(gen: int, best_ind: list[int]) -> None
    local_search_fn=None,    # optional callable(individual: list[int]) -> list[int]; Lamarckian
) -> list[int] | None:
    """
    Shared GA loop used by all three variants.

    Parameters
    ----------
    fitness_fn         : maps chromosome (list of indices into candidate_nodes)
                         to a scalar; lower = better.
    mutate_fn          : maps chromosome to a mutated chromosome of same length.
    initial_population : if provided, used as the starting population;
                         otherwise initialised randomly.
    label              : string shown in debug output.

    Returns
    -------
    Best individual found (list of indices), or None if population is empty.
    """
    n_cands = len(candidate_nodes)

    def _make_individual() -> list[int]:
        return sorted(rng.sample(range(n_cands), max_neurons))

    def _tournament(scored: list[tuple[float, list[int]]]) -> list[int]:
        pool = rng.sample(scored, min(tournament_k, len(scored)))
        return min(pool, key=lambda x: x[0])[1]

    def _crossover(p1: list[int], p2: list[int]) -> list[int]:
        s1, s2  = set(p1), set(p2)
        shared  = list(s1 & s2)
        unique  = list(s1.symmetric_difference(s2))
        rng.shuffle(unique)
        child   = list(shared)
        for gene in unique:
            if len(child) >= max_neurons:
                break
            child.append(gene)
        if len(child) < max_neurons:
            used      = set(child)
            remaining = [i for i in range(n_cands) if i not in used]
            rng.shuffle(remaining)
            child.extend(remaining[: max_neurons - len(child)])
        return sorted(child[:max_neurons])

    n_elite    = max(1, int(pop_size * elite_frac))
    population = initial_population if initial_population else [
        _make_individual() for _ in range(pop_size)
    ]
    # Ensure exactly pop_size individuals
    while len(population) < pop_size:
        population.append(_make_individual())
    population = population[:pop_size]

    best_individual: list[int] | None = None
    best_fitness = float("inf")

    for gen in range(n_generations):
        scored: list[tuple[float, list[int]]] = [
            (fitness_fn(ind), ind) for ind in population
        ]
        scored.sort(key=lambda x: x[0])

        if scored[0][0] < best_fitness:
            best_fitness    = scored[0][0]
            best_individual = scored[0][1][:]

        if debug:
            print(f"    [{label}] Gen {gen+1:3d}/{n_generations}  "
                  f"best={best_fitness:.6f}  gen_best={scored[0][0]:.6f}")

        if on_generation_end is not None and best_individual is not None:
            on_generation_end(gen, best_individual[:])

        new_pop: list[list[int]] = [ind for _, ind in scored[:n_elite]]
        while len(new_pop) < pop_size:
            p1    = _tournament(scored)
            p2    = _tournament(scored)
            child = _crossover(p1, p2)
            child = mutate_fn(child)
            # Guarantee exactly max_neurons unique genes.
            child = sorted(set(child))
            while len(child) < max_neurons:
                used   = set(child)
                unused = [i for i in range(n_cands) if i not in used]
                if not unused:
                    break
                child.append(rng.choice(unused))
                child.sort()
            # Lamarckian local search: refine offspring before evaluation.
            if local_search_fn is not None:
                child = local_search_fn(child)
            new_pop.append(child[:max_neurons])
        population = new_pop

    return best_individual


def _prune_winner(net: "PrunableNetwork",
                  candidate_nodes: list,
                  winner: list[int],
                  best_fitness: float) -> tuple[int, list]:
    """Apply the winning chromosome to the model and return (n_removed, history)."""
    to_prune = [candidate_nodes[i] for i in winner]
    n_before = _count_hidden_neurons(net.model)
    net.prune_neurons(to_prune)
    n_after  = _count_hidden_neurons(net.model)
    return n_before - n_after, [(nd, best_fitness) for nd in to_prune]


# ==================================================================
# Shared proxy fitness
# ==================================================================

def ga_proxy_fitness(
    graph,
    nodes_to_remove: list,
    N_groups: int,
    grouping_cap: float,
) -> float:
    """
    Fast proxy fitness for a candidate pruning set.

    Operates entirely on a graph copy — no model forward pass, no fine-tuning,
    no SA mapping.  Runtime is dominated by indegree_grouping (~1 ms for
    LeNet-300-100).

    Returns
    -------
    float
        Sum of |weight| for all cross-group edges in the pruned graph.
        Lower is better.  Returns ``inf`` if grouping raises an exception.
    """
    from grouping_heuristic import indegree_grouping

    g     = graph.copy()
    valid = [n for n in nodes_to_remove if g.has_node(n)]
    if valid:
        g.remove_nodes_from(valid)

    try:
        groups = indegree_grouping(g, N_groups, grouping_cap)
    except Exception:
        return float("inf")

    node_to_grp: dict = {}
    for gid, members in enumerate(groups):
        for nd in members:
            node_to_grp[nd] = gid

    cross_cost = 0.0
    for u, v, data in g.edges(data=True):
        gu = node_to_grp.get(u)
        gv = node_to_grp.get(v)
        if gu is not None and gv is not None and gu != gv:
            cross_cost += abs(data.get("weight", 1.0))

    return cross_cost


# ==================================================================
# GA variant 1 — plain proxy fitness  (Branches M / N)
# ==================================================================

def ga_neuron_step(
    net: "PrunableNetwork",
    *,
    max_neurons: int,
    N: int,
    grouping_cap: float = 10_000.0,
    low_mag_percentile: float = 90.0,
    pop_size: int = 30,
    n_generations: int = 20,
    tournament_k: int = 3,
    mutation_rate: float = 0.2,
    elite_frac: float = 0.1,
    ga_seed: int = 0,
    debug: bool = False,
    **_ignored,
) -> tuple[int, list]:
    """
    Plain GA neuron pruner.

    Fitness = ga_proxy_fitness (cross-group edge |weight| sum after removal).
    Mutation = random gene swap.
    Initialisation = fully random population.

    Use as the baseline metaheuristic (Branches M / N).
    """
    rng = _random.Random(ga_seed)

    graph        = net.graph
    all_layers   = sorted(set(n[0] for n in graph.nodes()))
    if len(all_layers) < 2:
        return 0, []

    input_layer  = all_layers[0]
    output_layer = all_layers[-1]
    hidden_nodes = [n for n in graph.nodes()
                    if n[0] != input_layer and n[0] != output_layer]
    if not hidden_nodes:
        return 0, []

    candidate_nodes = _magnitude_filter(net, hidden_nodes, low_mag_percentile)
    n_cands         = len(candidate_nodes)

    if n_cands <= max_neurons:
        cost = ga_proxy_fitness(graph, candidate_nodes, N, grouping_cap)
        if debug:
            print(f"    [GA] {n_cands} candidates ≤ budget. Pruning all. cost={cost:.4f}")
        return _prune_winner(net, candidate_nodes, list(range(n_cands)), cost)

    def _fitness(ind: list[int]) -> float:
        return ga_proxy_fitness(graph, [candidate_nodes[i] for i in ind], N, grouping_cap)

    def _mutate(ind: list[int]) -> list[int]:
        child = list(ind)
        used  = set(child)
        for idx in range(len(child)):
            if rng.random() < mutation_rate:
                unused = [i for i in range(n_cands) if i not in used]
                if unused:
                    used.discard(child[idx])
                    child[idx] = rng.choice(unused)
                    used.add(child[idx])
        return sorted(child)

    winner = _ga_core(
        graph, candidate_nodes, max_neurons, N, grouping_cap,
        pop_size, n_generations, tournament_k, mutation_rate, elite_frac,
        rng, _fitness, _mutate, None, debug, label="GA",
    )
    if winner is None:
        return 0, []

    best_cost = _fitness(winner)
    if debug:
        print(f"    [GA] Winner proxy={best_cost:.4f}")
    return _prune_winner(net, candidate_nodes, winner, best_cost)


# ==================================================================
# GA variant 2 — IterCC-biased fitness  (Branch O)
# ==================================================================

def ga_neuron_step_biased(
    net: "PrunableNetwork",
    *,
    max_neurons: int,
    N: int,
    grouping_cap: float = 10_000.0,
    low_mag_percentile: float = 90.0,
    pop_size: int = 30,
    n_generations: int = 20,
    tournament_k: int = 3,
    mutation_rate: float = 0.2,
    elite_frac: float = 0.1,
    ga_seed: int = 0,
    alpha: float = 0.7,
    beta: float = 0.3,
    debug: bool = False,
    **_ignored,
) -> tuple[int, list]:
    """
    Biased-fitness GA neuron pruner  (Branch O).

    Fitness combines two normalised terms:

        fitness = α × proxy_norm - β × itercc_norm

    where
        proxy_norm   = ga_proxy_fitness(chromosome) / baseline_cost
                       (fraction of current cross-group cost remaining;
                        lower is better → minimised)

        itercc_norm  = Σ itercc_score(n) / max_achievable_itercc_sum
                       (fraction of maximum possible IterCC score achieved
                        by this chromosome; higher is better → subtracted)

    Minimising this combined objective rewards chromosomes that
      (a) achieve low post-removal cross-group cost  (proxy term), AND
      (b) select neurons that individually contribute most to cross-group
          traffic  (IterCC domain knowledge term).

    α + β = 1 (recommended).  α controls how much weight goes to the
    proxy measurement vs the heuristic guidance.

    Parameters
    ----------
    alpha : Weight for the proxy cost term (default 0.7).
    beta  : Weight for the IterCC score term (default 0.3).
    """
    from grouping_heuristic import indegree_grouping

    rng = _random.Random(ga_seed)

    graph        = net.graph
    all_layers   = sorted(set(n[0] for n in graph.nodes()))
    if len(all_layers) < 2:
        return 0, []

    input_layer  = all_layers[0]
    output_layer = all_layers[-1]
    hidden_nodes = [n for n in graph.nodes()
                    if n[0] != input_layer and n[0] != output_layer]
    if not hidden_nodes:
        return 0, []

    candidate_nodes = _magnitude_filter(net, hidden_nodes, low_mag_percentile)
    n_cands         = len(candidate_nodes)

    if n_cands <= max_neurons:
        cost = ga_proxy_fitness(graph, candidate_nodes, N, grouping_cap)
        return _prune_winner(net, candidate_nodes, list(range(n_cands)), cost)

    # ---- Pre-compute IterCC scores (once, before GA loop) ----
    try:
        groups = indegree_grouping(graph, N, grouping_cap)
    except Exception:
        groups = []
    itercc_scores = _compute_itercc_scores(graph, candidate_nodes, groups)

    # Normalisation anchors:
    #   baseline_cost    = current cross-group cost (nothing removed)
    #   max_itercc_sum   = sum of top max_neurons individual IterCC scores
    baseline_cost  = ga_proxy_fitness(graph, [], N, grouping_cap)
    sorted_scores  = sorted(itercc_scores, reverse=True)
    max_itercc_sum = sum(sorted_scores[:max_neurons])

    _eps = 1e-12

    def _biased_fitness(ind: list[int]) -> float:
        nodes      = [candidate_nodes[i] for i in ind]
        proxy      = ga_proxy_fitness(graph, nodes, N, grouping_cap)
        proxy_norm = proxy / (baseline_cost + _eps)

        itercc_sum  = sum(itercc_scores[i] for i in ind)
        itercc_norm = itercc_sum / (max_itercc_sum + _eps)

        # Minimise: low remaining cost AND high IterCC-score neurons chosen.
        return alpha * proxy_norm - beta * itercc_norm

    def _mutate(ind: list[int]) -> list[int]:
        child = list(ind)
        used  = set(child)
        for idx in range(len(child)):
            if rng.random() < mutation_rate:
                unused = [i for i in range(n_cands) if i not in used]
                if unused:
                    used.discard(child[idx])
                    child[idx] = rng.choice(unused)
                    used.add(child[idx])
        return sorted(child)

    if debug:
        print(f"    [GA-Biased] baseline_cost={baseline_cost:.2f}  "
              f"max_itercc_sum={max_itercc_sum:.4f}  "
              f"alpha={alpha}  beta={beta}")

    winner = _ga_core(
        graph, candidate_nodes, max_neurons, N, grouping_cap,
        pop_size, n_generations, tournament_k, mutation_rate, elite_frac,
        rng, _biased_fitness, _mutate, None, debug, label="GA-Biased",
    )
    if winner is None:
        return 0, []

    best_fitness = _biased_fitness(winner)
    best_proxy   = ga_proxy_fitness(
        graph, [candidate_nodes[i] for i in winner], N, grouping_cap)
    if debug:
        print(f"    [GA-Biased] Winner: biased={best_fitness:.6f}  "
              f"proxy={best_proxy:.4f}")
    return _prune_winner(net, candidate_nodes, winner, best_proxy)


# ==================================================================
# GA variant 2b — IterCC-biased fitness with adaptive re-scoring  (Branch Q)
# ==================================================================

def ga_neuron_step_biased_v2(
    net: "PrunableNetwork",
    *,
    max_neurons: int,
    N: int,
    grouping_cap: float = 10_000.0,
    low_mag_percentile: float = 90.0,
    pop_size: int = 30,
    n_generations: int = 20,
    tournament_k: int = 3,
    mutation_rate: float = 0.2,
    elite_frac: float = 0.1,
    ga_seed: int = 0,
    alpha: float = 0.6,
    beta: float = 0.4,
    rescore_interval: int = 5,
    debug: bool = False,
    **_ignored,
) -> tuple[int, list]:
    """
    Refined biased-fitness GA neuron pruner  (Branch Q).

    Two improvements over Branch O (ga_neuron_step_biased):

    1. Higher beta (default 0.4 vs 0.3)
       Gives more weight to the IterCC domain-knowledge term in the
       composite fitness.  Early in training, the GA benefits more
       from the heuristic's signal; the higher beta amplifies this.

    2. Adaptive IterCC re-scoring every ``rescore_interval`` generations.
       After every rescore_interval GA generations, the IterCC scores are
       re-computed on a pruned copy of the graph with the CURRENT BEST
       individual's neurons removed.  This updates the bias term to reflect
       which *remaining* candidates are most valuable given the choices the
       GA has already converged toward — a form of iterated refinement.

       At generation 0 the scores are computed on the full graph (same as
       Branch O).  After re-scoring, the max_itercc_sum normaliser is also
       updated so the beta term remains well-scaled.

    Parameters
    ----------
    alpha           : Weight for the proxy cost term (default 0.6).
    beta            : Weight for the IterCC score term (default 0.4).
    rescore_interval: Re-compute IterCC scores every this many generations
                      (0 or negative = never re-score; degenerates to Branch O
                      with higher beta).
    """
    from grouping_heuristic import indegree_grouping

    rng = _random.Random(ga_seed)

    graph        = net.graph
    all_layers   = sorted(set(n[0] for n in graph.nodes()))
    if len(all_layers) < 2:
        return 0, []

    input_layer  = all_layers[0]
    output_layer = all_layers[-1]
    hidden_nodes = [n for n in graph.nodes()
                    if n[0] != input_layer and n[0] != output_layer]
    if not hidden_nodes:
        return 0, []

    candidate_nodes = _magnitude_filter(net, hidden_nodes, low_mag_percentile)
    n_cands         = len(candidate_nodes)

    if n_cands <= max_neurons:
        cost = ga_proxy_fitness(graph, candidate_nodes, N, grouping_cap)
        return _prune_winner(net, candidate_nodes, list(range(n_cands)), cost)

    # ---- Pre-compute initial IterCC scores on the full graph ----
    try:
        groups = indegree_grouping(graph, N, grouping_cap)
    except Exception:
        groups = []
    itercc_scores = list(_compute_itercc_scores(graph, candidate_nodes, groups))

    baseline_cost  = ga_proxy_fitness(graph, [], N, grouping_cap)
    sorted_scores  = sorted(itercc_scores, reverse=True)
    max_itercc_sum = sum(sorted_scores[:max_neurons])

    # Mutable state shared between fitness_fn and the re-score callback.
    _state = {
        "itercc":        itercc_scores,
        "max_itercc_sum": max_itercc_sum,
    }
    _eps = 1e-12

    def _biased_fitness(ind: list[int]) -> float:
        nodes       = [candidate_nodes[i] for i in ind]
        proxy       = ga_proxy_fitness(graph, nodes, N, grouping_cap)
        proxy_norm  = proxy / (baseline_cost + _eps)
        itercc_sum  = sum(_state["itercc"][i] for i in ind)
        itercc_norm = itercc_sum / (_state["max_itercc_sum"] + _eps)
        return alpha * proxy_norm - beta * itercc_norm

    def _mutate(ind: list[int]) -> list[int]:
        child = list(ind)
        used  = set(child)
        for idx in range(len(child)):
            if rng.random() < mutation_rate:
                unused = [i for i in range(n_cands) if i not in used]
                if unused:
                    used.discard(child[idx])
                    child[idx] = rng.choice(unused)
                    used.add(child[idx])
        return sorted(child)

    # ---- Re-scoring callback injected into _ga_core ----
    def _on_generation_end(gen: int, best_ind: list[int]) -> None:
        if rescore_interval <= 0 or (gen + 1) % rescore_interval != 0:
            return
        # Build a pruned graph: remove the best individual's neurons.
        g_pruned    = graph.copy()
        to_rm       = [candidate_nodes[i] for i in best_ind
                       if graph.has_node(candidate_nodes[i])]
        g_pruned.remove_nodes_from(to_rm)
        try:
            grp_pruned = indegree_grouping(g_pruned, N, grouping_cap)
        except Exception:
            grp_pruned = []
        # Re-score all candidates on the pruned graph (nodes already removed
        # will get score 0.0 from _compute_itercc_scores, which is fine since
        # they are already selected and won't improve the fitness further).
        new_scores = _compute_itercc_scores(g_pruned, candidate_nodes, grp_pruned)
        _state["itercc"] = list(new_scores)
        _state["max_itercc_sum"] = sum(sorted(new_scores, reverse=True)[:max_neurons]) + _eps
        if debug:
            print(f"    [GA-Biased-v2] Gen {gen+1}: re-scored IterCC on pruned graph  "
                  f"new max_itercc_sum={_state['max_itercc_sum']:.4f}")

    if debug:
        print(f"    [GA-Biased-v2] baseline_cost={baseline_cost:.2f}  "
              f"max_itercc_sum={max_itercc_sum:.4f}  "
              f"alpha={alpha}  beta={beta}  rescore_interval={rescore_interval}")

    winner = _ga_core(
        graph, candidate_nodes, max_neurons, N, grouping_cap,
        pop_size, n_generations, tournament_k, mutation_rate, elite_frac,
        rng, _biased_fitness, _mutate, None, debug, label="GA-Biased-v2",
        on_generation_end=_on_generation_end,
    )
    if winner is None:
        return 0, []

    best_fitness = _biased_fitness(winner)
    best_proxy   = ga_proxy_fitness(
        graph, [candidate_nodes[i] for i in winner], N, grouping_cap)
    if debug:
        print(f"    [GA-Biased-v2] Winner: biased={best_fitness:.6f}  "
              f"proxy={best_proxy:.4f}")
    return _prune_winner(net, candidate_nodes, winner, best_proxy)


# ==================================================================
# GA variant 2c — β-annealing + faster re-scoring  (Branch S)
# ==================================================================

def ga_neuron_step_biased_v3(
    net: "PrunableNetwork",
    *,
    max_neurons: int,
    N: int,
    grouping_cap: float = 10_000.0,
    low_mag_percentile: float = 90.0,
    pop_size: int = 30,
    n_generations: int = 20,
    tournament_k: int = 3,
    mutation_rate: float = 0.2,
    elite_frac: float = 0.1,
    ga_seed: int = 0,
    beta_start: float = 0.5,
    beta_end: float = 0.3,
    rescore_interval: int = 3,
    debug: bool = False,
    **_ignored,
) -> tuple[int, list]:
    """
    Further-refined biased-fitness GA neuron pruner  (Branch S).

    Two improvements over Branch Q (ga_neuron_step_biased_v2):

    1. β annealing  (targets the iso-accuracy / early-step gap)
       β decreases linearly from ``beta_start`` (default 0.5) to
       ``beta_end`` (default 0.3) across the n_generations.
       α is computed as 1 − β at each generation and is not a
       separate parameter.

       Rationale: early generations benefit from stronger IterCC
       guidance (high β) to quickly reach high-quality regions;
       later generations use more proxy-driven selection (low β)
       for fine-grained cost optimisation.  The annealing schedule
       is applied in the ``on_generation_end`` callback so each
       generation uses the β that is appropriate for that stage.

           β(gen) = beta_start − (beta_start − beta_end)
                    × gen / max(n_generations − 1, 1)

    2. Faster adaptive re-scoring  (rescore_interval=3 vs 5)
       Re-computes IterCC scores on the best individual's pruned
       graph every 3 generations instead of every 5.  This fires
       6–7 times in a 20-generation run (vs 4 times for Branch Q),
       allowing the bias term to adapt more quickly as the
       population converges.

    Both mechanisms operate through the same ``_state`` dict /
    ``on_generation_end`` hook as Branch Q, so the core GA loop
    (``_ga_core``) is unchanged.

    Parameters
    ----------
    beta_start      : Initial β at generation 0 (default 0.5).
    beta_end        : Final β at generation n_generations-1 (default 0.3).
    rescore_interval: Re-score every this many generations (default 3).
    """
    from grouping_heuristic import indegree_grouping

    rng = _random.Random(ga_seed)

    graph        = net.graph
    all_layers   = sorted(set(n[0] for n in graph.nodes()))
    if len(all_layers) < 2:
        return 0, []

    input_layer  = all_layers[0]
    output_layer = all_layers[-1]
    hidden_nodes = [n for n in graph.nodes()
                    if n[0] != input_layer and n[0] != output_layer]
    if not hidden_nodes:
        return 0, []

    candidate_nodes = _magnitude_filter(net, hidden_nodes, low_mag_percentile)
    n_cands         = len(candidate_nodes)

    if n_cands <= max_neurons:
        cost = ga_proxy_fitness(graph, candidate_nodes, N, grouping_cap)
        return _prune_winner(net, candidate_nodes, list(range(n_cands)), cost)

    # ---- Pre-compute initial IterCC scores on the full graph ----
    try:
        groups = indegree_grouping(graph, N, grouping_cap)
    except Exception:
        groups = []
    itercc_scores = list(_compute_itercc_scores(graph, candidate_nodes, groups))

    baseline_cost  = ga_proxy_fitness(graph, [], N, grouping_cap)
    sorted_scores  = sorted(itercc_scores, reverse=True)
    max_itercc_sum = sum(sorted_scores[:max_neurons])

    _eps = 1e-12

    # Mutable state: β/α annealing + adaptive IterCC scores.
    _state = {
        "beta":          beta_start,
        "alpha":         1.0 - beta_start,
        "itercc":        itercc_scores,
        "max_itercc_sum": max_itercc_sum + _eps,
    }

    def _biased_fitness(ind: list[int]) -> float:
        nodes       = [candidate_nodes[i] for i in ind]
        proxy       = ga_proxy_fitness(graph, nodes, N, grouping_cap)
        proxy_norm  = proxy / (baseline_cost + _eps)
        itercc_sum  = sum(_state["itercc"][i] for i in ind)
        itercc_norm = itercc_sum / _state["max_itercc_sum"]
        return _state["alpha"] * proxy_norm - _state["beta"] * itercc_norm

    def _mutate(ind: list[int]) -> list[int]:
        child = list(ind)
        used  = set(child)
        for idx in range(len(child)):
            if rng.random() < mutation_rate:
                unused = [i for i in range(n_cands) if i not in used]
                if unused:
                    used.discard(child[idx])
                    child[idx] = rng.choice(unused)
                    used.add(child[idx])
        return sorted(child)

    def _on_generation_end(gen: int, best_ind: list[int]) -> None:
        # 1. Anneal β for the NEXT generation.
        next_gen = gen + 1
        if next_gen < n_generations:
            new_beta = (beta_start
                        - (beta_start - beta_end) * next_gen
                        / max(n_generations - 1, 1))
            _state["beta"]  = new_beta
            _state["alpha"] = 1.0 - new_beta
            if debug:
                print(f"    [GA-Biased-v3] Gen {gen+1}: "
                      f"β → {new_beta:.4f}  α → {_state['alpha']:.4f}")

        # 2. Re-score IterCC on the pruned graph every rescore_interval gens.
        if rescore_interval > 0 and (gen + 1) % rescore_interval == 0:
            g_pruned = graph.copy()
            to_rm    = [candidate_nodes[i] for i in best_ind
                        if graph.has_node(candidate_nodes[i])]
            g_pruned.remove_nodes_from(to_rm)
            try:
                grp_pruned = indegree_grouping(g_pruned, N, grouping_cap)
            except Exception:
                grp_pruned = []
            new_scores = _compute_itercc_scores(g_pruned, candidate_nodes, grp_pruned)
            _state["itercc"]         = list(new_scores)
            _state["max_itercc_sum"] = (sum(sorted(new_scores, reverse=True)[:max_neurons])
                                        + _eps)
            if debug:
                print(f"    [GA-Biased-v3] Gen {gen+1}: "
                      f"re-scored IterCC on pruned graph  "
                      f"new max_itercc_sum={_state['max_itercc_sum']:.4f}")

    if debug:
        print(f"    [GA-Biased-v3] baseline_cost={baseline_cost:.2f}  "
              f"β: {beta_start}→{beta_end}  rescore_interval={rescore_interval}")

    winner = _ga_core(
        graph, candidate_nodes, max_neurons, N, grouping_cap,
        pop_size, n_generations, tournament_k, mutation_rate, elite_frac,
        rng, _biased_fitness, _mutate, None, debug, label="GA-Biased-v3",
        on_generation_end=_on_generation_end,
    )
    if winner is None:
        return 0, []

    best_proxy = ga_proxy_fitness(
        graph, [candidate_nodes[i] for i in winner], N, grouping_cap)
    if debug:
        print(f"    [GA-Biased-v3] Winner proxy={best_proxy:.4f}")
    return _prune_winner(net, candidate_nodes, winner, best_proxy)


# ==================================================================
# GA variant 3 — IterCC-guided mutation + warm-start  (Branch P)
# ==================================================================

def ga_neuron_step_guided_mutation(
    net: "PrunableNetwork",
    *,
    max_neurons: int,
    N: int,
    grouping_cap: float = 10_000.0,
    low_mag_percentile: float = 90.0,
    pop_size: int = 30,
    n_generations: int = 20,
    tournament_k: int = 3,
    mutation_rate: float = 0.2,
    elite_frac: float = 0.1,
    ga_seed: int = 0,
    warm_start_frac: float = 0.5,
    debug: bool = False,
    **_ignored,
) -> tuple[int, list]:
    """
    Guided-mutation GA neuron pruner  (Branch P).

    Two modifications over the plain GA:

    1. IterCC-guided mutation
       Standard random mutation swaps a gene for any random unused neuron.
       Here, the replacement is always the *highest-IterCC-scoring* unused
       candidate in the pool — so mutations always move toward what the
       heuristic considers the best available choice.

       This preserves the GA's combinatorial exploration (crossover still
       mixes genes freely) while making each mutation "intelligent" rather
       than blind.

    2. IterCC warm-start population
       A fraction ``warm_start_frac`` of the initial population is seeded
       from IterCC's greedy ranking rather than randomly:
         - Individual 0 = IterCC's exact greedy top-K answer.
         - Individuals 1..n_warm = variants that swap one gene at a time
           for the next-best IterCC candidate not yet in the chromosome.
         - Individuals n_warm..pop_size = random (exploration).

       This means evolution begins near the heuristic's answer and explores
       outward, rather than starting from scratch.

    Parameters
    ----------
    warm_start_frac : Fraction of population initialised from IterCC ranking.
                      0.0 = fully random (degenerates to plain GA).
                      1.0 = all individuals are IterCC variants.
                      Default 0.5.
    """
    from grouping_heuristic import indegree_grouping

    rng = _random.Random(ga_seed)

    graph        = net.graph
    all_layers   = sorted(set(n[0] for n in graph.nodes()))
    if len(all_layers) < 2:
        return 0, []

    input_layer  = all_layers[0]
    output_layer = all_layers[-1]
    hidden_nodes = [n for n in graph.nodes()
                    if n[0] != input_layer and n[0] != output_layer]
    if not hidden_nodes:
        return 0, []

    candidate_nodes = _magnitude_filter(net, hidden_nodes, low_mag_percentile)
    n_cands         = len(candidate_nodes)

    if n_cands <= max_neurons:
        cost = ga_proxy_fitness(graph, candidate_nodes, N, grouping_cap)
        return _prune_winner(net, candidate_nodes, list(range(n_cands)), cost)

    # ---- Pre-compute IterCC scores (once, before GA loop) ----
    try:
        groups = indegree_grouping(graph, N, grouping_cap)
    except Exception:
        groups = []
    itercc_scores = _compute_itercc_scores(graph, candidate_nodes, groups)

    # Candidate indices sorted by IterCC score descending.
    # Used both for warm-start and guided mutation.
    ranked_by_itercc = sorted(range(n_cands),
                              key=lambda i: itercc_scores[i], reverse=True)

    # ---- Build warm-start population ----
    n_warm = max(1, int(pop_size * warm_start_frac))

    # Individual 0: IterCC's greedy top-max_neurons answer.
    itercc_top  = sorted(ranked_by_itercc[:max_neurons])
    initial_pop = [itercc_top]

    # Individuals 1..n_warm: swap one gene at a time for next-best IterCC.
    others = [i for i in ranked_by_itercc if i not in set(itercc_top)]
    for k in range(1, n_warm):
        variant = list(itercc_top)
        if others:
            swap_out_pos = rng.randrange(max_neurons)
            swap_in      = others[(k - 1) % len(others)]
            used         = set(variant)
            used.discard(variant[swap_out_pos])
            variant[swap_out_pos] = swap_in
            used.add(swap_in)
        initial_pop.append(sorted(variant))

    if debug:
        print(f"    [GA-Guided] warm_start_frac={warm_start_frac}  "
              f"n_warm={n_warm}  "
              f"IterCC seed: {itercc_top}")

    # ---- Guided mutation operator ----
    def _guided_mutate(ind: list[int]) -> list[int]:
        child = list(ind)
        used  = set(child)
        for idx in range(len(child)):
            if rng.random() < mutation_rate:
                # Replace with the highest-IterCC-scoring unused candidate.
                for best in ranked_by_itercc:
                    if best not in used:
                        used.discard(child[idx])
                        child[idx] = best
                        used.add(best)
                        break
        return sorted(child)

    def _fitness(ind: list[int]) -> float:
        return ga_proxy_fitness(
            graph, [candidate_nodes[i] for i in ind], N, grouping_cap)

    winner = _ga_core(
        graph, candidate_nodes, max_neurons, N, grouping_cap,
        pop_size, n_generations, tournament_k, mutation_rate, elite_frac,
        rng, _fitness, _guided_mutate, initial_pop, debug, label="GA-Guided",
    )
    if winner is None:
        return 0, []

    best_cost = _fitness(winner)
    if debug:
        print(f"    [GA-Guided] Winner proxy={best_cost:.4f}")
    return _prune_winner(net, candidate_nodes, winner, best_cost)


# ==================================================================
# GA variant 3b — probabilistic guided mutation + minimal warm-start  (Branch T)
# ==================================================================

def ga_neuron_step_guided_mutation_v2(
    net: "PrunableNetwork",
    *,
    max_neurons: int,
    N: int,
    grouping_cap: float = 10_000.0,
    low_mag_percentile: float = 90.0,
    pop_size: int = 30,
    n_generations: int = 20,
    tournament_k: int = 3,
    mutation_rate: float = 0.2,
    elite_frac: float = 0.1,
    ga_seed: int = 0,
    warm_start_frac: float = 0.10,
    guided_mutation_prob: float = 0.50,
    debug: bool = False,
    **_ignored,
) -> tuple[int, list]:
    """
    Improved guided-mutation GA neuron pruner  (Branch T).

    Fixes the premature-convergence problem of Branch P by decoupling
    and weakening its two domain-knowledge mechanisms:

    Branch P problem
    ----------------
    P used two IterCC-guided operators simultaneously at full strength:
      • warm_start_frac=0.5  — 15 of 30 individuals seeded from IterCC's
        top-K ranking, leaving only 15 random individuals.
      • Deterministic guided mutation — every mutation ALWAYS replaces a
        gene with the highest-IterCC-scoring unused candidate.

    Together these collapsed population diversity by generation 3–4.
    The GA converged to IterCC's greedy answer immediately, exhausted its
    budget on a single coarse bridge-link edge step, and stopped at only
    −0.02% accuracy drop (81.77% cost reduction), leaving 0.98% of the
    1% accuracy budget untouched.

    Branch T fixes
    --------------
    1. Minimal warm-start  (warm_start_frac=0.10, default)
       Only 3 of 30 individuals are seeded from IterCC's top-K ranking.
       The remaining 27 start randomly, preserving the genetic diversity
       needed for crossover to explore genuinely different solutions.

    2. Probabilistic guided mutation  (guided_mutation_prob=0.50, default)
       When a gene is selected for mutation (prob = mutation_rate = 0.2),
       a second coin is flipped:
         • prob = guided_mutation_prob  → IterCC-directed replacement
           (same as P: swap for the highest-IterCC-scoring unused candidate)
         • prob = 1 − guided_mutation_prob → random replacement
           (same as plain GA)

       On average this means ~10% of all genes across the population
       receive an IterCC-directed nudge per generation (0.2 × 0.5 = 0.10),
       compared to 20% in Branch P (100% of triggered mutations → IterCC).
       The nudge biases the search without locking it in.

    The proxy fitness (cross-group edge |weight| sum after removal) is
    unchanged from Branch P and the plain GA — only the search operators
    are modified.

    Parameters
    ----------
    warm_start_frac      : Fraction of population from IterCC ranking
                           (default 0.10; 0.0 = fully random; 1.0 = all IterCC).
    guided_mutation_prob : Per-mutation probability of IterCC-directed swap
                           (default 0.50; 0.0 = plain GA; 1.0 = Branch P behaviour).
    """
    from grouping_heuristic import indegree_grouping

    rng = _random.Random(ga_seed)

    graph        = net.graph
    all_layers   = sorted(set(n[0] for n in graph.nodes()))
    if len(all_layers) < 2:
        return 0, []

    input_layer  = all_layers[0]
    output_layer = all_layers[-1]
    hidden_nodes = [n for n in graph.nodes()
                    if n[0] != input_layer and n[0] != output_layer]
    if not hidden_nodes:
        return 0, []

    candidate_nodes = _magnitude_filter(net, hidden_nodes, low_mag_percentile)
    n_cands         = len(candidate_nodes)

    if n_cands <= max_neurons:
        cost = ga_proxy_fitness(graph, candidate_nodes, N, grouping_cap)
        return _prune_winner(net, candidate_nodes, list(range(n_cands)), cost)

    # ---- Pre-compute IterCC scores (once, before GA loop) ----
    try:
        groups = indegree_grouping(graph, N, grouping_cap)
    except Exception:
        groups = []
    itercc_scores = _compute_itercc_scores(graph, candidate_nodes, groups)

    # Candidate indices sorted by IterCC score descending — used for
    # both warm-start seeding and the guided mutation operator.
    ranked_by_itercc = sorted(range(n_cands),
                              key=lambda i: itercc_scores[i], reverse=True)

    # ---- Build warm-start population (minimal: 10% of pop) ----
    n_warm = max(1, int(pop_size * warm_start_frac))

    itercc_top  = sorted(ranked_by_itercc[:max_neurons])
    initial_pop = [itercc_top]

    others = [i for i in ranked_by_itercc if i not in set(itercc_top)]
    for k in range(1, n_warm):
        variant = list(itercc_top)
        if others:
            swap_out_pos        = rng.randrange(max_neurons)
            swap_in             = others[(k - 1) % len(others)]
            used                = set(variant)
            used.discard(variant[swap_out_pos])
            variant[swap_out_pos] = swap_in
            used.add(swap_in)
        initial_pop.append(sorted(variant))

    if debug:
        print(f"    [GA-Guided-v2] warm_start_frac={warm_start_frac}  "
              f"n_warm={n_warm}  guided_mutation_prob={guided_mutation_prob}")

    # ---- Probabilistic guided mutation operator ----
    def _prob_guided_mutate(ind: list[int]) -> list[int]:
        child = list(ind)
        used  = set(child)
        for idx in range(len(child)):
            if rng.random() < mutation_rate:
                if rng.random() < guided_mutation_prob:
                    # IterCC-directed: pick highest-ranked unused candidate.
                    for best in ranked_by_itercc:
                        if best not in used:
                            used.discard(child[idx])
                            child[idx] = best
                            used.add(best)
                            break
                else:
                    # Random: same as plain GA mutation.
                    unused = [i for i in range(n_cands) if i not in used]
                    if unused:
                        used.discard(child[idx])
                        child[idx] = rng.choice(unused)
                        used.add(child[idx])
        return sorted(child)

    def _fitness(ind: list[int]) -> float:
        return ga_proxy_fitness(
            graph, [candidate_nodes[i] for i in ind], N, grouping_cap)

    winner = _ga_core(
        graph, candidate_nodes, max_neurons, N, grouping_cap,
        pop_size, n_generations, tournament_k, mutation_rate, elite_frac,
        rng, _fitness, _prob_guided_mutate, initial_pop, debug,
        label="GA-Guided-v2",
    )
    if winner is None:
        return 0, []

    best_cost = _fitness(winner)
    if debug:
        print(f"    [GA-Guided-v2] Winner proxy={best_cost:.4f}")
    return _prune_winner(net, candidate_nodes, winner, best_cost)


# ==================================================================
# GA variant 4 — Memetic: proxy fitness + IterCC-guided local search  (Branch U)
# ==================================================================

def ga_neuron_step_memetic(
    net: "PrunableNetwork",
    *,
    max_neurons: int,
    N: int,
    grouping_cap: float = 10_000.0,
    low_mag_percentile: float = 90.0,
    pop_size: int = 30,
    n_generations: int = 20,
    tournament_k: int = 3,
    mutation_rate: float = 0.2,
    elite_frac: float = 0.1,
    ga_seed: int = 0,
    local_search_iters: int = 2,
    local_search_candidates: int = 5,
    debug: bool = False,
    **_ignored,
) -> tuple[int, list]:
    """
    Memetic GA neuron pruner  (Branch U).

    Hybridises the plain GA (Branch M) with a Lamarckian local search:
    after crossover + mutation creates an offspring, a greedy hill-climbing
    step refines the chromosome before it enters the next generation.

    Local search design
    -------------------
    For each pass (up to ``local_search_iters``):
      For each gene position in the chromosome:
        - Collect the top-``local_search_candidates`` unused candidates
          ranked by IterCC score (= cross-group edge traffic; same metric
          used by the heuristic).
        - Evaluate the proxy fitness for each candidate swap.
        - Accept the best swap if it strictly improves the proxy.
    A pass that makes no improvement terminates early.

    Why this is different from Branch P / T (guided mutation)
    ----------------------------------------------------------
    P/T injected IterCC knowledge into the *mutation operator* — the
    direction of chromosome generation.  This killed diversity because
    all offspring were pulled toward the same top-IterCC genes before
    selection could act.

    Memetic local search operates *after* crossover and mutation have
    created a diverse offspring set.  Each offspring independently finds
    its own local optimum; the population's variety is preserved because
    the local search starts from different points.  The key differences:

      • Local search fires after crossover (diverse starting points).
      • Acceptance uses proxy fitness, not IterCC alone — a candidate
        only replaces a gene if the proxy confirms it improves the batch
        cross-group cost, not just the individual IterCC score.
      • Local search is bounded (``local_search_iters`` passes × genes ×
        ``local_search_candidates`` proxy evals), keeping runtime under
        control (~30 extra proxy evals per offspring, vs ~1 per offspring
        in plain GA).

    Parameters
    ----------
    local_search_iters      : Maximum hill-climbing passes per offspring
                              (default 2; 0 degenerates to plain GA / Branch M).
    local_search_candidates : Top-K IterCC candidates to try per gene per pass
                              (default 5; larger → better search, higher cost).
    """
    from grouping_heuristic import indegree_grouping

    rng = _random.Random(ga_seed)

    graph        = net.graph
    all_layers   = sorted(set(n[0] for n in graph.nodes()))
    if len(all_layers) < 2:
        return 0, []

    input_layer  = all_layers[0]
    output_layer = all_layers[-1]
    hidden_nodes = [n for n in graph.nodes()
                    if n[0] != input_layer and n[0] != output_layer]
    if not hidden_nodes:
        return 0, []

    candidate_nodes = _magnitude_filter(net, hidden_nodes, low_mag_percentile)
    n_cands         = len(candidate_nodes)

    if n_cands <= max_neurons:
        cost = ga_proxy_fitness(graph, candidate_nodes, N, grouping_cap)
        return _prune_winner(net, candidate_nodes, list(range(n_cands)), cost)

    # ---- Pre-compute IterCC scores (used to guide local search direction) ----
    try:
        groups = indegree_grouping(graph, N, grouping_cap)
    except Exception:
        groups = []
    itercc_scores = _compute_itercc_scores(graph, candidate_nodes, groups)

    # Candidate indices ranked by IterCC score descending.
    ranked_by_itercc = sorted(range(n_cands),
                              key=lambda i: itercc_scores[i], reverse=True)

    def _fitness(ind: list[int]) -> float:
        return ga_proxy_fitness(graph, [candidate_nodes[i] for i in ind], N, grouping_cap)

    def _mutate(ind: list[int]) -> list[int]:
        child = list(ind)
        used  = set(child)
        for idx in range(len(child)):
            if rng.random() < mutation_rate:
                unused = [i for i in range(n_cands) if i not in used]
                if unused:
                    used.discard(child[idx])
                    child[idx] = rng.choice(unused)
                    used.add(child[idx])
        return sorted(child)

    # ---- Lamarckian local search operator ----
    def _local_search(ind: list[int]) -> list[int]:
        child         = list(ind)
        current_proxy = _fitness(child)

        for _pass in range(local_search_iters):
            improved = False
            for gene_pos in range(len(child)):
                in_chrom = set(child)
                in_chrom.discard(child[gene_pos])

                # Collect top-K IterCC candidates not currently in chromosome.
                candidates_to_try: list[int] = []
                for top_idx in ranked_by_itercc:
                    if top_idx not in in_chrom and top_idx != child[gene_pos]:
                        candidates_to_try.append(top_idx)
                        if len(candidates_to_try) >= local_search_candidates:
                            break

                # Find the best swap that strictly improves the proxy.
                best_swap       = None
                best_swap_proxy = current_proxy
                for cand in candidates_to_try:
                    trial              = list(child)
                    trial[gene_pos]    = cand
                    trial_proxy        = _fitness(trial)
                    if trial_proxy < best_swap_proxy:
                        best_swap       = cand
                        best_swap_proxy = trial_proxy

                if best_swap is not None:
                    child[gene_pos] = best_swap
                    current_proxy   = best_swap_proxy
                    improved        = True

            if not improved:
                break

        return sorted(child)

    if debug:
        print(f"    [GA-Memetic] local_search_iters={local_search_iters}  "
              f"local_search_candidates={local_search_candidates}  "
              f"extra_proxy_evals_per_offspring≤"
              f"{local_search_iters * max_neurons * local_search_candidates}")

    winner = _ga_core(
        graph, candidate_nodes, max_neurons, N, grouping_cap,
        pop_size, n_generations, tournament_k, mutation_rate, elite_frac,
        rng, _fitness, _mutate, None, debug, label="GA-Memetic",
        local_search_fn=_local_search,
    )
    if winner is None:
        return 0, []

    best_cost = _fitness(winner)
    if debug:
        print(f"    [GA-Memetic] Winner proxy={best_cost:.4f}")
    return _prune_winner(net, candidate_nodes, winner, best_cost)


# ==================================================================
# GA variant 5 — Memetic + Biased: biased fitness + local search  (Branches V / W)
# ==================================================================

def ga_neuron_step_memetic_biased(
    net: "PrunableNetwork",
    *,
    max_neurons: int,
    N: int,
    grouping_cap: float = 10_000.0,
    low_mag_percentile: float = 90.0,
    pop_size: int = 30,
    n_generations: int = 20,
    tournament_k: int = 3,
    mutation_rate: float = 0.2,
    elite_frac: float = 0.1,
    ga_seed: int = 0,
    alpha: float = 0.6,
    beta: float = 0.4,
    local_search_iters: int = 2,
    local_search_candidates: int = 5,
    debug: bool = False,
    **_ignored,
) -> tuple[int, list]:
    """
    Memetic + Biased GA neuron pruner  (Branches V / W).

    Combines two orthogonal knowledge-injection mechanisms:

    1. Biased fitness for *selection*  (from Branch Q)
       fitness = α × proxy_norm − β × itercc_norm
       Selection rewards chromosomes that are good on BOTH the proxy
       cost signal AND the IterCC domain-knowledge signal.

    2. Lamarckian local search for *individual refinement*  (from Branch U)
       After crossover + mutation, each offspring is hill-climbed using
       the BIASED fitness as the oracle (not the pure proxy).

    Why the oracle choice matters
    -----------------------------
    Branch U used the pure proxy as the local search oracle.  The pure
    proxy landscape has a single dominant basin (IterCC's greedy answer),
    so every individual converged to the same local minimum → population
    collapsed → premature stopping (same as P/T).

    Here the oracle is the biased fitness.  The biased landscape has a
    different basin structure because the subtracted IterCC term reshapes
    it relative to the pure proxy.  Each individual hill-climbs to a
    biased-landscape local optimum; those optima differ across individuals
    because they started from different chromosomes after crossover.
    Population diversity is preserved.

    Local search mechanics  (same as Branch U, different oracle)
    -------------------------------------------------------------
    For each pass (up to ``local_search_iters``):
      For each gene position:
        - Collect top-``local_search_candidates`` unused candidates by
          IterCC rank.
        - Evaluate biased fitness for each candidate swap.
        - Accept the swap with the best biased fitness improvement, if any.
      Terminate pass early if no gene swap improved the biased fitness.

    Parameters
    ----------
    alpha                   : Weight for proxy cost term (default 0.6).
    beta                    : Weight for IterCC score term (default 0.4).
    local_search_iters      : Hill-climbing passes per offspring (default 2).
    local_search_candidates : Top-K IterCC candidates per gene per pass
                              (default 5).
    """
    from grouping_heuristic import indegree_grouping

    rng = _random.Random(ga_seed)

    graph        = net.graph
    all_layers   = sorted(set(n[0] for n in graph.nodes()))
    if len(all_layers) < 2:
        return 0, []

    input_layer  = all_layers[0]
    output_layer = all_layers[-1]
    hidden_nodes = [n for n in graph.nodes()
                    if n[0] != input_layer and n[0] != output_layer]
    if not hidden_nodes:
        return 0, []

    candidate_nodes = _magnitude_filter(net, hidden_nodes, low_mag_percentile)
    n_cands         = len(candidate_nodes)

    if n_cands <= max_neurons:
        cost = ga_proxy_fitness(graph, candidate_nodes, N, grouping_cap)
        return _prune_winner(net, candidate_nodes, list(range(n_cands)), cost)

    # ---- Pre-compute IterCC scores ----
    try:
        groups = indegree_grouping(graph, N, grouping_cap)
    except Exception:
        groups = []
    itercc_scores = _compute_itercc_scores(graph, candidate_nodes, groups)

    ranked_by_itercc = sorted(range(n_cands),
                              key=lambda i: itercc_scores[i], reverse=True)

    baseline_cost  = ga_proxy_fitness(graph, [], N, grouping_cap)
    sorted_scores  = sorted(itercc_scores, reverse=True)
    max_itercc_sum = sum(sorted_scores[:max_neurons])
    _eps = 1e-12

    def _biased_fitness(ind: list[int]) -> float:
        nodes       = [candidate_nodes[i] for i in ind]
        proxy       = ga_proxy_fitness(graph, nodes, N, grouping_cap)
        proxy_norm  = proxy / (baseline_cost + _eps)
        itercc_sum  = sum(itercc_scores[i] for i in ind)
        itercc_norm = itercc_sum / (max_itercc_sum + _eps)
        return alpha * proxy_norm - beta * itercc_norm

    def _mutate(ind: list[int]) -> list[int]:
        child = list(ind)
        used  = set(child)
        for idx in range(len(child)):
            if rng.random() < mutation_rate:
                unused = [i for i in range(n_cands) if i not in used]
                if unused:
                    used.discard(child[idx])
                    child[idx] = rng.choice(unused)
                    used.add(child[idx])
        return sorted(child)

    # ---- Lamarckian local search on the BIASED landscape ----
    def _local_search(ind: list[int]) -> list[int]:
        child          = list(ind)
        current_biased = _biased_fitness(child)

        for _pass in range(local_search_iters):
            improved = False
            for gene_pos in range(len(child)):
                in_chrom = set(child)
                in_chrom.discard(child[gene_pos])

                candidates_to_try: list[int] = []
                for top_idx in ranked_by_itercc:
                    if top_idx not in in_chrom and top_idx != child[gene_pos]:
                        candidates_to_try.append(top_idx)
                        if len(candidates_to_try) >= local_search_candidates:
                            break

                best_swap        = None
                best_swap_biased = current_biased
                for cand in candidates_to_try:
                    trial           = list(child)
                    trial[gene_pos] = cand
                    trial_biased    = _biased_fitness(trial)
                    if trial_biased < best_swap_biased:
                        best_swap        = cand
                        best_swap_biased = trial_biased

                if best_swap is not None:
                    child[gene_pos] = best_swap
                    current_biased  = best_swap_biased
                    improved        = True

            if not improved:
                break

        return sorted(child)

    if debug:
        print(f"    [GA-Memetic-Biased] alpha={alpha}  beta={beta}  "
              f"local_search_iters={local_search_iters}  "
              f"local_search_candidates={local_search_candidates}  "
              f"baseline_cost={baseline_cost:.2f}  "
              f"max_itercc_sum={max_itercc_sum:.4f}")

    winner = _ga_core(
        graph, candidate_nodes, max_neurons, N, grouping_cap,
        pop_size, n_generations, tournament_k, mutation_rate, elite_frac,
        rng, _biased_fitness, _mutate, None, debug, label="GA-Memetic-Biased",
        local_search_fn=_local_search,
    )
    if winner is None:
        return 0, []

    best_proxy = ga_proxy_fitness(
        graph, [candidate_nodes[i] for i in winner], N, grouping_cap)
    if debug:
        print(f"    [GA-Memetic-Biased] Winner proxy={best_proxy:.4f}")
    return _prune_winner(net, candidate_nodes, winner, best_proxy)


# ==================================================================
# PSO neuron pruner (placeholder)
# ==================================================================

def pso_neuron_step(
    net: "PrunableNetwork",
    *,
    max_neurons: int,
    N: int,
    grouping_cap: float = 10_000.0,
    low_mag_percentile: float = 90.0,
    n_particles: int = 30,
    n_iterations: int = 50,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    pso_seed: int = 0,
    debug: bool = False,
    **_ignored,
) -> tuple[int, list]:
    """PSO-based neuron pruner — not yet implemented."""
    raise NotImplementedError(
        "pso_neuron_step is not yet implemented. "
        "Use ga_neuron_step or its variants for now."
    )


# ==================================================================
# ACO edge pruner (placeholder)
# ==================================================================

def aco_edge_step(
    net: "PrunableNetwork",
    *,
    amount: float,
    N: int,
    grouping_cap: float = 10_000.0,
    n_ants: int = 20,
    n_iterations: int = 30,
    alpha: float = 1.0,
    beta: float = 2.0,
    rho: float = 0.1,
    aco_seed: int = 0,
    debug: bool = False,
    **_ignored,
) -> None:
    """ACO-based edge pruner — not yet implemented."""
    raise NotImplementedError(
        "aco_edge_step is not yet implemented. "
        "Use _bridge_edge_step for now."
    )
