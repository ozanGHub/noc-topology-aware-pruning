"""
main_LeNet_300_100_e2e_AccuracyBudget_Metaheuristic.py
=======================================================
Extends the accuracy-budget pruning comparison with metaheuristic branches.

This file is a drop-in extension of main_LeNet_300_100_e2e_AccuracyBudget.py.
All existing branches (A–G) are preserved unchanged so results are directly
comparable.  Branches M - V are added after Branch G.

Metaheuristic neuron strategies live in pruning_meta.py so they can be
reused across LeNet, MLP, Autoencoder, and future models without copy-paste.

GA design (details in pruning_meta.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
• Chromosome  : set of exactly max_neurons neuron indices from the Stage-1
                magnitude-filtered candidate pool.
• Fitness     : fast proxy — graph copy → indegree_grouping → sum |weight|
                of cross-group edges.  No fine-tuning, no SA mapping.
• Selection   : tournament (k=3) + elitism (top 10%).
• Crossover   : set-based uniform.
• Mutation    : random gene swap with probability GA_MUTATION_RATE.

Pipeline overview
-----------------
Step 1  — Train / load the base model.
Step 2  — Baseline evaluation.
Step 3  — 5 × 5% iterative unstructured L1 edge pre-pruning (25% total).
Step 4  — Record L1-checkpoint (= tolerance anchor).
Step 5  — Run branches independently from the L1 checkpoint:
    A : IterCC Neuron + Bridge-Link     (heuristic, primary baseline)
    B : IterCC Neuron + L2 edge
    #C : IterCC Neuron + L1 edge
    D : Taylor Neuron + L2 edge         (best classical)
    #E : Taylor Neuron + L1 edge
    #G : Taylor Neuron + Bridge-Link
    #M : GA Neuron + Bridge-Link         (metaheuristic — vs A) - GA classical implementation + heuristic edge step
    #N : GA Neuron + L2 edge             (metaheuristic — vs B) - GA classical implementation + classical L2 edge step
    #O : GA Biased + Bridge              (metaheuristic — vs A)
    #P : GA Guided + Bridge              (metaheuristic — vs A)
    Q : GA Biased v2 + Bridge           (metaheuristic v2 — vs A) - higher beta + adaptive re-scoring
    #R : GA Biased v2 + L2 Edge          (metaheuristic v2 — vs B)
    #S : GA Biased v3 + Bridge           (metaheuristic v3 — vs A) - β annealing + faster re-scoring
    #    β annealing: β(gen) = 0.5 − 0.2 × gen / (n_gens − 1) → starts at 0.5, ends at 0.3; α = 1 − β at every generation
    #T : GA Guided v2 + Bridge              (metaheuristic — vs A) - improved guided mutation to fix premature convergence in Branch P
    #U : GA Memetic + Bridge                (metaheuristic — vs M) - Lamarckian local search after crossover/mutation
    V : GA Memetic Biased + Bridge         (metaheuristic — vs Q) - biased fitness + local search on biased landscape
    W : GA Memetic Biased + L2             (metaheuristic — vs R) - same neuron as V, L2 edge pruning
    V2: GA Memetic Biased + Bridge (small) (metaheuristic — vs V) - reduced post-neuron bridge step (0.01 vs 0.05)
    #Q2: GA CC_Biased v2 + Bridge (small)   (metaheuristic — vs Q) - same neuron as Q, post-neuron step=0.01
    #S2: GA CC_Biased v3 + Bridge (small)   (metaheuristic — vs S) - same neuron as S, post-neuron step=0.01
    #V3: GA Memetic Biased + Bridge (xsmall)(metaheuristic — vs V2) - same neuron as V2, post-neuron step=0.005
Step 6  — Two summary tables (ISO-ACCURACY and FINAL).
"""

from __future__ import annotations

import copy
import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from DNN_LeNet_300_100 import load_or_train, fine_tune, evaluate_model
from prunable_network import PrunableNetwork
from eval_helper import eval_noc, eval_nn
from mapping_helpers import build_wctg_structural, compute_structural_cost
from pruning import iterative_commcost_neuron_pruning
from pruning_meta import (
    ga_neuron_step,
    ga_neuron_step_biased,
    ga_neuron_step_biased_v2,
    ga_neuron_step_biased_v3,
    ga_neuron_step_guided_mutation,
    ga_neuron_step_guided_mutation_v2,
    ga_neuron_step_memetic,
    ga_neuron_step_memetic_biased,
    ga_proxy_fitness,
)

# ==================================================================
# Configuration — shared with original branches
# ==================================================================

data_dir:     str   = "./data"
model_path:   str   = "DNN_LeNet_300_100.pth"
seed:         int   = 42
test_size:    float = 0.2
batch_size:   int   = 64
lr:           float = 1e-3
epochs:       int   = 200
weight_decay: float = 1e-4
threshold:    float = 0.5
force_train:  bool  = False

# --- Shared L1 pre-pruning ---
l1_target: float = 0.25
l1_steps:  int   = 5
ft_epochs: int   = 10

# --- Accuracy-budget settings ---
ACCURACY_TOLERANCE:         float = 0.01   # 1.0% hard stop
SNAPSHOT_INTERVAL:          float = 0.005  # snapshot every 0.5%
MICRO_FT_EPOCHS:            int   = 5
NEURON_STEPS_PER_EDGE_STEP: int   = 3
NEURONS_PER_STEP:           int   = 3
EDGE_STEP_FRACTION:         float = 0.02
BRIDGE_STEP_FRACTION:       float = 0.02
MAX_STEPS:                  int   = 1000
NEURON_FIRST_FRACTION:      float = 0.15

# IterCC parameters (Branches A/B/C)
iterCC_low_mag_pct:   float = 90.0
iterCC_degree_q:      float = 0.25
iterCC_min_in_degree: int   = 0

# NoC mapping parameters
P:         int = 36
N:         int = 36
mesh_rows: int = int(math.sqrt(P))
mesh_cols: int = int(math.sqrt(P))

# ==================================================================
# Configuration — GA Neuron Pruner (Branches M and N)
# ==================================================================

# Population size: number of candidate pruning sets per generation.
GA_POP_SIZE: int = 30

# Generations per micro-step.  30 individuals × 20 gens = 600 proxy
# evaluations per step; each proxy eval < 1 ms on LeNet-300-100.
GA_N_GENERATIONS: int = 20

# Tournament pool size.  Higher k → stronger selection pressure.
GA_TOURNAMENT_K: int = 3

# Per-gene mutation probability.
GA_MUTATION_RATE: float = 0.2

# Elitism: top fraction carried unchanged to the next generation.
GA_ELITE_FRAC: float = 0.1

# Independent RNG seed for the GA (decoupled from main training seed).
GA_SEED: int = 0

# Set True to print per-generation best fitness during GA runs.
GA_DEBUG: bool = False

# --- Branch O: biased-fitness GA ---
# α weights the proxy cost term; β weights the IterCC domain-knowledge term.
# α + β = 1 is recommended.  Higher β → more influence from IterCC scoring.
GA_ALPHA: float = 0.7   # weight for cross-group proxy cost
GA_BETA:  float = 0.3   # weight for IterCC neuron score

# --- Branch P: guided-mutation GA ---
# Fraction of the initial population seeded from IterCC's greedy ranking.
# 0.0 = fully random start (degenerates to plain GA).
# 1.0 = all individuals are IterCC variants (very little random exploration).
GA_WARM_START_FRAC: float = 0.5

# --- Branch Q: biased GA v2 (higher beta + adaptive re-scoring) ---
# Higher beta gives more influence to the IterCC domain-knowledge term.
# rescore_interval: re-compute IterCC scores every N GA generations on the
# best individual's pruned graph, so the bias adapts as the GA converges.
GA_ALPHA_Q:          float = 0.6   # weight for cross-group proxy cost
GA_BETA_Q:           float = 0.4   # weight for IterCC neuron score (vs 0.3 in O)
GA_RESCORE_INTERVAL: int   = 5     # re-score IterCC every 5 generations

# --- Branch R: GA Biased v2 + L2 edge ---
# Same neuron strategy as Q (ga_neuron_step_biased_v2), different edge method.
# R vs Q isolates the edge method (L2 vs bridge-link) within biased-v2 GA.
# R vs B isolates the neuron method (GA biased-v2 vs IterCC) within L2 edge.
# Uses the same _ga_biased_v2_kwargs as Q — no new GA hyperparameters needed.

# --- Branch S: biased GA v3 (β annealing + faster re-scoring) ---
# Two refinements over Q:
#   1. β annealing: start β=0.5 (strong IterCC guidance early) → end β=0.3
#      (proxy-driven late).  α = 1 - β at each generation.
#   2. rescore_interval=3 (fires 6-7 times in 20 gens vs 4 times for Q).
GA_BETA_S_START:      float = 0.5  # initial β (high IterCC influence)
GA_BETA_S_END:        float = 0.3  # final β   (more proxy-driven)
GA_RESCORE_INTERVAL_S: int  = 3    # re-score every 3 generations

# --- Branch T: improved guided-mutation GA (fixes Branch P's premature convergence) ---
# Two fixes over Branch P:
#   1. warm_start_frac=0.10: only 3 of 30 individuals seeded from IterCC
#      (vs 15 of 30 in P), leaving 27 random individuals for diversity.
#   2. guided_mutation_prob=0.50: each triggered mutation has a 50% chance
#      of being IterCC-directed vs random (vs 100% deterministic in P).
# Proxy fitness unchanged (same as P and M) — only search operators differ.
GA_WARM_START_FRAC_T:    float = 0.10   # fraction of pop from IterCC (vs 0.50 in P)
GA_GUIDED_MUTATION_PROB: float = 0.50   # prob of IterCC-directed swap per mutation

# --- Branch U: Memetic GA (Lamarckian local search after crossover/mutation) ---
# After each offspring is created by crossover + mutation, a greedy hill-climbing
# step refines the chromosome before it enters the next generation.
# Local search uses IterCC scores to select WHICH candidates to try, and the
# proxy fitness to decide WHETHER to accept each swap.  This is orthogonal to:
#   - Biased fitness (O/Q/S): knowledge injected into selection pressure.
#   - Guided mutation (P/T): knowledge injected into mutation direction.
#   Memetic injects knowledge into individual refinement (post-crossover).
# Parameters:
#   local_search_iters      = 2  passes of hill climbing per offspring
#   local_search_candidates = 5  top-IterCC candidates to try per gene per pass
# Cost: ≤ 2 × 3 × 5 = 30 extra proxy evals per offspring (vs 1 in plain GA).
GA_LOCAL_SEARCH_ITERS:      int = 2   # hill-climbing passes per offspring
GA_LOCAL_SEARCH_CANDIDATES: int = 5   # top-K IterCC candidates per gene per pass
# Branches V / W reuse GA_ALPHA_Q (0.6), GA_BETA_Q (0.4) for the biased fitness
# and GA_LOCAL_SEARCH_ITERS / GA_LOCAL_SEARCH_CANDIDATES for the local search.

# Branch V2: reduced bridge-link step fraction to spread budget over more steps.
# Problem with V (0.05): after effective neuron removal only ~1,500 cross-group
# edges survive; bridge-link targets ALL of them in one 5%-of-total-edges shot,
# exhausting the 1% accuracy budget immediately (same failure as P/T/U).
# Fix: reduce to 0.01 so each step removes ~2,000 edges instead of ~10,000,
# allowing the pipeline to iterate many neuron + edge micro-steps before stopping.
BRIDGE_POST_NEURON_SMALL:  float = 0.01   # V2, Q2, S2 — spread budget over ~10-20 steps
BRIDGE_POST_NEURON_XSMALL: float = 0.005  # V3 — push even further, ~20-40 steps

# ==================================================================
# Metrics tracker
# ==================================================================

stages:           list[dict] = []
trajectory:       list[dict] = []
_anchor_accuracy: float      = 0.0


def _traj_record(
    branch: str,
    step:   int,
    phase:  str,
    acc:    float,
    drop:   float,
    net:    "PrunableNetwork",
) -> None:
    """Append one lightweight trajectory row after every micro-step."""
    try:
        from grouping_heuristic import indegree_grouping as _ig
        groups = _ig(net.graph, N, 10_000.0)
        node_to_grp: dict = {}
        for gid, members in enumerate(groups):
            for nd in members:
                node_to_grp[nd] = gid

        n_cross = 0
        w_cross = 0.0
        linears = [m for m in net.model.modules() if isinstance(m, nn.Linear)]
        for li, layer in enumerate(linears):
            w = layer.weight.data
            nz_r, nz_c = (w != 0).nonzero(as_tuple=True)
            for r, c in zip(nz_r.tolist(), nz_c.tolist()):
                sg = node_to_grp.get((li,     c))
                dg = node_to_grp.get((li + 1, r))
                if sg is not None and dg is not None and sg != dg:
                    n_cross += 1
                    w_cross += float(abs(w[r, c].item()))
    except Exception:
        n_cross, w_cross = -1, -1.0

    layers  = [m for m in net.model.modules() if isinstance(m, nn.Linear)]
    n_edges = sum(int((m.weight.data != 0).sum().item()) for m in layers)

    trajectory.append({
        "branch":            branch,
        "step":              step,
        "phase":             phase,
        "accuracy":          acc,
        "accuracy_drop":     drop,
        "neurons":           _count_hidden_neurons(net.model),
        "edges":             n_edges,
        "cross_group_edges": n_cross,
        "cross_group_w_sum": round(w_cross, 4),
        "flops":             compute_model_flops(net.model),
    })


# ==================================================================
# Helper utilities
# ==================================================================

def _extract_accuracy(report) -> float:
    if isinstance(report, (int, float)):
        return float(report)
    for attr in ("accuracy", "acc", "test_accuracy", "score"):
        if hasattr(report, attr):
            return float(getattr(report, attr))
    if hasattr(report, "__getitem__"):
        for key in ("accuracy", "acc", "test_accuracy", "score"):
            try:
                return float(report[key])
            except (KeyError, TypeError):
                pass
    return float(report)


def _extract_inference_time(report) -> dict:
    if not hasattr(report, "batch_ms"):
        return {"batch_ms": float("nan")}
    return {"batch_ms": float(report.batch_ms)}


def _extract_metrics(report) -> dict:
    result = {"f1": float("nan"), "precision": float("nan"),
              "recall": float("nan"), "state_size_kb": float("nan")}
    for attr in result:
        if hasattr(report, attr):
            result[attr] = float(getattr(report, attr))
    return result


def compute_model_flops(model: nn.Module) -> int:
    total_macs = 0
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            total_macs += mod.in_features * mod.out_features
    return 2 * total_macs


def _count_hidden_neurons(model: nn.Module) -> int:
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return sum(m.out_features for m in linears[:-1])


def reapply_zero_masks(net: PrunableNetwork) -> None:
    for mod in net.model.modules():
        if isinstance(mod, nn.Linear):
            if hasattr(mod, "weight_orig"):
                prune.remove(mod, "weight")
            mask = (mod.weight.data != 0).float()
            if mask.all():
                continue
            prune.custom_from_mask(mod, "weight", mask)


def _noc_and_record(name: str, net: PrunableNetwork,
                    X_test, y_test, device, timings) -> None:
    """Run NoC mapping on current net and append one row to `stages`."""
    report  = evaluate_model(net.model, X_test, y_test, device=device, timings=timings)
    acc     = _extract_accuracy(report)
    inf     = _extract_inference_time(report)
    metrics = _extract_metrics(report)

    noc = eval_noc(net.graph, P=P, N=N,
                   mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                   title=name, visualize=False)

    sp_wctg = build_wctg_structural(net.graph, noc.groups, net.model, eps=1e-8)
    sp_cost = compute_structural_cost(sp_wctg, noc.spg)

    stages.append({
        "name":          name,
        "accuracy":      acc,
        "f1":            metrics["f1"],
        "precision":     metrics["precision"],
        "recall":        metrics["recall"],
        "cost":          float(noc.cost),
        "sp_cost":       sp_cost,
        "flops":         compute_model_flops(net.model),
        "edges":         net.graph.number_of_edges(),
        "neurons":       net.graph.number_of_nodes(),
        "batch_ms":      inf["batch_ms"],
        "state_size_kb": metrics["state_size_kb"],
    })
    print(f"  [recorded] '{name}'  acc={acc:.4f}  "
          f"cost={noc.cost:.0f}  sp_cost={sp_cost:.0f}  "
          f"edges={net.graph.number_of_edges():,}  "
          f"neurons={_count_hidden_neurons(net.model)}")


# ==================================================================
# Neuron / edge pruning helpers
# ==================================================================

def _replace_module(model: nn.Module, old: nn.Module, new: nn.Module) -> None:
    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if child is old:
                setattr(parent, child_name, new)
                return


def physically_remove_zeroed_neurons(model: nn.Module) -> None:
    while True:
        linears      = [m for m in model.modules() if isinstance(m, nn.Linear)]
        hidden       = linears[:-1]
        replaced_any = False
        for i, layer_out in enumerate(hidden):
            row_norms = layer_out.weight.data.abs().sum(dim=1)
            live_idx  = (row_norms > 0).nonzero(as_tuple=False).squeeze(1)
            if live_idx.numel() == layer_out.out_features:
                continue
            dev, dtype = layer_out.weight.device, layer_out.weight.dtype
            new_out = nn.Linear(layer_out.in_features, live_idx.numel(),
                                bias=layer_out.bias is not None, dtype=dtype, device=dev)
            with torch.no_grad():
                new_out.weight.copy_(layer_out.weight[live_idx])
                if layer_out.bias is not None:
                    new_out.bias.copy_(layer_out.bias[live_idx])
            _replace_module(model, layer_out, new_out)

            linears_now   = [m for m in model.modules() if isinstance(m, nn.Linear)]
            layer_in_next = linears_now[i + 1]
            new_in = nn.Linear(live_idx.numel(), layer_in_next.out_features,
                               bias=layer_in_next.bias is not None, dtype=dtype, device=dev)
            with torch.no_grad():
                new_in.weight.copy_(layer_in_next.weight[:, live_idx])
                if layer_in_next.bias is not None:
                    new_in.bias.copy_(layer_in_next.bias)
            _replace_module(model, layer_in_next, new_in)
            replaced_any = True
            break
        if not replaced_any:
            break


def prune_neurons_taylor(model: nn.Module, amount: float,
                         X_calib, y_calib, device,
                         n_batches: int = 10, batch_sz: int = 128) -> None:
    if isinstance(X_calib, np.ndarray):
        X_calib = torch.from_numpy(X_calib).float()
    if isinstance(y_calib, np.ndarray):
        y_calib = torch.from_numpy(y_calib).long()

    linears    = [m for m in model.modules() if isinstance(m, nn.Linear)]
    hidden     = linears[:-1]
    importance = {id(m): torch.zeros(m.out_features) for m in hidden}

    was_training = model.training
    model.train()
    criterion = nn.CrossEntropyLoss()
    n = X_calib.shape[0]

    for b in range(n_batches):
        idx_b = torch.randperm(n, generator=torch.Generator().manual_seed(b))[:batch_sz]
        xb, yb = X_calib[idx_b].to(device), y_calib[idx_b].to(device)
        model.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        with torch.no_grad():
            for layer in hidden:
                if layer.weight.grad is not None:
                    importance[id(layer)] += (
                        layer.weight.data * layer.weight.grad
                    ).abs().sum(dim=1).cpu()

    model.zero_grad()
    if not was_training:
        model.eval()

    for layer in hidden:
        importance[id(layer)] /= n_batches

    all_scores = torch.cat([importance[id(m)] for m in hidden])
    n_prune    = max(1, int(round(amount * all_scores.numel())))
    threshold  = torch.topk(all_scores, n_prune, largest=False).values.max()

    for layer in hidden:
        mask = (importance[id(layer)] > threshold).float().to(layer.weight.device)
        layer.weight.data *= mask.unsqueeze(1)
        if layer.bias is not None:
            layer.bias.data *= mask


def prune_neurons_structured_l1(model: nn.Module, amount: float) -> None:
    linears   = [m for m in model.modules() if isinstance(m, nn.Linear)]
    hidden    = linears[:-1]
    all_norms = torch.cat([m.weight.data.abs().sum(dim=1) for m in hidden])
    n_prune   = max(1, int(round(amount * all_norms.numel())))
    threshold = torch.topk(all_norms, n_prune, largest=False).values.max()
    for layer in hidden:
        norms = layer.weight.data.abs().sum(dim=1)
        mask  = (norms > threshold).float().to(layer.weight.device)
        layer.weight.data *= mask.unsqueeze(1)
        if layer.bias is not None:
            layer.bias.data *= mask


def prune_l2_global(model: nn.Module, amount: float) -> None:
    layers     = [mod for mod in model.modules() if isinstance(mod, nn.Linear)]
    zero_masks = {mod: (mod.weight.data == 0.0) for mod in layers}
    all_scores = torch.cat([(mod.weight.data ** 2).view(-1) for mod in layers])
    all_zero   = torch.cat([zero_masks[mod].view(-1) for mod in layers])

    n_total        = all_scores.numel()
    n_already_zero = int(all_zero.sum().item())
    n_prune_active = max(0, int(round(amount * n_total)) - n_already_zero)
    if n_prune_active <= 0:
        return

    active_scores  = all_scores[~all_zero]
    n_prune_active = min(n_prune_active, active_scores.numel())
    threshold      = torch.topk(active_scores, n_prune_active, largest=False).values.max()

    with torch.no_grad():
        n_done       = 0
        layer_scores = {mod: (mod.weight.data ** 2) for mod in layers}
        for mod in layers:
            below = (~zero_masks[mod]) & (layer_scores[mod] < threshold)
            mod.weight.data[below] = 0.0
            n_done += int(below.sum().item())
        remaining = n_prune_active - n_done
        if remaining > 0:
            for mod in layers:
                if remaining <= 0:
                    break
                tied = (
                    (~zero_masks[mod])
                    & (layer_scores[mod] == threshold)
                    & (mod.weight.data != 0.0)
                )
                for idx in tied.nonzero(as_tuple=False)[:remaining]:
                    mod.weight.data[tuple(idx)] = 0.0
                remaining -= int(min(remaining, tied.sum().item()))


def _prune_l1_global_by_amount(model: nn.Module, amount: float) -> None:
    PrunableNetwork(model, eps=0.0).prune_links_l1_global(amount=amount)


def _compute_edge_prune_amount(net: PrunableNetwork, target_edge_count: int) -> float:
    layers        = [mod for mod in net.model.modules() if isinstance(mod, nn.Linear)]
    total_weights = sum(m.weight.numel() for m in layers)
    already_zero  = sum(int((m.weight.data == 0).sum().item()) for m in layers)
    active_edges  = total_weights - already_zero
    edges_to_prune = max(0, active_edges - target_edge_count)
    if total_weights == 0:
        return 0.0
    return float((already_zero + edges_to_prune) / total_weights)


# ==================================================================
# Neuron-step wrappers
# Signature: fn(net, *, max_neurons, **kwargs) -> (n_removed, history)
# ==================================================================

def _taylor_neuron_step(net: PrunableNetwork, *, max_neurons: int,
                        X_calib, y_calib, device,
                        n_batches: int = 10, batch_sz: int = 128) -> tuple[int, list]:
    before   = _count_hidden_neurons(net.model)
    fraction = max_neurons / max(before, 1)
    prune_neurons_taylor(net.model, fraction, X_calib=X_calib, y_calib=y_calib,
                         device=device, n_batches=n_batches, batch_sz=batch_sz)
    physically_remove_zeroed_neurons(net.model)
    net.commit_pruning()
    net.resync_graph()
    return before - _count_hidden_neurons(net.model), []


def _l1_neuron_step(net: PrunableNetwork, *, max_neurons: int) -> tuple[int, list]:
    before   = _count_hidden_neurons(net.model)
    fraction = max_neurons / max(before, 1)
    prune_neurons_structured_l1(net.model, fraction)
    physically_remove_zeroed_neurons(net.model)
    net.commit_pruning()
    net.resync_graph()
    return before - _count_hidden_neurons(net.model), []


# ==================================================================
# Edge-step wrappers
# Signature: fn(net, *, amount, **kwargs) -> None
# ==================================================================

def _l2_edge_step(net: PrunableNetwork, *, amount: float, **_) -> None:
    prune_l2_global(net.model, amount)


def _l1_edge_step(net: PrunableNetwork, *, amount: float, **_) -> None:
    _prune_l1_global_by_amount(net.model, amount)


def _bridge_edge_step(net: PrunableNetwork, *, amount: float, **_) -> None:
    """Granular cross-group edge pruner — same logic as Branch A."""
    noc_tmp = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                       title="_bridge_tmp", visualize=False)
    groups = noc_tmp.groups

    node_to_group: dict = {}
    for gid, members in enumerate(groups):
        for node in members:
            node_to_group[node] = gid

    layers        = [m for m in net.model.modules() if isinstance(m, nn.Linear)]
    active_now    = sum(int((m.weight.data != 0).sum().item()) for m in layers)
    total_weights = sum(m.weight.numel() for m in layers)
    already_zero  = total_weights - active_now
    edges_to_prune = max(1, int(round(amount * total_weights)) - already_zero)
    edges_to_prune = min(edges_to_prune, active_now - 1)

    if edges_to_prune <= 0:
        return

    linears: list[nn.Linear] = [m for m in net.model.modules() if isinstance(m, nn.Linear)]
    bridge_candidates: list[tuple[float, int, int, int]] = []

    for li, layer in enumerate(linears):
        w = layer.weight.data
        nz_rows, nz_cols = (w != 0).nonzero(as_tuple=True)
        for r, c in zip(nz_rows.tolist(), nz_cols.tolist()):
            src_grp = node_to_group.get((li,     c))
            dst_grp = node_to_group.get((li + 1, r))
            if src_grp is None or dst_grp is None or src_grp == dst_grp:
                continue
            bridge_candidates.append((float(w[r, c].abs()), li, r, c))

    if not bridge_candidates:
        prune_l2_global(net.model, amount)
        net.commit_pruning()
        net.resync_graph()
        return

    bridge_candidates.sort(key=lambda x: x[0])
    with torch.no_grad():
        for _, li, r, c in bridge_candidates[:edges_to_prune]:
            linears[li].weight.data[r, c] = 0.0

    net.remove_lonely_neurons()
    net.commit_pruning()
    net.resync_graph()


# ==================================================================
# Core accuracy-budget loop
# ==================================================================

def run_accuracy_budget_branch(
    label: str,
    l1_model_copy: nn.Module,
    anchor_accuracy: float,
    X_train, y_train,
    X_test, y_test,
    device,
    timings,
    *,
    neuron_step_fn: Callable,
    neuron_step_kwargs: dict,
    edge_step_fn: Callable,
    edge_step_kwargs: dict,
    accuracy_tolerance:         float       = ACCURACY_TOLERANCE,
    snapshot_interval:          float       = SNAPSHOT_INTERVAL,
    edge_step_fraction:         float       = EDGE_STEP_FRACTION,
    post_neuron_edge_fraction:  float|None  = None,
    neuron_steps_per_edge_step: int         = NEURON_STEPS_PER_EDGE_STEP,
    micro_ft_epochs:            int         = MICRO_FT_EPOCHS,
    neuron_tolerance:           float|None  = None,
    neuron_first_steps:         int         = 0,
    neurons_per_step:           int         = 1,
) -> None:
    floor             = anchor_accuracy - accuracy_tolerance
    _post_neuron_frac = (post_neuron_edge_fraction
                         if post_neuron_edge_fraction is not None
                         else edge_step_fraction)
    _n_per_edge       = max(0, neuron_steps_per_edge_step)
    _neuron_tol       = (neuron_tolerance
                         if neuron_tolerance is not None
                         else accuracy_tolerance)

    if snapshot_interval and snapshot_interval > 0:
        n_snaps    = max(1, int(round(accuracy_tolerance / snapshot_interval)))
        thresholds = [round(snapshot_interval * (i + 1), 8) for i in range(n_snaps)]
    else:
        thresholds = []

    print("\n" + "=" * 70)
    print(f"  BRANCH: {label}")
    print(f"  Anchor={anchor_accuracy:.4f}  Tolerance={accuracy_tolerance:.4f}  "
          f"Floor={floor:.4f}")
    if neuron_tolerance is not None:
        print(f"  Neuron phase cap: {_neuron_tol:.4f}  "
              f"Edge phase gets remaining headroom up to {accuracy_tolerance:.4f}")
    if neuron_first_steps > 0:
        print(f"  Neuron-first phase: {neuron_first_steps} pure neuron steps "
              f"before interleaving")
    if thresholds:
        print(f"  Snapshot thresholds (drop): "
              f"{[f'{t*100:.1f}%' for t in thresholds]}")
    print(f"  Edge fraction (interleaved/post-neuron): "
          f"{edge_step_fraction:.4f} / {_post_neuron_frac:.4f}  "
          f"Neuron:Edge ratio={_n_per_edge}:1  "
          f"FT epochs={micro_ft_epochs}  Max steps={MAX_STEPS}")
    print("=" * 70)

    net        = PrunableNetwork(copy.deepcopy(l1_model_copy), eps=0.0)
    best_model = copy.deepcopy(net.model)
    best_acc   = _extract_accuracy(
        evaluate_model(net.model, X_test, y_test, device=device, timings=timings))
    print(f"[{label}] Starting accuracy: {best_acc:.4f}")

    next_snap_idx        = 0
    total_steps          = 0
    neuron_steps_done    = 0
    neuron_exhausted     = False
    edge_exhausted       = False
    neuron_tol_exhausted = False

    # ------------------------------------------------------------------
    # Phase 0 — optional pure neuron-first steps before interleaving
    # ------------------------------------------------------------------
    if neuron_first_steps > 0:
        print(f"[{label}] === Neuron-first phase: up to {neuron_first_steps} steps ===")
        for nf_step in range(1, neuron_first_steps + 1):
            pre_step_model = copy.deepcopy(net.model)
            kwargs         = dict(neuron_step_kwargs)
            kwargs["max_neurons"] = neurons_per_step
            n_removed, _   = neuron_step_fn(net, **kwargs)

            if n_removed == 0:
                neuron_exhausted = True
                net._model = pre_step_model
                net.resync_graph()
                print(f"[{label}] Neuron-first: exhausted at step {nf_step}. "
                      f"Entering interleaved loop.")
                break

            net.remove_lonely_neurons()
            reapply_zero_masks(net)
            fine_tune(net.model, X_train, y_train, device=device, lr=lr,
                      epochs=micro_ft_epochs, seed=seed + nf_step)
            net.commit_pruning()
            net.resync_graph()

            acc  = _extract_accuracy(
                evaluate_model(net.model, X_test, y_test, device=device, timings=timings))
            drop = anchor_accuracy - acc
            neuron_steps_done += 1
            total_steps       += 1

            print(f"[{label}] NF-step {nf_step:3d}: "
                  f"acc={acc:.4f}  drop={drop:+.4f}  "
                  f"neurons={_count_hidden_neurons(net.model)}  "
                  f"edges={net.graph.number_of_edges():,}")
            _traj_record(label, nf_step, "neuron-first", acc, drop, net)

            if drop <= accuracy_tolerance:
                best_model = copy.deepcopy(net.model)
                best_acc   = acc
                while (next_snap_idx < len(thresholds)
                       and drop >= thresholds[next_snap_idx]):
                    snap_pct   = thresholds[next_snap_idx] * 100
                    snap_label = f"{label} @ drop≤{snap_pct:.1f}%"
                    print(f"[{label}] → Snapshot at drop≤{snap_pct:.1f}%")
                    _noc_and_record(snap_label, net, X_test, y_test, device, timings)
                    next_snap_idx += 1
            else:
                print(f"[{label}] Neuron-first: tolerance exceeded at step {nf_step}. "
                      f"Reverting and entering interleaved loop.")
                net._model = best_model
                net.resync_graph()
                break

        print(f"[{label}] === Neuron-first phase complete. "
              f"Neurons: {_count_hidden_neurons(net.model)}  "
              f"Entering interleaved loop. ===")

    # ------------------------------------------------------------------
    # Phase 1 — interleaved neuron + edge loop
    # ------------------------------------------------------------------
    for step in range(1, MAX_STEPS + 1):
        if neuron_exhausted and edge_exhausted:
            print(f"[{label}] Both phases exhausted at step {step}. Stopping.")
            break

        if _n_per_edge == 0:
            do_neuron = False
        else:
            do_neuron = (
                not neuron_exhausted
                and not neuron_tol_exhausted
                and (neuron_steps_done % (_n_per_edge + 1) != _n_per_edge)
            )
        if edge_exhausted:
            do_neuron = True
        if neuron_exhausted or neuron_tol_exhausted:
            do_neuron = False

        neuron_phase_done = neuron_exhausted or neuron_tol_exhausted
        active_edge_frac  = (edge_step_fraction
                             if not neuron_phase_done else _post_neuron_frac)

        pre_step_model = copy.deepcopy(net.model)

        # ---- Apply micro-step ----
        if do_neuron:
            kwargs = dict(neuron_step_kwargs)
            kwargs["max_neurons"] = neurons_per_step
            n_removed, _ = neuron_step_fn(net, **kwargs)
            if n_removed == 0:
                neuron_exhausted = True
                net._model       = pre_step_model
                net.resync_graph()
                current_drop       = anchor_accuracy - best_acc
                remaining_headroom = accuracy_tolerance - current_drop
                print(f"[{label}] Step {step}: neuron phase exhausted. "
                      f"Remaining headroom: {remaining_headroom:.4f} "
                      f"({remaining_headroom*100:.2f}%). "
                      f"Switching to edge-only (step={_post_neuron_frac:.4f}).")
                continue
            net.remove_lonely_neurons()
            neuron_steps_done += 1
            phase_label = "neuron"
        else:
            layers_now     = [m for m in net.model.modules() if isinstance(m, nn.Linear)]
            current_edges  = sum(int((m.weight.data != 0).sum().item()) for m in layers_now)
            edges_to_prune = max(1, int(current_edges * active_edge_frac))
            target_edges   = current_edges - edges_to_prune
            if target_edges <= 0:
                edge_exhausted = True
                print(f"[{label}] Step {step}: edge phase exhausted (no edges left).")
                continue
            amount = _compute_edge_prune_amount(net, target_edges)
            if not (0.0 < amount < 1.0):
                edge_exhausted = True
                _tw = sum(p.numel() for name, p in net.model.named_parameters()
                          if "weight" in name and p.dim() == 2)
                _ae = net.graph.number_of_edges()
                print(f"[{label}] Step {step}: edge phase exhausted "
                      f"(amount={amount:.6f}, total_weights={_tw}, "
                      f"active_edges={_ae}).")
                continue
            edge_step_fn(net, amount=amount, **edge_step_kwargs)
            net.commit_pruning()
            net.resync_graph()
            phase_label = "edge" if not neuron_phase_done else "edge*"

        # ---- Fine-tune ----
        reapply_zero_masks(net)
        fine_tune(net.model, X_train, y_train, device=device, lr=lr,
                  epochs=micro_ft_epochs, seed=seed + step)
        net.commit_pruning()
        net.resync_graph()

        # ---- Evaluate ----
        acc  = _extract_accuracy(
            evaluate_model(net.model, X_test, y_test, device=device, timings=timings))
        drop = anchor_accuracy - acc
        print(f"[{label}] Step {step:3d} ({phase_label:6s}): "
              f"acc={acc:.4f}  drop={drop:+.4f}  "
              f"neurons={_count_hidden_neurons(net.model)}  "
              f"edges={net.graph.number_of_edges():,}")

        _traj_record(label, total_steps, phase_label, acc, drop, net)
        total_steps += 1

        if drop <= accuracy_tolerance:
            best_model = copy.deepcopy(net.model)
            best_acc   = acc

            if (not neuron_tol_exhausted
                    and neuron_tolerance is not None
                    and drop >= _neuron_tol):
                neuron_tol_exhausted = True
                remaining = accuracy_tolerance - drop
                print(f"[{label}] Step {step}: neuron tolerance cap reached "
                      f"(drop={drop:.4f} ≥ neuron_tol={_neuron_tol:.4f}). "
                      f"Reserving remaining {remaining:.4f} ({remaining*100:.2f}%) "
                      f"for edge-only phase (step={_post_neuron_frac:.4f}).")

            while (next_snap_idx < len(thresholds)
                   and drop >= thresholds[next_snap_idx]):
                snap_pct   = thresholds[next_snap_idx] * 100
                snap_label = f"{label} @ drop≤{snap_pct:.1f}%"
                print(f"[{label}] → Snapshot at drop≤{snap_pct:.1f}%: '{snap_label}'")
                _noc_and_record(snap_label, net, X_test, y_test, device, timings)
                next_snap_idx += 1

        else:
            print(f"[{label}] Tolerance exceeded (drop={drop:.4f} > "
                  f"{accuracy_tolerance:.4f}). Reverting to last good state.")
            net._model = best_model
            net.resync_graph()
            break

    # ---- Record final state ----
    final_label = f"{label} [final]"
    print(f"\n[{label}] Finished. {total_steps} micro-steps completed.")
    print(f"  Final: acc={best_acc:.4f}  "
          f"neurons={_count_hidden_neurons(net.model)}  "
          f"edges={net.graph.number_of_edges():,}")
    print(f"\n=== Final evaluation: {label} ===")
    eval_nn(net.model, X_test=X_test, y_test=y_test, device=device,
            timings=timings, evaluate_fn=evaluate_model)
    _noc_and_record(final_label, net, X_test, y_test, device, timings)


# ==================================================================
# Summary tables
# ==================================================================

def _fmt_delta(cur, bv, spec=".4f"):
    if bv == 0:
        return f"{cur:>10}"
    pct  = (cur - bv) / bv * 100
    sign = "+" if pct >= 0 else ""
    return f"{cur:>10{spec}}  ({sign}{pct:.2f}%)"

def _fmt_int(cur, bv):
    if bv == 0:
        return f"{cur:>10}"
    pct  = (cur - bv) / bv * 100
    sign = "+" if pct >= 0 else ""
    return f"{cur:>10,}  ({sign}{pct:.2f}%)"

def _fmt_ms(cur, bv):
    if bv == 0 or math.isnan(bv):
        return f"{cur:>10.2f} ms"
    pct  = (cur - bv) / bv * 100
    sign = "+" if pct >= 0 else ""
    return f"{cur:>7.2f} ms  ({sign}{pct:.2f}%)"

def _fmt_kb(cur, bv):
    if bv == 0 or math.isnan(bv):
        return f"{cur:>8.2f} KB"
    pct  = (cur - bv) / bv * 100
    sign = "+" if pct >= 0 else ""
    return f"{cur:>5.2f} KB  ({sign}{pct:.2f}%)"


def _print_table(title_lines: list[str],
                 row_stages:  list[dict],
                 base_stage:  dict) -> None:
    w_name = max(len("Stage"), *(len(s["name"]) for s in row_stages)) + 2
    w_val  = 22
    n_cols = 11
    sep    = "+" + "-" * w_name + "+" + ("-" * w_val + "+") * n_cols
    width  = w_name + w_val * n_cols + n_cols + 1

    print("\n\n" + "=" * width)
    for line in title_lines:
        print("  " + line)
    print("=" * width)
    print(sep)
    print(f"|{'Stage':^{w_name}}|{'Accuracy':^{w_val}}|"
          f"{'Comm. Cost (WCTG n2n)':^{w_val}}|"
          f"{'Cross Group Comm. Cost':^{w_val}}|"
          f"{'Edges':^{w_val}}|{'Neurons':^{w_val}}|"
          f"{'Batch Inf. (ms)':^{w_val}}|{'FLOPs':^{w_val}}|"
          f"{'Model Size (KB)':^{w_val}}|"
          f"{'F1':^{w_val}}|{'Precision':^{w_val}}|{'Recall':^{w_val}}|")
    print(sep)

    b = base_stage
    for s in row_stages:
        is_b   = (s is b)
        tag    = "  (baseline)" if is_b else ""
        acc_s  = f'{s["accuracy"]:>10.4f}{tag}'       if is_b else _fmt_delta(s["accuracy"],  b["accuracy"])
        cost_s = f'{s["cost"]:>10.4f}{tag}'            if is_b else _fmt_delta(s["cost"],      b["cost"])
        spc_s  = f'{s["sp_cost"]:>10.4f}{tag}'         if is_b else _fmt_delta(s["sp_cost"],   b["sp_cost"])
        edge_s = f'{s["edges"]:>10,}{tag}'              if is_b else _fmt_int(s["edges"],       b["edges"])
        neur_s = f'{s["neurons"]:>10,}{tag}'            if is_b else _fmt_int(s["neurons"],     b["neurons"])
        bat_s  = f'{s["batch_ms"]:>7.2f} ms{tag}'      if is_b else _fmt_ms(s["batch_ms"],     b["batch_ms"])
        flop_s = f'{s["flops"]:>10,}{tag}'              if is_b else _fmt_int(s["flops"],       b["flops"])
        size_s = f'{s["state_size_kb"]:>5.2f} KB{tag}' if is_b else _fmt_kb(s["state_size_kb"],b["state_size_kb"])
        f1_s   = f'{s["f1"]:>10.4f}{tag}'              if is_b else _fmt_delta(s["f1"],         b["f1"])
        prec_s = f'{s["precision"]:>10.4f}{tag}'       if is_b else _fmt_delta(s["precision"],  b["precision"])
        rec_s  = f'{s["recall"]:>10.4f}{tag}'          if is_b else _fmt_delta(s["recall"],     b["recall"])

        print(f"|{s['name']:<{w_name}}|{acc_s:>{w_val}}|{cost_s:>{w_val}}|"
              f"{spc_s:>{w_val}}|{edge_s:>{w_val}}|{neur_s:>{w_val}}|"
              f"{bat_s:>{w_val}}|{flop_s:>{w_val}}|{size_s:>{w_val}}|"
              f"{f1_s:>{w_val}}|{prec_s:>{w_val}}|{rec_s:>{w_val}}|")
        print(sep)
    print()


def save_trajectories(path_csv: str = "trajectory.csv",
                      path_json: str = "trajectory.json") -> None:
    if not trajectory:
        print("[trajectory] Nothing to save — trajectory list is empty.")
        return

    import csv, json

    fields = list(trajectory[0].keys())
    with open(path_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trajectory)
    print(f"[trajectory] Saved {len(trajectory)} rows → {path_csv}")

    with open(path_json, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"[trajectory] Saved {len(trajectory)} rows → {path_json}")


def print_summary_tables() -> None:
    if not stages:
        return

    base = stages[0]

    # ----------------------------------------------------------------
    # Table 1 — ISO-ACCURACY: one snapshot row per branch × threshold
    # ----------------------------------------------------------------
    snap_rows = [s for s in stages if "@ drop≤" in s["name"]]
    if snap_rows:
        _print_table(
            title_lines=[
                "ISO-ACCURACY TABLE — snapshot at each accuracy-drop threshold",
                "Model: LeNet 784-300-100-10 | Dataset: MNIST",
                f"Tolerance: ≤{ACCURACY_TOLERANCE*100:.1f}% drop from BASELINE"
                f"  | Snapshot interval: every {SNAPSHOT_INTERVAL*100:.1f}%",
                "A=IterCC+Bridge  B=IterCC+L2  C=IterCC+L1  "
                "D=Taylor+L2  E=Taylor+L1  G=Taylor+Bridge  "
                "M=GA+Bridge  N=GA+L2  "
                "O=GA-CC_Biased+Bridge  P=GA-CC_Guided+Bridge  "
                "Q=GA-CC_Biased-v2+Bridge  R=GA-CC_Biased-v2+L2  "
                "S=GA-CC_Biased-v3+Bridge  T=GA-CC_Guided-v2+Bridge  "
                "U=GA-Memetic+Bridge  V=GA-Memetic-Biased+Bridge  "
                "W=GA-Memetic-Biased+L2  V2=GA-Memetic-Biased+Bridge(small)  "
                "Q2=GA-CC_Biased-v2+Bridge(small)  S2=GA-CC_Biased-v3+Bridge(small)  "
                "V3=GA-Memetic-Biased+Bridge(xsmall)",
                "Each row = model state the first time drop crossed that threshold.",
                "Deltas relative to Baseline.",
            ],
            row_stages=[base] + snap_rows,
            base_stage=base,
        )
    else:
        print("\n[No intermediate snapshots recorded "
              "(SNAPSHOT_INTERVAL may be None or too large)]")

    # ----------------------------------------------------------------
    # Table 2 — FINAL: last within-tolerance state per branch
    # ----------------------------------------------------------------
    anchor_row = next(
        (s for s in stages if s["name"] == "After L1 25% pre-pruning"), None)
    final_rows = [s for s in stages if s["name"].endswith("[final]")]
    fixed_rows = [base] + ([anchor_row] if anchor_row else []) + final_rows

    _print_table(
        title_lines=[
            "FINAL TABLE — each branch at its stopping point (last within tolerance)",
            "Model: LeNet 784-300-100-10 | Dataset: MNIST",
            f"Tolerance: ≤{ACCURACY_TOLERANCE*100:.1f}% drop from BASELINE",
            "A=IterCC+Bridge  B=IterCC+L2  C=IterCC+L1  "
            "D=Taylor+L2  E=Taylor+L1  G=Taylor+Bridge  "
            "M=GA+Bridge  N=GA+L2  "
            "O=GA-CC_Biased+Bridge  P=GA-Guided+Bridge  "
            "Q=GA-CC_Biased-v2+Bridge  R=GA-CC_Biased-v2+L2  "
            "S=GA-CC_Biased-v3+Bridge  T=GA-CC_Guided-v2+Bridge  "
            "U=GA-Memetic+Bridge  V=GA-Memetic-Biased+Bridge  "
            "W=GA-Memetic-Biased+L2  V2=GA-Memetic-Biased+Bridge(small)  "
            "Q2=GA-CC_Biased-v2+Bridge(small)  S2=GA-CC_Biased-v3+Bridge(small)  "
            "V3=GA-Memetic-Biased+Bridge(xsmall)",
            "Deltas relative to Baseline.",
        ],
        row_stages=fixed_rows,
        base_stage=base,
    )


# ==================================================================
# 1. Train / load
# ==================================================================

model, scaler, (X_train, X_test, y_train, y_test), timings, device = load_or_train(
    data_dir, model_path,
    seed=seed, test_size=test_size, batch_size=batch_size,
    lr=lr, epochs=epochs, weight_decay=weight_decay,
    threshold=threshold, force_train=force_train,
    patience=15, min_delta=1e-4,
)

net = PrunableNetwork(model, eps=0.0)

# ==================================================================
# 2. Baseline evaluation
# ==================================================================

print("\n=== Initial model ===")
eval_nn(net.model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
_noc_and_record("Baseline", net, X_test, y_test, device, timings)

# ==================================================================
# 3. Iterative unstructured L1 edge pre-pruning  (5 × 5% = 25%)
# ==================================================================

original_edge_count = net.graph.number_of_edges()
target_removals     = round(original_edge_count * l1_target)
total_weights       = sum(p.numel() for name, p in net.model.named_parameters()
                          if "weight" in name and p.dim() == 2)

for step in range(1, l1_steps + 1):
    print(f"L1 loop: {step}")
    already_removed   = original_edge_count - net.graph.number_of_edges()
    remaining         = target_removals - already_removed
    cumulative_target = already_removed + round(remaining / (l1_steps - step + 1))
    amount            = cumulative_target / total_weights

    if amount >= 1.0:
        print(f"  [Step {step}/{l1_steps}] Skipping — not enough edges remaining.")
        break

    net.prune_links_l1_global(amount=amount)
    fine_tune(net.model, X_train, y_train, device=device, lr=lr,
              epochs=ft_epochs, seed=seed + step)

    removed = original_edge_count - net.graph.number_of_edges()
    print(f"  [Step {step}/{l1_steps}] Edges removed: {removed}/{original_edge_count} "
          f"({removed / original_edge_count:.1%})")

net.commit_pruning()
net.remove_lonely_neurons()

print(f"\n=== After iterative L1 pruning ({l1_target:.0%} removed) ===")
eval_nn(net.model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
_noc_and_record("After L1 25% pre-pruning", net, X_test, y_test, device, timings)

# ==================================================================
# 4. Anchor accuracy + checkpoint
# ==================================================================

l1_model_copy   = copy.deepcopy(net.model)
l1_neuron_count = _count_hidden_neurons(net.model)
l1_edge_count   = net.graph.number_of_edges()

anchor_accuracy  = stages[0]["accuracy"]
_anchor_accuracy = anchor_accuracy

print(f"\n[Checkpoint] L1 model: {l1_neuron_count} neurons, {l1_edge_count:,} edges.")
print(f"[Checkpoint] Tolerance anchor : BASELINE accuracy = {anchor_accuracy:.4f}")
print(f"[Checkpoint] Hard stop floor  : {anchor_accuracy - ACCURACY_TOLERANCE:.4f}")
print(f"[Checkpoint] iterCC_low_mag_pct = {iterCC_low_mag_pct}  "
      f"(permissive gate — CommCost score drives Stage-2 selection)")
if SNAPSHOT_INTERVAL:
    n_snaps = int(ACCURACY_TOLERANCE / SNAPSHOT_INTERVAL)
    floors  = [f"{anchor_accuracy - SNAPSHOT_INTERVAL*(i+1):.4f}"
               for i in range(n_snaps)]
    print(f"[Checkpoint] Snapshot floors  : {', '.join(floors)}")
print(f"\n[Checkpoint] GA settings: "
      f"pop={GA_POP_SIZE}  gens={GA_N_GENERATIONS}  "
      f"tournament_k={GA_TOURNAMENT_K}  mutation_rate={GA_MUTATION_RATE}  "
      f"elite_frac={GA_ELITE_FRAC}  seed={GA_SEED}")

# Shared kwargs for IterCC branches (A/B/C).
_iterCC_kwargs = dict(
    N=N, grouping_cap=10_000.0,
    low_mag_percentile=iterCC_low_mag_pct,
    degree_quantile=iterCC_degree_q,
    min_in_degree=iterCC_min_in_degree,
    min_score=0.0, debug=False,
)
# Shared kwargs for Taylor branches (D/E/G).
_taylor_kwargs = dict(
    X_calib=X_train, y_calib=y_train, device=device,
    n_batches=10, batch_sz=128,
)
# GA kwargs — base settings shared by all GA branches (M / N / O / P).
# Uses same Stage-1 filter and grouping settings as IterCC for fair comparison.
_ga_kwargs = dict(
    N=N, grouping_cap=10_000.0,
    low_mag_percentile=iterCC_low_mag_pct,
    pop_size=GA_POP_SIZE,
    n_generations=GA_N_GENERATIONS,
    tournament_k=GA_TOURNAMENT_K,
    mutation_rate=GA_MUTATION_RATE,
    elite_frac=GA_ELITE_FRAC,
    ga_seed=GA_SEED,
    debug=GA_DEBUG,
)
# Branch O: biased fitness — adds alpha/beta on top of base GA settings.
_ga_biased_kwargs = dict(**_ga_kwargs, alpha=GA_ALPHA, beta=GA_BETA)
# Branch P: guided mutation — adds warm_start_frac on top of base GA settings.
_ga_guided_kwargs = dict(**_ga_kwargs, warm_start_frac=GA_WARM_START_FRAC)
# Branch Q/R: biased GA v2 — higher beta + adaptive IterCC re-scoring.
# R reuses the same neuron kwargs as Q; only the edge step function differs.
_ga_biased_v2_kwargs = dict(**_ga_kwargs, alpha=GA_ALPHA_Q, beta=GA_BETA_Q,
                            rescore_interval=GA_RESCORE_INTERVAL)
# Branch S: biased GA v3 — β annealing + faster re-scoring.
_ga_biased_v3_kwargs = dict(**_ga_kwargs,
                            beta_start=GA_BETA_S_START,
                            beta_end=GA_BETA_S_END,
                            rescore_interval=GA_RESCORE_INTERVAL_S)
# Branch T: probabilistic guided mutation + minimal warm-start.
_ga_guided_v2_kwargs = dict(**_ga_kwargs,
                            warm_start_frac=GA_WARM_START_FRAC_T,
                            guided_mutation_prob=GA_GUIDED_MUTATION_PROB)
# Branch U: Memetic GA — Lamarckian local search (IterCC-guided, proxy-confirmed).
_ga_memetic_kwargs = dict(**_ga_kwargs,
                          local_search_iters=GA_LOCAL_SEARCH_ITERS,
                          local_search_candidates=GA_LOCAL_SEARCH_CANDIDATES)
# Branches V / W: Memetic + Biased — biased fitness (Q's α/β) + local search on
# biased landscape.  W uses the same neuron kwargs as V; only edge method differs.
_ga_memetic_biased_kwargs = dict(**_ga_kwargs,
                                 alpha=GA_ALPHA_Q, beta=GA_BETA_Q,
                                 local_search_iters=GA_LOCAL_SEARCH_ITERS,
                                 local_search_candidates=GA_LOCAL_SEARCH_CANDIDATES)

# ==================================================================
# 5. Branch A — IterCC Neuron + Bridge-Link  (heuristic, primary)
# ==================================================================

run_accuracy_budget_branch(
    label="A: IterCC Neuron + Bridge-Link",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=iterative_commcost_neuron_pruning,
    neuron_step_kwargs=_iterCC_kwargs,
    edge_step_fn=_bridge_edge_step,
    edge_step_kwargs={},
    edge_step_fraction=BRIDGE_STEP_FRACTION,
    post_neuron_edge_fraction=0.05,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)

# ==================================================================
# 6. Branch B — IterCC Neuron + L2 edge
# ==================================================================

run_accuracy_budget_branch(
    label="B: IterCC Neuron + L2 edge",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=iterative_commcost_neuron_pruning,
    neuron_step_kwargs=_iterCC_kwargs,
    edge_step_fn=_l2_edge_step,
    edge_step_kwargs={},
    neuron_tolerance=0.005,
    post_neuron_edge_fraction=0.05,
    neurons_per_step=3,
    micro_ft_epochs=2,
)

# ==================================================================
# 7. Branch C — IterCC Neuron + L1 edge
# ==================================================================
"""
run_accuracy_budget_branch(
    label="C: IterCC Neuron + L1 edge",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=iterative_commcost_neuron_pruning,
    neuron_step_kwargs=_iterCC_kwargs,
    edge_step_fn=_l1_edge_step,
    edge_step_kwargs={},
    neuron_tolerance=0.005,
    post_neuron_edge_fraction=0.05,
    neurons_per_step=3,
    micro_ft_epochs=2,
)
"""
# ==================================================================
# 8. Branch D — Taylor Neuron + L2 edge  (best classical)
# ==================================================================

run_accuracy_budget_branch(
    label="D: Taylor Neuron + L2 edge",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=_taylor_neuron_step,
    neuron_step_kwargs=_taylor_kwargs,
    edge_step_fn=_l2_edge_step,
    edge_step_kwargs={},
    neuron_steps_per_edge_step=1,
    post_neuron_edge_fraction=0.05,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)

# ==================================================================
# 9. Branch E — Taylor Neuron + L1 edge
# ==================================================================
"""
run_accuracy_budget_branch(
    label="E: Taylor Neuron + L1 edge",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=_taylor_neuron_step,
    neuron_step_kwargs=_taylor_kwargs,
    edge_step_fn=_l1_edge_step,
    edge_step_kwargs={},
    neuron_steps_per_edge_step=1,
    post_neuron_edge_fraction=0.05,
)
"""
# ==================================================================
# 10. Branch F — L1 Neuron + L2 edge  (classical baseline, disabled)
# ==================================================================
"""
run_accuracy_budget_branch(
    label="F: L1 Neuron + L2 edge",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=_l1_neuron_step,
    neuron_step_kwargs={},
    edge_step_fn=_l2_edge_step,
    edge_step_kwargs={},
    neuron_steps_per_edge_step=1,
    post_neuron_edge_fraction=0.05,
)
"""

# ==================================================================
# 11. Branch G — Taylor Neuron + Bridge-Link
# ==================================================================

run_accuracy_budget_branch(
    label="G: Taylor Neuron + Bridge-Link",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=_taylor_neuron_step,
    neuron_step_kwargs=_taylor_kwargs,
    edge_step_fn=_bridge_edge_step,
    edge_step_kwargs={},
    edge_step_fraction=BRIDGE_STEP_FRACTION,
    post_neuron_edge_fraction=0.05,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)

# ==================================================================
# 12. Branch M — GA Neuron + Bridge-Link  (metaheuristic)
#
# Controlled comparison with Branch A (IterCC + Bridge-Link):
#   - Same edge method      : bridge-link (_bridge_edge_step)
#   - Same accuracy budget  : neuron_tolerance=0.005, tolerance=0.01
#   - Same interleave ratio : neuron_steps_per_edge_step=1
#   - Same neurons per step : neurons_per_step=3
#   - Same FT epochs        : micro_ft_epochs=2
#   - Same Stage-1 filter   : low_mag_percentile=90.0
#
# Only the neuron SELECTION differs:
#   Branch A — greedy IterCC (one-at-a-time, cross-group edge score)
#   Branch M — GA jointly selects which 3 neurons to remove each step
#
# Any difference in NoC cost isolates the neuron selection quality.
# ==================================================================
"""
run_accuracy_budget_branch(
    label="M: GA Neuron + Bridge-Link",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=ga_neuron_step,
    neuron_step_kwargs=_ga_kwargs,
    edge_step_fn=_bridge_edge_step,
    edge_step_kwargs={},
    edge_step_fraction=BRIDGE_STEP_FRACTION,
    post_neuron_edge_fraction=0.05,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)
"""
# ==================================================================
# 13. Branch N — GA Neuron + L2 Edge  (metaheuristic)
#
# Controlled comparison with Branch B (IterCC + L2 edge):
#   - Same edge method      : L2 global (_l2_edge_step)
#   - Same accuracy budget  : neuron_tolerance=0.005, tolerance=0.01
#   - Same interleave ratio : neuron_steps_per_edge_step=1
#   - Same neurons per step : neurons_per_step=3
#   - Same FT epochs        : micro_ft_epochs=2
#   - Same Stage-1 filter   : low_mag_percentile=90.0
#
# Cross-comparisons enabled by the M/N pair:
#   M vs A  — GA vs IterCC neuron selection (bridge-link edge, same)
#   N vs B  — GA vs IterCC neuron selection (L2 edge, same)
#   M vs N  — bridge-link vs L2 edge pruning (GA neuron, same)
#   A vs B  — bridge-link vs L2 edge pruning (IterCC neuron, same)
# ==================================================================
"""
run_accuracy_budget_branch(
    label="N: GA Neuron + L2 edge",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=ga_neuron_step,
    neuron_step_kwargs=_ga_kwargs,
    edge_step_fn=_l2_edge_step,
    edge_step_kwargs={},
    post_neuron_edge_fraction=0.05,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)
"""
# ==================================================================
# 14. Branch O — GA Comm. Cost Heuristic Biased Fitness + Bridge-Link  (metaheuristic)
#
# Controlled comparison with Branch M (plain GA + Bridge-Link).
# The only change is the fitness function:
#   Branch M — fitness = proxy cost only (cross-group |w| sum)
#   Branch O — fitness = α × proxy_norm − β × itercc_norm
#              (combines proxy measurement with IterCC domain knowledge)
#
# α=0.7, β=0.3 gives 70% weight to the measured cost and 30% to the
# heuristic guidance.  A chromosome that picks high-IterCC-score neurons
# gets a fitness bonus even before the proxy confirms it.
# ==================================================================
"""
run_accuracy_budget_branch(
    label="O: GA CC_Biased Fitness + Bridge-Link",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=ga_neuron_step_biased,
    neuron_step_kwargs=_ga_biased_kwargs,
    edge_step_fn=_bridge_edge_step,
    edge_step_kwargs={},
    edge_step_fraction=BRIDGE_STEP_FRACTION,
    post_neuron_edge_fraction=0.05,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)
"""
# ==================================================================
# 15. Branch P — GA Comm. Cost Heuristic Guided Mutation + Bridge-Link  (metaheuristic)
#
# Controlled comparison with Branch M (plain GA + Bridge-Link).
# Two changes from the plain GA:
#   1. Guided mutation: swaps a gene for the highest-IterCC-scoring
#      unused candidate instead of a random one.
#   2. Warm-start: 50% of initial population seeded from IterCC's
#      greedy ranking; evolution begins near the heuristic's answer.
#
# The proxy fitness function is unchanged (same as Branch M) so
# any improvement comes purely from the smarter search strategy.
# ==================================================================
"""
run_accuracy_budget_branch(
    label="P: GA CC_Guided Mutation + Bridge-Link",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=ga_neuron_step_guided_mutation,
    neuron_step_kwargs=_ga_guided_kwargs,
    edge_step_fn=_bridge_edge_step,
    edge_step_kwargs={},
    edge_step_fraction=BRIDGE_STEP_FRACTION,
    post_neuron_edge_fraction=0.05,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)
"""
# ==================================================================
# 16. Branch Q — GA Comm. Cost Heuristic Biased Fitness v2 + Bridge-Link  (metaheuristic)
#
# Controlled comparison with Branch O (GA Biased + Bridge-Link).
# Two targeted refinements over Branch O:
#
#   1. Higher beta (0.4 vs 0.3):
#      Increases the weight of IterCC domain knowledge in the composite
#      fitness.  Branch O's result showed the bias term is beneficial;
#      amplifying it may push selection quality closer to IterCC.
#
#   2. Adaptive IterCC re-scoring (every 5 generations):
#      After every 5 GA generations the IterCC scores are recomputed on
#      a pruned graph copy with the current best individual's neurons
#      removed.  This updates the bias term to reflect which *remaining*
#      candidates are most valuable given where the GA is converging —
#      a form of iterated domain-knowledge refinement.
#
#      At generation 0: scores on full graph (identical to Branch O).
#      At generation 5: scores on graph minus current best neurons.
#      At generation 10: scores on graph minus new best neurons.
#      ...and so on until generation 20.
#
# Same edge method, budget, interleave ratio, FT epochs as O for
# controlled comparison.  Any difference isolates the two Q refinements.
# ==================================================================

run_accuracy_budget_branch(
    label="Q: GA CC_Biased v2 + Bridge-Link",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=ga_neuron_step_biased_v2,
    neuron_step_kwargs=_ga_biased_v2_kwargs,
    edge_step_fn=_bridge_edge_step,
    edge_step_kwargs={},
    edge_step_fraction=BRIDGE_STEP_FRACTION,
    post_neuron_edge_fraction=0.05,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)

# ==================================================================
# 17. Branch R — Comm. Cost Heuristic GA Biased v2 + L2 Edge  (metaheuristic)
#
# Controlled comparison with Branch Q (GA Biased v2 + Bridge-Link).
# The only change is the edge pruning method:
#   Branch Q — bridge-link (_bridge_edge_step)
#   Branch R — L2 global  (_l2_edge_step)
#
# Cross-comparisons enabled by adding R:
#   R vs Q  — L2 vs bridge-link edge (GA biased-v2 neuron fixed)
#   R vs B  — GA biased-v2 vs IterCC neuron (L2 edge fixed)
#   R vs N  — GA biased-v2 vs plain GA neuron (L2 edge fixed)
#   Q vs O  — biased-v2 vs biased-v1 neuron (bridge-link fixed)
# ==================================================================
"""
run_accuracy_budget_branch(
    label="R: GA CC_Biased v2 + L2 edge",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=ga_neuron_step_biased_v2,
    neuron_step_kwargs=_ga_biased_v2_kwargs,
    edge_step_fn=_l2_edge_step,
    edge_step_kwargs={},
    post_neuron_edge_fraction=0.05,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)
"""
# ==================================================================
# 18. Branch S — GA Comm. Cost Heuristic Biased v3 + Bridge-Link  (metaheuristic)
#
# Controlled comparison with Branch Q (GA Biased v2 + Bridge-Link).
# Two refinements over Q:
#
#   1. β annealing (targets the iso-accuracy / early-step gap):
#      β decreases from 0.5 (generation 0) to 0.3 (final generation).
#      α = 1 − β at each generation (no separate α parameter).
#      High β early: strong IterCC guidance → quickly reach good regions.
#      Low β late:  proxy-driven selection → fine-grained cost search.
#
#   2. Faster adaptive re-scoring (rescore_interval=3 vs 5 for Q):
#      IterCC scores are refreshed every 3 generations (6–7 times in
#      20 generations, vs 4 times for Q), so the bias term adapts more
#      quickly as the population converges each micro-step.
#
# Same edge method, budget, interleave ratio, and FT epochs as Q.
# Any difference isolates the two S refinements.
# ==================================================================
"""
run_accuracy_budget_branch(
    label="S: GA CC_Biased v3 + Bridge-Link",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=ga_neuron_step_biased_v3,
    neuron_step_kwargs=_ga_biased_v3_kwargs,
    edge_step_fn=_bridge_edge_step,
    edge_step_kwargs={},
    edge_step_fraction=BRIDGE_STEP_FRACTION,
    post_neuron_edge_fraction=0.05,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)
"""
# ==================================================================
# 19. Branch T — GA Comm. Cost Heuristic Guided Mutation v2 + Bridge-Link  (metaheuristic)
#
# Controlled comparison with Branch P (GA Guided Mutation + Bridge-Link).
# Fixes P's premature-convergence problem with two targeted weakening steps:
#
#   1. Minimal warm-start (warm_start_frac=0.10 vs 0.50 in P):
#      Only 3 of 30 individuals are seeded from IterCC's top-K ranking.
#      The remaining 27 start randomly, giving the GA enough diversity
#      for crossover to explore combinations that IterCC's greedy path
#      would never reach.
#
#   2. Probabilistic guided mutation (guided_mutation_prob=0.50 vs 1.0 in P):
#      When a gene is triggered for mutation (prob=mutation_rate=0.2),
#      a second coin-flip decides:
#        50% → IterCC-directed swap (highest-IterCC unused candidate)
#        50% → random swap (plain GA behaviour)
#      Average IterCC-directed moves per individual per generation:
#        0.2 × 0.5 × max_neurons ≈ 0.3  (vs 0.6 in P → half the pressure)
#
#   Proxy fitness unchanged from P (cross-group edge |weight| sum) —
#   only the search operators differ, so the comparison is controlled.
#
# Expected behaviour: T should use significantly more of the 1% accuracy
# budget (vs P's −0.02% final drop) while still benefiting from the
# IterCC-guided nudges in the mutation operator.
# ==================================================================
"""
run_accuracy_budget_branch(
    label="T: GA CC_Guided Mutation v2 + Bridge-Link",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=ga_neuron_step_guided_mutation_v2,
    neuron_step_kwargs=_ga_guided_v2_kwargs,
    edge_step_fn=_bridge_edge_step,
    edge_step_kwargs={},
    edge_step_fraction=BRIDGE_STEP_FRACTION,
    post_neuron_edge_fraction=0.05,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)
"""
# ==================================================================
# 20. Branch U — Memetic GA + Bridge-Link  (Lamarckian local search)
#
# Controlled comparison with Branch M (plain GA + Bridge-Link).
# The only addition is a Lamarckian local search step applied to each
# offspring after crossover + mutation, before it enters the population.
#
# Local search mechanics:
#   For up to local_search_iters passes:
#     For each gene in the chromosome (max_neurons=3):
#       - Try the top local_search_candidates unused candidates by IterCC rank.
#       - Evaluate the proxy fitness for each (IterCC guides WHERE to look;
#         proxy confirms WHETHER the swap is actually better for the batch).
#       - Accept the swap with the best proxy improvement, if any.
#   Each pass terminates early if no gene swap improved the proxy.
#
# This is orthogonal to all previous knowledge-injection points:
#   O/Q/S  — biased fitness        (selection pressure)
#   P/T    — guided mutation       (offspring generation direction)
#   U      — Lamarckian refinement (post-crossover individual improvement)
#
# Unlike P/T, diversity is preserved because crossover creates varied
# offspring BEFORE local search refines each one independently.
# Unlike biased fitness (O/Q/S), local search modifies chromosomes rather
# than reshaping the selection landscape — these mechanisms are complementary.
#
# Compute cost: ≤ 2 × 3 × 5 = 30 extra proxy evals per non-elite offspring
# (vs 1 per offspring in plain GA).  On LeNet-300-100: still <5 s per step.
# ==================================================================
"""
run_accuracy_budget_branch(
    label="U: GA Memetic + Bridge-Link",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=ga_neuron_step_memetic,
    neuron_step_kwargs=_ga_memetic_kwargs,
    edge_step_fn=_bridge_edge_step,
    edge_step_kwargs={},
    edge_step_fraction=BRIDGE_STEP_FRACTION,
    post_neuron_edge_fraction=0.05,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)
"""
# ==================================================================
# 21. Branch V — Memetic + Biased GA + Bridge-Link
#
# Controlled comparison with Branch Q (GA Biased v2 + Bridge-Link) and
# Branch U (plain Memetic + Bridge-Link).
#
# Fixes Branch U's premature convergence by changing the local search oracle:
#   Branch U — local search hill-climbs on pure proxy → IterCC's basin → collapse
#   Branch V — local search hill-climbs on BIASED fitness (α×proxy_norm − β×itercc_norm)
#              → biased landscape's local optima → different solutions per individual
#
# The biased landscape has a different basin structure from the pure proxy
# because the subtracted IterCC term reshapes it.  Each individual converges
# to a different local optimum of the biased landscape, preserving diversity.
#
# Two orthogonal knowledge-injection mechanisms combined:
#   Selection  (biased fitness)  — from Branch Q
#   Refinement (local search)    — from Branch U, now on biased oracle
#
# Controlled comparison enables:
#   V vs Q  — does memetic local search add value on top of biased fitness?
#   V vs U  — does switching oracle from proxy to biased fix convergence?
#   V vs A  — can Memetic + Biased match or beat the IterCC heuristic?
# ==================================================================

run_accuracy_budget_branch(
    label="V: GA Memetic Biased + Bridge-Link",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=ga_neuron_step_memetic_biased,
    neuron_step_kwargs=_ga_memetic_biased_kwargs,
    edge_step_fn=_bridge_edge_step,
    edge_step_kwargs={},
    edge_step_fraction=BRIDGE_STEP_FRACTION,
    post_neuron_edge_fraction=0.05,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)

# ==================================================================
# 22. Branch W — Memetic + Biased GA + L2 Edge
#
# Controlled comparison with Branch R (GA Biased v2 + L2 edge) and
# Branch V (Memetic + Biased + Bridge-Link).
#
# Same neuron strategy as V (ga_neuron_step_memetic_biased); only the
# edge pruning method changes.
#
# Cross-comparisons enabled:
#   W vs V  — L2 vs bridge-link edge (Memetic+Biased neuron fixed)
#   W vs R  — Memetic+Biased vs plain Biased-v2 neuron (L2 edge fixed)
#   W vs N  — Memetic+Biased vs plain GA neuron (L2 edge fixed)
# ==================================================================

run_accuracy_budget_branch(
    label="W: GA Memetic Biased + L2 edge",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=ga_neuron_step_memetic_biased,
    neuron_step_kwargs=_ga_memetic_biased_kwargs,
    edge_step_fn=_l2_edge_step,
    edge_step_kwargs={},
    post_neuron_edge_fraction=0.05,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)

# ==================================================================
# 23. Branch V2 — Memetic + Biased GA + Bridge-Link (small step)
#
# Controlled comparison with Branch V (same neuron + edge methods,
# different post_neuron_edge_fraction).
#
# Why V failed:
#   V used post_neuron_edge_fraction=0.05 → bridge-link targeted 5% of
#   all edges (~10,000 edges) per neuron step.  After effective Memetic+
#   Biased neuron removal only ~1,500 cross-group edges survive, so
#   bridge-link removes ALL of them in one shot, consuming the entire
#   accuracy budget in a single step.  Result: −81.76% cost, −0.02%
#   accuracy drop (same failure as P/T/U but caused by edge aggression,
#   not GA convergence).
#
# Fix:
#   Reduce post_neuron_edge_fraction to 0.01 (1% of edges, ~2,000 edges
#   per step).  This keeps each edge micro-step affordable in accuracy
#   terms, allowing 10-20 interleaved neuron + edge steps before the
#   budget is exhausted.
#
# Hypothesis:
#   If the fix works, V2 should show a similar final comm. cost to A or
#   better, confirming that V's failure was purely the edge step size,
#   not a GA convergence problem.  If V2 still stops early, the
#   neuron-selection GA itself is the cause.
# ==================================================================

run_accuracy_budget_branch(
    label="V2: GA Memetic Biased + Bridge-Link (small)",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=ga_neuron_step_memetic_biased,
    neuron_step_kwargs=_ga_memetic_biased_kwargs,
    edge_step_fn=_bridge_edge_step,
    edge_step_kwargs={},
    edge_step_fraction=BRIDGE_STEP_FRACTION,
    post_neuron_edge_fraction=BRIDGE_POST_NEURON_SMALL,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)

# ==================================================================
# 24. Branch Q2 — GA CC_Biased v2 + Bridge-Link (small step)
#
# Controlled comparison with Branch Q (same neuron function, smaller
# post_neuron_edge_fraction: 0.01 vs 0.05).
#
# Hypothesis: Q's biased neuron selection is genuinely different from
# IterCC's greedy answer.  With 0.05 the budget was consumed after ~N
# steps.  At 0.01 the pipeline runs ~5× more neuron+edge micro-steps,
# allowing Q's per-step advantage to compound further.
# Compare: Q2 vs Q (step size effect), Q2 vs V2 (biased-v2 vs memetic+biased).
# ==================================================================
"""
run_accuracy_budget_branch(
    label="Q2: GA CC_Biased v2 + Bridge-Link (small)",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=ga_neuron_step_biased_v2,
    neuron_step_kwargs=_ga_biased_v2_kwargs,
    edge_step_fn=_bridge_edge_step,
    edge_step_kwargs={},
    edge_step_fraction=BRIDGE_STEP_FRACTION,
    post_neuron_edge_fraction=BRIDGE_POST_NEURON_SMALL,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)
"""
# ==================================================================
# 25. Branch S2 — GA CC_Biased v3 + Bridge-Link (small step)
#
# Controlled comparison with Branch S (same neuron function, smaller
# post_neuron_edge_fraction: 0.01 vs 0.05).
#
# S wins at ISO-ACCURACY (best early-phase comm cost) because its
# high initial β guides the GA toward IterCC-preferred regions quickly.
# With a smaller step the pipeline runs more iterations — hypothesis:
# S's strong early guidance remains effective across more steps, giving
# it an advantage over Q2 at iso-accuracy AND a competitive final result.
# Compare: S2 vs S (step size), S2 vs Q2 (annealing vs fixed β at small step).
# ==================================================================
"""
run_accuracy_budget_branch(
    label="S2: GA CC_Biased v3 + Bridge-Link (small)",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=ga_neuron_step_biased_v3,
    neuron_step_kwargs=_ga_biased_v3_kwargs,
    edge_step_fn=_bridge_edge_step,
    edge_step_kwargs={},
    edge_step_fraction=BRIDGE_STEP_FRACTION,
    post_neuron_edge_fraction=BRIDGE_POST_NEURON_SMALL,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)
"""
# ==================================================================
# 26. Branch V3 — GA Memetic Biased + Bridge-Link (extra-small step)
#
# Controlled comparison with Branch V2 (same neuron function, smaller
# post_neuron_edge_fraction: 0.005 vs 0.01).
#
# V2 at 0.01 achieved −97.69% final comm cost with better model
# compression than A/Q (more neurons removed, better FLOPs/size/speed,
# only −0.76% accuracy drop).  V3 at 0.005 pushes further:
#   - ~2× more micro-steps before budget exhaustion
#   - Each bridge-link step removes ~1,000 edges vs ~2,000 for V2
#   - Hypothesis: even more neurons removed, deeper FLOPs/size reduction,
#     with similar or better final comm cost
# Compare: V3 vs V2 (step size effect within Memetic+Biased+Bridge).
# ==================================================================
"""
run_accuracy_budget_branch(
    label="V3: GA Memetic Biased + Bridge-Link (xsmall)",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=ga_neuron_step_memetic_biased,
    neuron_step_kwargs=_ga_memetic_biased_kwargs,
    edge_step_fn=_bridge_edge_step,
    edge_step_kwargs={},
    edge_step_fraction=BRIDGE_STEP_FRACTION,
    post_neuron_edge_fraction=BRIDGE_POST_NEURON_XSMALL,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)
"""
# ==================================================================
# 27. Summary
# ==================================================================

print_summary_tables()

save_trajectories(
    path_csv="trajectory_LeNet300_100_Metaheuristic.csv",
    path_json="trajectory_LeNet300_100_Metaheuristic.json"
)
