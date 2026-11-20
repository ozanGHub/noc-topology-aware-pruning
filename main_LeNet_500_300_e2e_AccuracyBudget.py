"""
main_LeNet_500_300_e2e_AccuracyBudget.py
=====================================
Accuracy-budget pruning comparison for LeNet 784-500-300-10 on MNIST.

Design rationale
----------------
The fixed-amount comparison forces every method to remove exactly the same
number of neurons and edges, neutralising the heuristic's core advantage.
This file instead fixes an ACCURACY_TOLERANCE and lets every method prune
iteratively until that tolerance is exhausted.  The comparison metric is:

    At the same worst-case accuracy level, which method achieves the
    lowest communication cost (WCTG n2n and cross-group)?

Pipeline overview
-----------------
Step 1  — Train / load the base model.
Step 2  — Baseline evaluation.
Step 3  — 5 × 5% iterative unstructured L1 edge pre-pruning (25% total).
Step 4  — Record L1-checkpoint accuracy (= tolerance anchor).
Step 5  — Run each branch independently from the L1 checkpoint:
    Branch A : IterativeCC Neuron + Bridge-Link  (heuristic, primary)
    Branch B : IterativeCC Neuron + L2 edge      (heuristic + classical edge)
    Branch C : IterativeCC Neuron + L1 edge      (heuristic + classical edge)
    Branch D : Taylor Neuron + L2 edge           (best classical)
    Branch E : Taylor Neuron + L1 edge           (classical)
    #Branch F : L1 Neuron + L2 edge               (classical baseline)
    Branch G : Taylor Neuron + Bridge-Link       (classical + heuristic edge)

    Method ranking (from prior individual experiments on this model)
    ------------------------------------------------------------------
    A = Comm.Cost Neuron + Bridge-Link 	(combined heuristic reference) 
    B = Comm.Cost Neuron + L2 Link 		(Best Neuron + Best Link - Heuristic Comm.Cost or Bridge-Link included) & (Best Neuron incl. + 2nd-Best Link)
    C = Comm.Cost Neuron + L1 Link  
    D = Taylor Neuron    + L2 Link      (Best Neuron + Best Link - Heuristic Comm.Cost or Bridge-Link excluded) & (2nd-Best Neuron + 2nd-Best Link)  
    E = Taylor Neuron    + L1 Link 
    G = Taylor Neuron + Bridge-Link     (2nd-Best Neuron + Best Link) 
        
Step 6  — Two summary tables:
    (a) ISO-ACCURACY table: snapshot at each SNAPSHOT_INTERVAL drop level.
    (b) FINAL table: each branch at its stopping point.

Accuracy-budget loop
--------------------
Each branch iterates micro-steps (one neuron or one edge batch per step).
After each step: fine-tune briefly → evaluate → check thresholds.

Snapshots are recorded at every SNAPSHOT_INTERVAL drop level (default 0.5%).
The branch stops hard when drop > ACCURACY_TOLERANCE (default 1%).
Both are top-level parameters easy to adjust without touching branch code.

Post-neuron exhaustion handling
--------------------------------
When the heuristic neuron filter finds no more candidates (Stage-1 filter
returns empty), the neuron phase is marked exhausted.  Rather than stopping,
the loop switches to an edge-only phase using `post_neuron_edge_fraction`
(default: EDGE_STEP_FRACTION = 2%), which is typically coarser than the
interleaved bridge step (0.5%).  This allows the remaining accuracy headroom
to be fully exploited by edge pruning after neurons are done.
"""

from __future__ import annotations

import copy
import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from DNN_LeNet_500_300 import load_or_train, fine_tune, evaluate_model
from prunable_network import PrunableNetwork
from eval_helper import eval_noc, eval_nn
from mapping_helpers import build_wctg_structural, compute_structural_cost
from pruning import iterative_commcost_neuron_pruning

# ==================================================================
# Configuration
# ==================================================================

data_dir:     str   = "./data"
model_path:   str   = "DNN_LeNet_500_300.pth"
seed:         int   = 42
test_size:    float = 0.2
batch_size:   int   = 64
lr:           float = 1e-3
epochs:       int   = 200
weight_decay: float = 1e-4
threshold:    float = 0.5
force_train:  bool  = True

# --- Shared L1 pre-pruning ---
l1_target: float = 0.25
l1_steps:  int   = 5
ft_epochs: int   = 10

# --- Accuracy-budget settings ---

# Hard stop: branch terminates when accuracy drop exceeds this value.
# The last checkpoint within tolerance is kept as the final result.
ACCURACY_TOLERANCE: float = 0.01   # 1.0%

# A snapshot (NoC mapping + stage record) is taken the first time
# the accuracy drop crosses each multiple of this interval.
# E.g. 0.005 → snapshots at 0.5% and 1.0% drops.
# Set to None to disable intermediate snapshots.
SNAPSHOT_INTERVAL: float = 0.005   # 0.5%

# Fine-tune epochs applied after each pruning micro-step.
MICRO_FT_EPOCHS: int = 5

# Neuron/edge interleaving ratio: N neuron steps per 1 edge step.
NEURON_STEPS_PER_EDGE_STEP: int = 3

# Number of neurons to attempt removing per neuron micro-step.
# Increase this if the neuron phase is being outpaced by the edge phase
# (e.g. Branch B finishing with almost no neurons removed).
NEURONS_PER_STEP: int = 3

# Fraction of remaining active edges removed per edge micro-step.
EDGE_STEP_FRACTION: float   = 0.02    # 2%  — L1 / L2 edge methods
BRIDGE_STEP_FRACTION: float = 0.02    # 2%  — granular cross-group pruner (safe at any sparsity)

# Safety ceiling on micro-steps per branch.
MAX_STEPS: int = 1000

# Neuron-first phase: run this fraction of the L1-checkpoint neuron count as
# pure neuron micro-steps before the interleaved loop begins.
# Computed at runtime as: round(l1_neuron_count * NEURON_FIRST_FRACTION / NEURONS_PER_STEP)
# Set to 0.0 to disable the neuron-first phase (default for all branches except B/C).
NEURON_FIRST_FRACTION: float = 0.15   # 15% of L1-checkpoint hidden neuron count

# IterativeCC neuron pruning settings.
# low_mag_pct raised to 90.0 so the magnitude gate is permissive and the
# CommCost score in Stage-2 does the real discrimination between candidates.
# The previous 75.0 excluded neurons with moderate magnitude but high
# comm. cost contribution — exactly the wrong ones to keep.
iterCC_low_mag_pct:   float = 90.0
iterCC_degree_q:      float = 0.25
iterCC_min_in_degree: int   = 0

# NoC mapping parameters.
P:         int = 64
N:         int = 64
mesh_rows: int = int(math.sqrt(P))
mesh_cols: int = int(math.sqrt(P))

# ==================================================================
# Metrics tracker
# ==================================================================

stages:      list[dict] = []   # receives baselines, snapshots, and finals
trajectory:  list[dict] = []   # one row per micro-step across all branches
_anchor_accuracy: float = 0.0  # set to BASELINE accuracy after step 2


def _traj_record(
    branch:      str,
    step:        int,
    phase:       str,       # "neuron-first" | "neuron" | "edge" | "edge*"
    acc:         float,
    drop:        float,
    net:         "PrunableNetwork",
) -> None:
    """
    Append one lightweight trajectory row — called after every micro-step.

    Avoids running the full SA mapping (expensive) at every step.
    Instead records fast structural proxies:
      - n_cross_group_edges  : number of non-zero weights that connect neurons
                               in different indegree groups (fresh grouping).
      - cross_group_w_sum    : sum of |w| for those edges  (≈ WCTG edge weight total)
      - flops                : 2 × Σ in_features × out_features  (exact, cheap)
      - neurons / edges      : counts from model weights

    The full SA comm. cost is only computed at snapshots and finals.
    """
    # Fast grouping for cross-group edge count.
    try:
        from grouping_heuristic import indegree_grouping as _ig
        groups = _ig(net.graph, N, 10_000.0)
        node_to_grp: dict = {}
        for gid, members in enumerate(groups):
            for nd in members:
                node_to_grp[nd] = gid

        n_cross   = 0
        w_cross   = 0.0
        linears   = [m for m in net.model.modules() if isinstance(m, nn.Linear)]
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
    flops   = compute_model_flops(net.model)

    trajectory.append({
        "branch":             branch,
        "step":               step,
        "phase":              phase,
        "accuracy":           acc,
        "accuracy_drop":      drop,
        "neurons":            _count_hidden_neurons(net.model),
        "edges":              n_edges,
        "cross_group_edges":  n_cross,
        "cross_group_w_sum":  round(w_cross, 4),
        "flops":              flops,
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
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        hidden  = linears[:-1]
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
                    importance[id(layer)] += (layer.weight.data * layer.weight.grad).abs().sum(dim=1).cpu()

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
        n_done = 0
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
                tied = (~zero_masks[mod]) & (layer_scores[mod] == threshold) & (mod.weight.data != 0.0)
                for idx in tied.nonzero(as_tuple=False)[:remaining]:
                    mod.weight.data[tuple(idx)] = 0.0
                remaining -= int(min(remaining, tied.sum().item()))


def _prune_l1_global_by_amount(model: nn.Module, amount: float) -> None:
    PrunableNetwork(model, eps=0.0).prune_links_l1_global(amount=amount)


def _compute_edge_prune_amount(net: PrunableNetwork, target_edge_count: int) -> float:
    """Compute the cumulative L1/L2 `amount` needed to reach `target_edge_count`.

    After structural (neuron) pruning the weight matrices are physically resized,
    so we count zero slots directly from the model rather than inferring them
    from (total_weights - graph_edges), which can become inconsistent.
    """
    layers       = [mod for mod in net.model.modules() if isinstance(mod, nn.Linear)]
    total_weights = sum(m.weight.numel() for m in layers)
    already_zero  = sum(int((m.weight.data == 0).sum().item()) for m in layers)
    active_edges  = total_weights - already_zero   # ground truth from model weights
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
    """Granular cross-group edge pruner — replaces bridge-link heuristic in micro-steps.

    Instead of calling prune_links_heuristic (which uses a group-pair guard that
    causes burst pruning at low edge counts), this function:
      1. Re-computes fresh NoC groups from the current graph.
      2. Identifies all edges that cross group boundaries (= bridge links).
      3. Sorts them by |weight| ascending (cheapest first).
      4. Zeroes exactly `edges_to_prune` of them — no group-pair guard, no burst.

    This gives the same semantic as bridge-link (target cross-group edges first)
    but with exact per-step granularity, making it safe at any sparsity level.
    """
    # Fresh grouping from current graph.
    noc_tmp = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                       title="_bridge_tmp", visualize=False)
    groups = noc_tmp.groups  # list of sets/lists of node ids

    # Build node → group-id map.
    node_to_group: dict = {}
    for gid, members in enumerate(groups):
        for node in members:
            node_to_group[node] = gid

    # Compute how many edges to remove this step (from cumulative `amount`).
    layers        = [m for m in net.model.modules() if isinstance(m, nn.Linear)]
    active_now    = sum(int((m.weight.data != 0).sum().item()) for m in layers)
    total_weights = sum(m.weight.numel() for m in layers)
    already_zero  = total_weights - active_now
    edges_to_prune = max(1, int(round(amount * total_weights)) - already_zero)
    edges_to_prune = min(edges_to_prune, active_now - 1)

    if edges_to_prune <= 0:
        return

    # Collect all cross-group (bridge) edges as (abs_weight, layer_idx, row, col).
    # We iterate over Linear layers in forward order; graph nodes are (layer, neuron).
    linears = [m for m in net.model.modules() if isinstance(m, nn.Linear)]
    bridge_candidates: list[tuple[float, int, int, int]] = []

    for li, layer in enumerate(linears):
        w = layer.weight.data  # shape: (out, in)
        # Non-zero entries only.
        nz_rows, nz_cols = (w != 0).nonzero(as_tuple=True)
        for r, c in zip(nz_rows.tolist(), nz_cols.tolist()):
            # In the graph, this edge goes from (li, c) → (li+1, r).
            src_node = (li,     c)
            dst_node = (li + 1, r)
            src_grp  = node_to_group.get(src_node)
            dst_grp  = node_to_group.get(dst_node)
            if src_grp is None or dst_grp is None or src_grp == dst_grp:
                continue  # within-group — skip
            bridge_candidates.append((float(w[r, c].abs()), li, r, c))

    if not bridge_candidates:
        # No cross-group edges remain — fall back to L2 on all edges.
        prune_l2_global(net.model, amount)
        net.commit_pruning()
        net.resync_graph()
        return

    # Sort by magnitude ascending; remove the cheapest `edges_to_prune`.
    bridge_candidates.sort(key=lambda x: x[0])
    to_remove = bridge_candidates[:edges_to_prune]

    with torch.no_grad():
        for _, li, r, c in to_remove:
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
    accuracy_tolerance: float = ACCURACY_TOLERANCE,
    snapshot_interval:  float = SNAPSHOT_INTERVAL,
    edge_step_fraction: float = EDGE_STEP_FRACTION,
    post_neuron_edge_fraction: float | None = None,
    neuron_steps_per_edge_step: int = NEURON_STEPS_PER_EDGE_STEP,
    micro_ft_epochs: int = MICRO_FT_EPOCHS,
    neuron_tolerance: float | None = None,
    neuron_first_steps: int = 0,
    neurons_per_step: int = 1,
) -> None:
    """
    Prune one branch iteratively under an accuracy budget.

    Parameters
    ----------
    accuracy_tolerance         : Hard stop threshold (default 1%).
    snapshot_interval          : Record snapshot at each multiple of this drop.
    edge_step_fraction         : Fraction of active edges removed per edge step
                                 during the interleaved neuron+edge phase.
    post_neuron_edge_fraction  : Edge step fraction after neuron phase exhausts.
                                 Defaults to edge_step_fraction.
    neuron_steps_per_edge_step : How many neuron steps between each edge step.
                                 Set to 1 for 1:1 ratio, 0 for edge-only.
                                 Default = NEURON_STEPS_PER_EDGE_STEP (global).
    micro_ft_epochs            : Fine-tune epochs after each micro-step.
                                 Default = MICRO_FT_EPOCHS (global).
    neuron_tolerance           : When set, the neuron phase is forcibly ended once
                                 the accuracy drop exceeds this value, reserving the
                                 remaining headroom up to accuracy_tolerance exclusively
                                 for the edge phase.  E.g. neuron_tolerance=0.005 gives
                                 the first 0.5% to neurons and the remaining 0.5% to
                                 edges (Branch A / A3 strategy).
                                 None = no split (neuron and edge share the full budget).
    neuron_first_steps         : Run this many pure neuron micro-steps before the
                                 interleaved loop begins.  Each step still respects the
                                 tolerance and reverts on overshoot.
                                 0 = disabled (default).
    neurons_per_step           : Number of neurons to attempt removing per neuron
                                 micro-step.  Default = 1 (conservative, preserves
                                 accuracy budget for edge phase).  Set to 3 for
                                 IterCC branches (A/B/C) where more aggressive neuron
                                 removal is desired.
    """
    floor = anchor_accuracy - accuracy_tolerance

    _post_neuron_frac  = post_neuron_edge_fraction if post_neuron_edge_fraction is not None \
                         else edge_step_fraction
    _n_per_edge        = max(0, neuron_steps_per_edge_step)
    # neuron_tolerance: cap for the neuron phase; None means share the full budget.
    _neuron_tol        = neuron_tolerance if neuron_tolerance is not None else accuracy_tolerance

    # Build snapshot threshold list.
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
        print(f"  Neuron-first phase: {neuron_first_steps} pure neuron steps before interleaving")
    if thresholds:
        print(f"  Snapshot thresholds (drop): "
              f"{[f'{t*100:.1f}%' for t in thresholds]}")
    print(f"  Edge fraction (interleaved/post-neuron): "
          f"{edge_step_fraction:.4f} / {_post_neuron_frac:.4f}  "
          f"Neuron:Edge ratio={_n_per_edge}:1  "
          f"FT epochs={micro_ft_epochs}  Max steps={MAX_STEPS}")
    print("=" * 70)

    net = PrunableNetwork(copy.deepcopy(l1_model_copy), eps=0.0)

    best_model = copy.deepcopy(net.model)
    best_acc   = _extract_accuracy(
        evaluate_model(net.model, X_test, y_test, device=device, timings=timings))
    print(f"[{label}] Starting accuracy: {best_acc:.4f}")

    next_snap_idx          = 0
    total_steps            = 0
    neuron_steps_done      = 0
    neuron_exhausted       = False
    edge_exhausted         = False
    neuron_tol_exhausted   = False   # True once drop > _neuron_tol (A3 split)
    neuron_first_done      = False   # True once the pre-interleave phase is complete

    # ------------------------------------------------------------------
    # Phase 0 — Neuron-first (B/C): pure neuron steps before interleaving
    # ------------------------------------------------------------------
    if neuron_first_steps > 0:
        print(f"[{label}] === Neuron-first phase: up to {neuron_first_steps} steps ===")
        for nf_step in range(1, neuron_first_steps + 1):
            pre_step_model = copy.deepcopy(net.model)

            kwargs = dict(neuron_step_kwargs)
            kwargs["max_neurons"] = neurons_per_step
            n_removed, _ = neuron_step_fn(net, **kwargs)

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
                # Record snapshots if threshold crossed during neuron-first phase.
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

        neuron_first_done = True
        print(f"[{label}] === Neuron-first phase complete. "
              f"Neurons: {_count_hidden_neurons(net.model)}  "
              f"Entering interleaved loop. ===")

    # ------------------------------------------------------------------
    # Phase 1 — Interleaved neuron + edge loop
    # ------------------------------------------------------------------
    for step in range(1, MAX_STEPS + 1):
        if neuron_exhausted and edge_exhausted:
            print(f"[{label}] Both phases exhausted at step {step}. Stopping.")
            break

        # ---- Decide phase ----
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

        # Edge step fraction: use post-neuron fraction once neuron phase is done.
        neuron_phase_done = neuron_exhausted or neuron_tol_exhausted
        active_edge_frac  = edge_step_fraction if not neuron_phase_done else _post_neuron_frac

        # ---- Pre-step snapshot for revert ----
        pre_step_model = copy.deepcopy(net.model)

        # ---- Apply micro-step ----
        if do_neuron:
            kwargs = dict(neuron_step_kwargs)
            kwargs["max_neurons"] = neurons_per_step
            n_removed, _ = neuron_step_fn(net, **kwargs)
            if n_removed == 0:
                neuron_exhausted = True
                net._model = pre_step_model
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
                      f"active_edges={_ae}, already_zero={_tw - _ae}, "
                      f"target_edges={target_edges}, "
                      f"edges_to_prune={max(0, _ae - target_edges)}).")
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

            # Check neuron_tolerance split (A3): if neuron drop cap is reached,
            # force neuron phase to end so edge phase gets the remaining headroom.
            if (not neuron_tol_exhausted
                    and neuron_tolerance is not None
                    and drop >= _neuron_tol):
                neuron_tol_exhausted = True
                remaining = accuracy_tolerance - drop
                print(f"[{label}] Step {step}: neuron tolerance cap reached "
                      f"(drop={drop:.4f} ≥ neuron_tol={_neuron_tol:.4f}). "
                      f"Reserving remaining {remaining:.4f} ({remaining*100:.2f}%) "
                      f"for edge-only phase (step={_post_neuron_frac:.4f}).")

            # Record snapshots if threshold crossed.
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
                 row_stages: list[dict],
                 base_stage: dict) -> None:
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
        is_b  = (s is b)
        tag   = "  (baseline)" if is_b else ""
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
    """
    Write the full per-step trajectory to CSV and JSON.

    Columns
    -------
    branch, step, phase, accuracy, accuracy_drop,
    neurons, edges, cross_group_edges, cross_group_w_sum, flops

    Use these files to plot:
      - comm. cost proxy (cross_group_w_sum) vs neurons    — heuristic quality
      - comm. cost proxy vs edges                          — edge method quality
      - comm. cost proxy vs accuracy_drop                  — Pareto frontier
      - flops vs accuracy_drop                             — inference efficiency
    """
    if not trajectory:
        print("[trajectory] Nothing to save — trajectory list is empty.")
        return

    import csv, json

    # CSV
    fields = list(trajectory[0].keys())
    with open(path_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trajectory)
    print(f"[trajectory] Saved {len(trajectory)} rows → {path_csv}")

    # JSON (pretty-printed for easy inspection)
    with open(path_json, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"[trajectory] Saved {len(trajectory)} rows → {path_json}")


def print_summary_tables() -> None:
    if not stages:
        return

    base = stages[0]   # Baseline row is always first

    # ----------------------------------------------------------------
    # Table 1 — ISO-ACCURACY: snapshot rows only, one per branch × threshold
    # ----------------------------------------------------------------
    snap_rows = [s for s in stages if "@ drop≤" in s["name"]]
    if snap_rows:
        _print_table(
            title_lines=[
                "ISO-ACCURACY TABLE — snapshot at each accuracy-drop threshold",
                f"Model: LeNet 784-500-300-10 | Dataset: MNIST",
                f"Tolerance: ≤{ACCURACY_TOLERANCE*100:.1f}% drop from BASELINE"
                f"  | Snapshot interval: every {SNAPSHOT_INTERVAL*100:.1f}%",
                "A=IterCC+Bridge  B=IterCC+L2  C=IterCC+L1  "
                "D=Taylor+L2  E=Taylor+L1  F=L1Neuron+L2  G=Taylor+Bridge",
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
    anchor_row  = next((s for s in stages if s["name"] == "After L1 25% pre-pruning"), None)
    final_rows  = [s for s in stages if s["name"].endswith("[final]")]
    fixed_rows  = [base] + ([anchor_row] if anchor_row else []) + final_rows

    _print_table(
        title_lines=[
            "FINAL TABLE — each branch at its stopping point (last within tolerance)",
            f"Model: LeNet 784-500-300-10 | Dataset: MNIST",
            f"Tolerance: ≤{ACCURACY_TOLERANCE*100:.1f}% drop from BASELINE",
            "A=IterCC+Bridge  B=IterCC+L2  C=IterCC+L1  "
            "D=Taylor+L2  E=Taylor+L1  F=L1Neuron+L2",
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

# Tolerance anchor = BASELINE accuracy (stages[0]), not the L1 checkpoint.
anchor_accuracy  = stages[0]["accuracy"]
_anchor_accuracy = anchor_accuracy

print(f"\n[Checkpoint] L1 model: {l1_neuron_count} neurons, {l1_edge_count:,} edges.")
print(f"[Checkpoint] Tolerance anchor : BASELINE accuracy = {anchor_accuracy:.4f}")
print(f"[Checkpoint] Hard stop floor  : {anchor_accuracy - ACCURACY_TOLERANCE:.4f}")
print(f"[Checkpoint] iterCC_low_mag_pct = {iterCC_low_mag_pct}  "
      f"(permissive gate — CommCost score drives Stage-2 selection)")
if SNAPSHOT_INTERVAL:
    n_snaps = int(ACCURACY_TOLERANCE / SNAPSHOT_INTERVAL)
    floors  = [f"{anchor_accuracy - SNAPSHOT_INTERVAL*(i+1):.4f}" for i in range(n_snaps)]
    print(f"[Checkpoint] Snapshot floors  : {', '.join(floors)}")

# Shared kwargs for all IterCC and Taylor branches.
_iterCC_kwargs = dict(
    N=N, grouping_cap=10_000.0,
    low_mag_percentile=iterCC_low_mag_pct,
    degree_quantile=iterCC_degree_q,
    min_in_degree=iterCC_min_in_degree,
    min_score=0.0, debug=False,
)
_taylor_kwargs = dict(
    X_calib=X_train, y_calib=y_train, device=device,
    n_batches=10, batch_sz=128,
)

# ==================================================================
# 5. Branch A — IterativeCC Neuron + Bridge-Link  (heuristic, primary)
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
    neurons_per_step=3,              # IterCC: 3 neurons per step
)

# ==================================================================
# 6. Branch B — IterativeCC Neuron + L2 edge
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
    neuron_tolerance=0.005,           # first 0.5% budget reserved for neurons
    post_neuron_edge_fraction=0.05,   # aggressive edge step once neurons done
    neurons_per_step=3,               # IterCC: 3 neurons per step
    micro_ft_epochs=2,                # less recovery → budget consumed faster
)

# ==================================================================
# 7. Branch C — IterativeCC Neuron + L1 edge  (new)
# ==================================================================

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
    neuron_tolerance=0.005,           # first 0.5% budget reserved for neurons
    post_neuron_edge_fraction=0.05,   # aggressive edge step once neurons done
    neurons_per_step=3,               # IterCC: 3 neurons per step
    micro_ft_epochs=2,                # less recovery → budget consumed faster
)

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
    neuron_steps_per_edge_step=1,        # 1:1 ratio — edge gets equal share of steps
    post_neuron_edge_fraction=0.05,      # aggressive edge step after neurons exhaust
)

# ==================================================================
# 9. Branch E — Taylor Neuron + L1 edge
# ==================================================================

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
    neuron_steps_per_edge_step=1,        # 1:1 ratio — edge gets equal share of steps
    post_neuron_edge_fraction=0.05,      # aggressive edge step after neurons exhaust
)

# ==================================================================
# 10. Branch F — L1 Neuron + L2 edge  (classical baseline)
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
    neuron_steps_per_edge_step=1,        # 1:1 ratio — edge gets equal share of steps
    post_neuron_edge_fraction=0.05,      # aggressive edge step after neurons exhaust
)
"""
# ==================================================================
# 11. Branch G — Taylor Neuron + Bridge-Link
#     Tests whether bridge-link edge pruning benefits Taylor neuron selection
#     the same way it benefits IterCC (Branch A).
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
)

# ==================================================================
# 12. Summary
# ==================================================================

print_summary_tables()
save_trajectories(
    path_csv="trajectory_LeNet500_300.csv",
    path_json="trajectory_LeNet500_300.json"
)
