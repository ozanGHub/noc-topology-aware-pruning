"""
main_LeNet_300_100_CIFAR10_e2e_AccuracyBudget_Metaheuristic.py
===============================================================
Accuracy-budget pruning comparison for LeNet-300-100 on CIFAR-10.

Based on main_LeNet_300_100_e2e_AccuracyBudget_Metaheuristic.py (MNIST),
adapted for CIFAR-10 with 3072-dimensional flattened input.

Architecture: fc1 (3072→300) → fc2 (300→100) → fc3 (100→10)
Hidden neurons: 300 + 100 = 400 → 6×6 = 36 cores

Anchor accuracy: POST-L1 model (not baseline), consistent with other CIFAR-10 files.

Branches
--------
  A : IterCC Neuron + Bridge-Link     (heuristic, primary)
  B : IterCC Neuron + L2 edge
  D : Taylor Neuron + L2 edge         (classical)
  G : Taylor Neuron + Bridge-Link
  Q : GA CC_Biased v2 + Bridge-Link  (metaheuristic)
  V : GA Memetic Biased + Bridge     (metaheuristic)
  W : GA Memetic Biased + L2 edge    (metaheuristic)
  V2: GA Memetic Biased + Bridge(s)  (metaheuristic — bridge 0.01)
"""

from __future__ import annotations

import copy
import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from DNN_LeNet_300_100_CIFAR10 import load_or_train, fine_tune, evaluate_model
from prunable_network import PrunableNetwork
from eval_helper import eval_noc, eval_nn
from mapping_helpers import build_wctg_structural, compute_structural_cost
from pruning import iterative_commcost_neuron_pruning
from pruning_meta import (
    ga_neuron_step_biased_v2,
    ga_neuron_step_memetic_biased,
)

# ==================================================================
# Configuration
# ==================================================================

data_dir:     str   = "./data"
model_path:   str   = "DNN_LeNet_300_100_CIFAR10.pth"
seed:         int   = 42
batch_size:   int   = 128
lr:           float = 1e-3
epochs:       int   = 200
weight_decay: float = 1e-4
force_train:  bool  = False

# --- Shared L1 pre-pruning ---
l1_target: float = 0.25 #original = 0.25
l1_steps:  int   = 5
ft_epochs: int   = 10

# --- Accuracy-budget settings ---
ACCURACY_TOLERANCE:         float = 0.01
SNAPSHOT_INTERVAL:          float = 0.005
MICRO_FT_EPOCHS:            int   = 5
NEURON_STEPS_PER_EDGE_STEP: int   = 3
NEURONS_PER_STEP:           int   = 3
EDGE_STEP_FRACTION:         float = 0.02
BRIDGE_STEP_FRACTION:       float = 0.02
MAX_STEPS:                  int   = 1000

# ==================================================================
# Configuration — GA Metaheuristic
# ==================================================================

GA_POP_SIZE:               int   = 30
GA_N_GENERATIONS:          int   = 20
GA_TOURNAMENT_K:           int   = 3
GA_MUTATION_RATE:          float = 0.2
GA_ELITE_FRAC:             float = 0.1
GA_ALPHA_Q:                float = 0.6
GA_BETA_Q:                 float = 0.4
GA_RESCORE_INTERVAL:       int   = 5
GA_LOCAL_SEARCH_ITERS:     int   = 2
GA_LOCAL_SEARCH_CANDIDATES: int  = 5
BRIDGE_POST_NEURON_SMALL:  float = 0.01

# IterCC neuron filter settings
iterCC_low_mag_pct:   float = 90.0
iterCC_degree_q:      float = 0.25
iterCC_min_in_degree: int   = 0

# NoC mapping parameters — 11×11 mesh (400 hidden neurons, ~3,3/core)
P:         int = 121
N:         int = 121
mesh_rows: int = int(math.sqrt(P))
mesh_cols: int = int(math.sqrt(P))

# ==================================================================
# Metrics tracker
# ==================================================================

stages:    list[dict] = []
trajectory: list[dict] = []


def _traj_record(branch: str, step: int, phase: str,
                 acc: float, drop: float, net: PrunableNetwork) -> None:
    try:
        from grouping_heuristic import indegree_grouping as _ig
        groups = _ig(net.graph, N, 10_000.0)
        node_to_grp: dict = {}
        for gid, members in enumerate(groups):
            for nd in members:
                node_to_grp[nd] = gid
        n_cross, w_cross = 0, 0.0
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
        "branch": branch, "step": step, "phase": phase,
        "accuracy": acc, "accuracy_drop": drop,
        "neurons": _count_hidden_neurons(net.model),
        "edges": n_edges,
        "cross_group_edges": n_cross,
        "cross_group_w_sum": round(w_cross, 4),
        "flops": compute_model_flops(net.model),
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
        "name": name, "accuracy": acc,
        "f1": metrics["f1"], "precision": metrics["precision"], "recall": metrics["recall"],
        "cost": float(noc.cost), "sp_cost": sp_cost,
        "flops": compute_model_flops(net.model),
        "edges": net.graph.number_of_edges(),
        "neurons": net.graph.number_of_nodes(),
        "batch_ms": inf["batch_ms"], "state_size_kb": metrics["state_size_kb"],
    })
    print(f"  [recorded] '{name}'  acc={acc:.4f}  cost={noc.cost:.0f}  "
          f"sp_cost={sp_cost:.0f}  edges={net.graph.number_of_edges():,}  "
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


# ==================================================================
# Edge-step wrappers
# Signature: fn(net, *, amount, **kwargs) -> None
# ==================================================================

def _l2_edge_step(net: PrunableNetwork, *, amount: float, **_) -> None:
    prune_l2_global(net.model, amount)


def _bridge_edge_step(net: PrunableNetwork, *, amount: float, **_) -> None:
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

    bridge_candidates: list[tuple[float, int, int, int]] = []
    for li, layer in enumerate(layers):
        w = layer.weight.data
        nz_rows, nz_cols = (w != 0).nonzero(as_tuple=True)
        for r, c in zip(nz_rows.tolist(), nz_cols.tolist()):
            src_grp = node_to_group.get((li,     c))
            dst_grp = node_to_group.get((li + 1, r))
            if src_grp is None or dst_grp is None or src_grp == dst_grp:
                continue
            bridge_candidates.append((float(layer.weight.data[r, c].abs()), li, r, c))

    if not bridge_candidates:
        prune_l2_global(net.model, amount)
        net.commit_pruning()
        net.resync_graph()
        return

    bridge_candidates.sort(key=lambda x: x[0])
    with torch.no_grad():
        for _, li, r, c in bridge_candidates[:edges_to_prune]:
            layers[li].weight.data[r, c] = 0.0

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
    device, timings,
    *,
    neuron_step_fn: Callable,
    neuron_step_kwargs: dict,
    edge_step_fn: Callable,
    edge_step_kwargs: dict,
    accuracy_tolerance: float      = ACCURACY_TOLERANCE,
    snapshot_interval:  float      = SNAPSHOT_INTERVAL,
    edge_step_fraction: float      = EDGE_STEP_FRACTION,
    post_neuron_edge_fraction: float | None = None,
    neuron_steps_per_edge_step: int = NEURON_STEPS_PER_EDGE_STEP,
    micro_ft_epochs: int           = MICRO_FT_EPOCHS,
    neuron_tolerance: float | None = None,
    neurons_per_step: int          = 1,
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
        print(f"  Neuron phase cap: {_neuron_tol:.4f}")
    if thresholds:
        print(f"  Snapshot thresholds: {[f'{t*100:.1f}%' for t in thresholds]}")
    print(f"  Edge frac (interleaved/post): {edge_step_fraction:.4f}/{_post_neuron_frac:.4f}  "
          f"N:E ratio={_n_per_edge}:1  FT={micro_ft_epochs}  Max={MAX_STEPS}")
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

    for step in range(1, MAX_STEPS + 1):
        if neuron_exhausted and edge_exhausted:
            print(f"[{label}] Both phases exhausted. Stopping.")
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
        pre_step_model    = copy.deepcopy(net.model)

        if do_neuron:
            kwargs = dict(neuron_step_kwargs)
            kwargs["max_neurons"] = neurons_per_step
            n_removed, _ = neuron_step_fn(net, **kwargs)
            if n_removed == 0:
                neuron_exhausted = True
                net._model       = pre_step_model
                net.resync_graph()
                remaining = accuracy_tolerance - (anchor_accuracy - best_acc)
                print(f"[{label}] Step {step}: neuron exhausted. "
                      f"Remaining headroom: {remaining:.4f}.")
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
                print(f"[{label}] Step {step}: edge exhausted.")
                continue
            amount = _compute_edge_prune_amount(net, target_edges)
            if not (0.0 < amount < 1.0):
                edge_exhausted = True
                print(f"[{label}] Step {step}: edge exhausted (amount={amount:.4f}).")
                continue
            edge_step_fn(net, amount=amount, **edge_step_kwargs)
            net.commit_pruning()
            net.resync_graph()
            phase_label = "edge" if not neuron_phase_done else "edge*"

        reapply_zero_masks(net)
        fine_tune(net.model, X_train, y_train, device=device, lr=lr,
                  epochs=micro_ft_epochs, seed=seed + step)
        net.commit_pruning()
        net.resync_graph()

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
                print(f"[{label}] Step {step}: neuron tolerance cap "
                      f"(drop={drop:.4f} ≥ {_neuron_tol:.4f}). "
                      f"Reserving {remaining:.4f} for edge-only ({_post_neuron_frac:.4f}).")

            while (next_snap_idx < len(thresholds)
                   and drop >= thresholds[next_snap_idx]):
                snap_pct   = thresholds[next_snap_idx] * 100
                snap_label = f"{label} @ drop≤{snap_pct:.1f}%"
                print(f"[{label}] → Snapshot at drop≤{snap_pct:.1f}%")
                _noc_and_record(snap_label, net, X_test, y_test, device, timings)
                next_snap_idx += 1
        else:
            print(f"[{label}] Tolerance exceeded (drop={drop:.4f}). Reverting.")
            net._model = best_model
            net.resync_graph()
            break

    final_label = f"{label} [final]"
    print(f"\n[{label}] Finished. {total_steps} steps. "
          f"Final: acc={best_acc:.4f}  "
          f"neurons={_count_hidden_neurons(net.model)}  "
          f"edges={net.graph.number_of_edges():,}")
    eval_nn(net.model, X_test=X_test, y_test=y_test,
            device=device, timings=timings, evaluate_fn=evaluate_model)
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


def print_summary_tables() -> None:
    if not stages:
        return
    base = stages[0]

    snap_rows  = [s for s in stages if "@ drop≤" in s["name"]]
    final_rows = [s for s in stages if s["name"].endswith("[final]")]
    #anchor_row = next((s for s in stages if s["name"] == "After L1 25% pre-pruning"), None)
    anchor_row = next((s for s in stages if s["name"] == f"After L1 {l1_target:.0%} pre-pruning"), None)

    title_common = [
        "Model: LeNet 3072-300-100-10 | Dataset: CIFAR-10",
        f"Tolerance: ≤{ACCURACY_TOLERANCE*100:.1f}% drop from POST-L1 anchor",
        "A=IterCC+Bridge  B=IterCC+L2  D=Taylor+L2  G=Taylor+Bridge",
        "Q=GABiasedv2+Bridge  V=GAMemBiased+Bridge  W=GAMemBiased+L2  V2=GAMemBiased+Bridge(0.01)",
        "Deltas relative to Baseline.",
    ]

    if snap_rows:
        _print_table(
            title_lines=["ISO-ACCURACY TABLE — snapshot at each accuracy-drop threshold",
                         f"Snapshot interval: every {SNAPSHOT_INTERVAL*100:.1f}%"] + title_common,
            row_stages=[base] + snap_rows,
            base_stage=base,
        )
    else:
        print("\n[No intermediate snapshots recorded]")

    _print_table(
        title_lines=["FINAL TABLE — each branch at its stopping point (last within tolerance)"]
                    + title_common,
        row_stages=[base] + ([anchor_row] if anchor_row else []) + final_rows,
        base_stage=base,
    )


def save_trajectories(path_csv: str = "trajectory_LeNet300_CIFAR10_Meta.csv",
                      path_json: str = "trajectory_LeNet300_CIFAR10_Meta.json") -> None:
    if not trajectory:
        print("[trajectory] Nothing to save.")
        return
    import csv, json
    fields = list(trajectory[0].keys())
    with open(path_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(trajectory)
    print(f"[trajectory] Saved {len(trajectory)} rows → {path_csv}")
    with open(path_json, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"[trajectory] Saved {len(trajectory)} rows → {path_json}")


# ==================================================================
# 1. Train / load
# ==================================================================

model, scaler, (X_train, X_test, y_train, y_test), timings, device = load_or_train(
    data_dir, model_path,
    seed=seed, batch_size=batch_size,
    lr=lr, epochs=epochs, weight_decay=weight_decay,
    force_train=force_train,
    patience=20, min_delta=1e-4,
)

net = PrunableNetwork(model, eps=0.0)

# ==================================================================
# 2. Baseline evaluation
# ==================================================================

print("\n=== Initial model ===")
eval_nn(net.model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
noc = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
               title="Baseline", visualize=False)
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
        print(f"  [Step {step}/{l1_steps}] Skipping.")
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
noc = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
               title="After L1 pruning", visualize=False)
#_noc_and_record("After L1 25% pre-pruning", net, X_test, y_test, device, timings)
_noc_and_record(f"After L1 {l1_target:.0%} pre-pruning", net, X_test, y_test, device, timings)

# ==================================================================
# 4. Checkpoint + anchor  (POST-L1 accuracy is the tolerance anchor)
# ==================================================================

l1_model_copy   = copy.deepcopy(net.model)
l1_neuron_count = _count_hidden_neurons(net.model)
l1_edge_count   = net.graph.number_of_edges()
anchor_accuracy = stages[-1]["accuracy"]   # post-L1, not baseline

print(f"\n[Checkpoint] L1 model: {l1_neuron_count} neurons, {l1_edge_count:,} edges.")
print(f"[Checkpoint] Tolerance anchor: POST-L1 = {anchor_accuracy:.4f}")
print(f"[Checkpoint] Hard stop floor : {anchor_accuracy - ACCURACY_TOLERANCE:.4f}")

# Shared kwargs
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
_ga_kwargs = dict(
    N=N, grouping_cap=10_000.0,
    low_mag_percentile=iterCC_low_mag_pct,
    pop_size=GA_POP_SIZE, n_generations=GA_N_GENERATIONS,
    tournament_k=GA_TOURNAMENT_K, mutation_rate=GA_MUTATION_RATE,
    elite_frac=GA_ELITE_FRAC, ga_seed=seed, debug=False,
)
_ga_biased_v2_kwargs = dict(**_ga_kwargs, alpha=GA_ALPHA_Q, beta=GA_BETA_Q,
                             rescore_interval=GA_RESCORE_INTERVAL)
_ga_memetic_biased_kwargs = dict(**_ga_kwargs, alpha=GA_ALPHA_Q, beta=GA_BETA_Q,
                                  local_search_iters=GA_LOCAL_SEARCH_ITERS,
                                  local_search_candidates=GA_LOCAL_SEARCH_CANDIDATES)

# ==================================================================
# 5. Branch A — IterCC Neuron + Bridge-Link
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
"""
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
    neuron_steps_per_edge_step=1,
    neurons_per_step=3,
    micro_ft_epochs=2,
)
"""
# ==================================================================
# 7. Branch D — Taylor Neuron + L2 edge
# ==================================================================
"""
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
"""
# ==================================================================
# 8. Branch G — Taylor Neuron + Bridge-Link
# ==================================================================
"""
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
"""
# ==================================================================
# 9. Branch Q — GA CC_Biased v2 + Bridge-Link
# ==================================================================
"""
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
"""
# ==================================================================
# 10. Branch V — GA Memetic Biased + Bridge-Link
# ==================================================================
"""
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
"""
# ==================================================================
# 11. Branch W — GA Memetic Biased + L2 Edge
# ==================================================================
"""
run_accuracy_budget_branch(
    label="W: GA Memetic Biased + L2 Edge",
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
"""
# ==================================================================
# 12. Branch V2 — GA Memetic Biased + Bridge-Link (small step)
# ==================================================================
"""
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
"""
# ==================================================================
# 13. Summary
# ==================================================================

print_summary_tables()
"""
save_trajectories(
    path_csv="trajectory_LeNet300_CIFAR10_Meta.csv",
    path_json="trajectory_LeNet300_CIFAR10_Meta.json"
)
"""