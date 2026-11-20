"""
main_LeNet5_CIFAR10_e2e_AccuracyBudget_Metaheuristic.py
========================================================
Accuracy-budget pruning for modern LeNet-5 on CIFAR-10.

Architecture (channel-level graph):
  Input(3) → conv1(6) → c3(16) → c5(120) → f6(84) → output(10)
  Total hidden channels/neurons: 6+16+120+84 = 226 → 6×6 = 36 cores

Graph model (graph_conv.py)
  Conv2d  : one node per output channel; edge weight = ||W[j,i,:,:]||_1
  Linear  : one node per neuron;         edge weight = ||W[j, i*sp:(i+1)*sp]||_1
            (sp=spatial block size; 25 for c5 after 5×5 conv, 1 for f6/output)

Pruning strategy (middle path)
  Neuron/channel step  : IterCC or Taylor — operates on all graph nodes,
                         but only Linear nodes are physically removed.
  Edge step (bridge)   : Conv2d transitions → zero entire kernel W[j,i,:,:]
                         Linear-after-conv  → zero weight block W[j, i*sp:(i+1)*sp]
                         Linear-after-linear→ zero single weight W[j,i]
  Edge step (L2)       : same kernel/block/weight granularity as bridge.

Anchor: POST-L1 accuracy (stages[-1]).

Branches
--------
  A : IterCC Neuron + Bridge-Link  (active)
  B : IterCC Neuron + L2           (active)
  D : Taylor Neuron + L2           (active)
  G : Taylor Neuron + Bridge-Link  (active)
  Q/V/W/V2 : GA variants           (commented out)
"""

from __future__ import annotations

import copy
import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from DNN_LeNet5_CIFAR10 import load_or_train, fine_tune, evaluate_model
from prunable_network_conv import PrunableNetworkConv
from eval_helper import eval_noc, eval_nn
from mapping_helpers import build_wctg_structural, compute_structural_cost
from pruning import iterative_commcost_neuron_pruning

# ==================================================================
# Configuration
# ==================================================================

data_dir:     str   = "./data"
model_path:   str   = "DNN_LeNet5_CIFAR10.pth"
seed:         int   = 42
batch_size:   int   = 128
lr:           float = 1e-3
epochs:       int   = 200
weight_decay: float = 1e-4
force_train:  bool  = False

l1_target: float = 0.25
l1_steps:  int   = 5
ft_epochs: int   = 10

ACCURACY_TOLERANCE:         float = 0.01
SNAPSHOT_INTERVAL:          float = 0.005
MICRO_FT_EPOCHS:            int   = 5
NEURON_STEPS_PER_EDGE_STEP: int   = 1
NEURONS_PER_STEP:           int   = 3
EDGE_STEP_FRACTION:         float = 0.02
BRIDGE_STEP_FRACTION:       float = 0.02
MAX_STEPS:                  int   = 1000

# NoC — 3×3 mesh (226 hidden nodes, ~6.3/core)
P:         int = 9
N:         int = 9
mesh_rows: int = 3
mesh_cols: int = 3

iterCC_low_mag_pct:   float = 90.0
iterCC_degree_q:      float = 0.25
iterCC_min_in_degree: int   = 0

# ==================================================================
# Metrics
# ==================================================================

stages:     list[dict] = []
trajectory: list[dict] = []


def _traj_record(branch, step, phase, acc, drop, net):
    layers  = [m for m in net.model.modules() if isinstance(m, nn.Linear)]
    n_edges = sum(int((m.weight.data != 0).sum().item()) for m in layers)
    trajectory.append({
        "branch": branch, "step": step, "phase": phase,
        "accuracy": acc, "accuracy_drop": drop,
        "fc_neurons": _count_fc_neurons(net.model),
        "edges": n_edges,
        "flops": compute_model_flops(net.model),
    })


# ==================================================================
# Helpers
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
    return {"batch_ms": float(report.batch_ms) if hasattr(report, "batch_ms") else float("nan")}


def _extract_metrics(report) -> dict:
    result = {"f1": float("nan"), "precision": float("nan"),
              "recall": float("nan"), "state_size_kb": float("nan")}
    for attr in result:
        if hasattr(report, attr):
            result[attr] = float(getattr(report, attr))
    return result


def compute_model_flops(model: nn.Module) -> int:
    """Approximate MACs × 2 for Linear layers only (conv excluded for simplicity)."""
    total = 0
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            total += mod.in_features * mod.out_features
    return 2 * total


def _count_fc_neurons(model: nn.Module) -> int:
    """Count hidden neurons in Linear layers only (c5 + f6)."""
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return sum(m.out_features for m in linears[:-1])


def _get_ordered_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Conv2d and Linear layers in forward order — matches graph layer indices."""
    return [(name, m) for name, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))]


def reapply_zero_masks(net: PrunableNetworkConv) -> None:
    for mod in net.model.modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            if hasattr(mod, "weight_orig"):
                prune.remove(mod, "weight")
            mask = (mod.weight.data != 0).float()
            if mask.all():
                continue
            prune.custom_from_mask(mod, "weight", mask)


def _noc_and_record(name, net, X_test, y_test, device, timings):
    report  = evaluate_model(net.model, X_test, y_test, device=device, timings=timings)
    acc     = _extract_accuracy(report)
    inf     = _extract_inference_time(report)
    metrics = _extract_metrics(report)

    noc = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                   title=name, visualize=False)

    sp_wctg = build_wctg_structural(net.graph, noc.groups, net.model, eps=1e-8)
    sp_cost = compute_structural_cost(sp_wctg, noc.spg)

    stages.append({
        "name": name, "accuracy": acc,
        "f1": metrics["f1"], "precision": metrics["precision"],
        "recall": metrics["recall"],
        "cost": float(noc.cost), "sp_cost": sp_cost,
        "flops": compute_model_flops(net.model),
        "edges": net.graph.number_of_edges(),
        "neurons": net.graph.number_of_nodes(),
        "batch_ms": inf["batch_ms"], "state_size_kb": metrics["state_size_kb"],
    })
    print(f"  [recorded] '{name}'  acc={acc:.4f}  cost={noc.cost:.0f}  "
          f"sp_cost={sp_cost:.0f}  edges={net.graph.number_of_edges():,}  "
          f"fc_neurons={_count_fc_neurons(net.model)}")


# ==================================================================
# Neuron / edge pruning — conv-aware
# ==================================================================

def _conv_edge_weight(m: nn.Module, out_j: int, in_i: int, sp: int) -> float:
    """Return the scalar edge weight for a single channel-pair connection."""
    if isinstance(m, nn.Conv2d):
        return float(m.weight.data[out_j, in_i].abs().sum().item())
    else:  # Linear (possibly after conv, sp > 1)
        return float(m.weight.data[out_j, in_i * sp : (in_i + 1) * sp].abs().sum().item())


def _conv_zero_edge(m: nn.Module, out_j: int, in_i: int, sp: int) -> None:
    """Zero the kernel or weight block connecting in_i → out_j."""
    with torch.no_grad():
        if isinstance(m, nn.Conv2d):
            m.weight.data[out_j, in_i] = 0.0         # entire kH×kW kernel
        else:  # Linear
            m.weight.data[out_j, in_i * sp : (in_i + 1) * sp] = 0.0


def _conv_bridge_edge_step(net: PrunableNetworkConv, *, amount: float, **_) -> None:
    """
    Bridge-link pruning for mixed Conv+Linear networks.
    For Conv2d transitions: zeros entire kernel W[out_j, in_i, :, :].
    For Linear transitions: zeros weight block W[out_j, in_i*sp:(in_i+1)*sp].
    """
    noc_tmp = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                       title="_bridge_tmp", visualize=False)
    node_to_group = {node: gid for gid, members in enumerate(noc_tmp.groups)
                     for node in members}

    G           = net.graph
    layer_types = G.graph["layer_types"]
    spatial_sizes = G.graph["spatial_sizes"]
    ordered     = _get_ordered_layers(net.model)

    # Count active channel-pair connections and total edge weights
    total_weights = sum(
        m.weight.numel() for _, m in ordered
    )
    already_zero = sum(
        int((m.weight.data == 0).sum().item()) for _, m in ordered
    )
    active_edges = total_weights - already_zero
    edges_to_prune = max(1, int(round(amount * total_weights)) - already_zero)
    edges_to_prune = min(edges_to_prune, active_edges - 1)
    if edges_to_prune <= 0:
        return

    # Collect cross-group edges sorted by magnitude ascending
    bridge_candidates: list[tuple[float, int, int, int]] = []
    for (src_node, dst_node, _) in G.edges(data=True):
        src_grp = node_to_group.get(src_node)
        dst_grp = node_to_group.get(dst_node)
        if src_grp is None or dst_grp is None or src_grp == dst_grp:
            continue
        layer_idx = src_node[0]
        in_i      = src_node[1]
        out_j     = dst_node[1]
        m         = ordered[layer_idx][1]
        sp        = spatial_sizes[layer_idx]
        w_scalar  = _conv_edge_weight(m, out_j, in_i, sp)
        bridge_candidates.append((w_scalar, layer_idx, out_j, in_i))

    if not bridge_candidates:
        _conv_l2_edge_step(net, amount=amount)
        return

    bridge_candidates.sort(key=lambda x: x[0])

    pruned = 0
    for w_scalar, layer_idx, out_j, in_i in bridge_candidates:
        if pruned >= edges_to_prune:
            break
        m  = ordered[layer_idx][1]
        sp = spatial_sizes[layer_idx]
        _conv_zero_edge(m, out_j, in_i, sp)
        pruned += 1

    net.remove_lonely_neurons()
    net.commit_pruning()
    net.resync_graph()


def _conv_l2_edge_step(net: PrunableNetworkConv, *, amount: float, **_) -> None:
    """
    L2 (smallest-magnitude) edge pruning for mixed Conv+Linear networks.
    Operates at kernel/block granularity matching the NoC cost model.
    """
    G           = net.graph
    spatial_sizes = G.graph["spatial_sizes"]
    ordered     = _get_ordered_layers(net.model)

    # Build a flat list of (magnitude, layer_idx, out_j, in_i)
    all_edges: list[tuple[float, int, int, int]] = []
    for layer_idx, (name, m) in enumerate(ordered):
        sp       = spatial_sizes[layer_idx]
        src_size = G.graph["layers"][layer_idx]
        if isinstance(m, nn.Conv2d):
            for out_j in range(m.out_channels):
                for in_i in range(m.in_channels):
                    w = float(m.weight.data[out_j, in_i].abs().sum().item())
                    all_edges.append((w, layer_idx, out_j, in_i))
        else:  # Linear
            for out_j in range(m.out_features):
                for in_i in range(src_size):
                    w = float(m.weight.data[out_j, in_i * sp : (in_i + 1) * sp]
                               .abs().sum().item())
                    all_edges.append((w, layer_idx, out_j, in_i))

    total_connections = len(all_edges)
    n_zero_now = sum(1 for (w, *_) in all_edges if w == 0.0)
    n_to_prune = max(1, int(round(amount * total_connections)) - n_zero_now)
    n_to_prune = min(n_to_prune, total_connections - n_zero_now - 1)
    if n_to_prune <= 0:
        return

    all_edges.sort(key=lambda x: x[0])
    pruned = 0
    for w_scalar, layer_idx, out_j, in_i in all_edges:
        if pruned >= n_to_prune:
            break
        if w_scalar == 0.0:
            continue
        m  = ordered[layer_idx][1]
        sp = spatial_sizes[layer_idx]
        _conv_zero_edge(m, out_j, in_i, sp)
        pruned += 1

    net.commit_pruning()
    net.resync_graph()


def _taylor_neuron_step(net: PrunableNetworkConv, *, max_neurons: int,
                        X_calib, y_calib, device,
                        n_batches: int = 10, batch_sz: int = 128) -> tuple[int, list]:
    """
    Taylor-based channel/neuron pruning.
    Scores all nodes (conv channels + FC neurons) by |activation × gradient|.
    Only Linear nodes are physically removed (conv channels skipped by
    PrunableNetworkConv.prune_neurons).
    """
    if isinstance(X_calib, np.ndarray):
        X_calib = torch.from_numpy(X_calib.astype(np.float32))
    if isinstance(y_calib, np.ndarray):
        y_calib = torch.from_numpy(y_calib.astype(np.int64))

    model     = net.model
    ordered   = _get_ordered_layers(model)
    G         = net.graph
    layer_types = G.graph["layer_types"]

    # importance: keyed by (layer_idx+1, out_j) = graph node  (1-indexed output)
    importance: dict[tuple[int,int], float] = {}
    for layer_idx, (name, m) in enumerate(ordered):
        node_layer = layer_idx + 1
        if isinstance(m, nn.Conv2d):
            for j in range(m.out_channels):
                importance[(node_layer, j)] = 0.0
        else:
            for j in range(m.out_features):
                importance[(node_layer, j)] = 0.0

    was_training = model.training
    model.train()
    criterion = nn.CrossEntropyLoss()
    n = X_calib.shape[0]

    # Hook-based activation capture for conv layers
    activations: dict[str, torch.Tensor] = {}
    hooks = []
    for name, m in ordered:
        def _hook(mod, inp, out, _name=name):
            activations[_name] = out.detach()
        hooks.append(m.register_forward_hook(_hook))

    for b in range(n_batches):
        idx_b = torch.randperm(n, generator=torch.Generator().manual_seed(b))[:batch_sz]
        xb, yb = X_calib[idx_b].to(device), y_calib[idx_b].to(device)
        model.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()

        with torch.no_grad():
            for layer_idx, (name, m) in enumerate(ordered):
                node_layer = layer_idx + 1
                if isinstance(m, nn.Conv2d):
                    act = activations.get(name)  # [B, C_out, H, W]
                    if act is None or m.weight.grad is None:
                        continue
                    # Taylor score: mean over batch + spatial of |act × grad_w|
                    # Use output activation as proxy
                    grad_out = m.weight.grad  # [C_out, C_in, kH, kW]
                    # Approximate importance as L1 of (weight × grad) summed per filter
                    score_per_filter = (m.weight.data * grad_out).abs().sum(dim=(1, 2, 3))
                    for j in range(m.out_channels):
                        importance[(node_layer, j)] += float(score_per_filter[j].item()) / n_batches
                else:  # Linear
                    if m.weight.grad is None:
                        continue
                    scores = (m.weight.data * m.weight.grad).abs().sum(dim=1)
                    for j in range(m.out_features):
                        importance[(node_layer, j)] += float(scores[j].item()) / n_batches

    for h in hooks:
        h.remove()
    model.zero_grad()
    if not was_training:
        model.eval()

    # Only consider hidden nodes (exclude input layer 0 and output layer)
    layers     = G.graph["layers"]
    last_layer = len(layers) - 1
    candidates = {node: score for node, score in importance.items()
                  if 0 < node[0] < last_layer}

    if not candidates:
        return 0, []

    # Select max_neurons lowest-importance nodes
    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1])
    to_prune     = [node for node, _ in sorted_nodes[:max_neurons]]

    before = _count_fc_neurons(model)
    net.prune_neurons(to_prune)   # conv nodes silently skipped inside
    net.remove_lonely_neurons()
    net.commit_pruning()
    net.resync_graph()
    after  = _count_fc_neurons(net.model)
    return before - after, []


def _compute_edge_prune_amount(net: PrunableNetworkConv, target_channel_pairs: int) -> float:
    """Compute `amount` fraction of total connections to zero to reach target."""
    ordered     = _get_ordered_layers(net.model)
    G           = net.graph
    spatial_sizes = G.graph["spatial_sizes"]
    total = 0
    zero  = 0
    for layer_idx, (name, m) in enumerate(ordered):
        sp       = spatial_sizes[layer_idx]
        src_size = G.graph["layers"][layer_idx]
        if isinstance(m, nn.Conv2d):
            n_pairs = m.in_channels * m.out_channels
        else:
            n_pairs = src_size * m.out_features
        total += n_pairs
        for out_j in range(m.out_channels if isinstance(m, nn.Conv2d) else m.out_features):
            for in_i in range(m.in_channels if isinstance(m, nn.Conv2d) else src_size):
                if _conv_edge_weight(m, out_j, in_i, sp) == 0.0:
                    zero += 1
    active  = total - zero
    to_zero = max(0, active - target_channel_pairs)
    if total == 0:
        return 0.0
    return float((zero + to_zero) / total)


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
    accuracy_tolerance:       float = ACCURACY_TOLERANCE,
    snapshot_interval:        float = SNAPSHOT_INTERVAL,
    edge_step_fraction:       float = EDGE_STEP_FRACTION,
    post_neuron_edge_fraction: float | None = None,
    neuron_steps_per_edge_step: int = NEURON_STEPS_PER_EDGE_STEP,
    micro_ft_epochs:          int   = MICRO_FT_EPOCHS,
    neuron_tolerance:         float | None = None,
    neurons_per_step:         int   = 1,
) -> None:
    floor             = anchor_accuracy - accuracy_tolerance
    _post_neuron_frac = post_neuron_edge_fraction if post_neuron_edge_fraction is not None \
                        else edge_step_fraction
    _n_per_edge       = max(0, neuron_steps_per_edge_step)
    _neuron_tol       = neuron_tolerance if neuron_tolerance is not None else accuracy_tolerance

    thresholds: list[float] = []
    if snapshot_interval and snapshot_interval > 0:
        n_snaps    = max(1, int(round(accuracy_tolerance / snapshot_interval)))
        thresholds = [round(snapshot_interval * (i + 1), 8) for i in range(n_snaps)]

    print("\n" + "=" * 70)
    print(f"  BRANCH: {label}")
    print(f"  Anchor={anchor_accuracy:.4f}  Floor={floor:.4f}  "
          f"Tolerance={accuracy_tolerance:.4f}")
    print("=" * 70)

    net        = PrunableNetworkConv(copy.deepcopy(l1_model_copy), eps=0.0)
    best_model = copy.deepcopy(net.model)
    best_acc   = _extract_accuracy(
        evaluate_model(net.model, X_test, y_test, device=device, timings=timings))
    print(f"[{label}] Start: acc={best_acc:.4f}  "
          f"fc_neurons={_count_fc_neurons(net.model)}  "
          f"graph_edges={net.graph.number_of_edges():,}")

    next_snap_idx        = 0
    total_steps          = 0
    neuron_steps_done    = 0
    neuron_exhausted     = False
    edge_exhausted       = False
    neuron_tol_exhausted = False

    for step in range(1, MAX_STEPS + 1):
        if neuron_exhausted and edge_exhausted:
            print(f"[{label}] Both phases exhausted.")
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
        active_edge_frac  = edge_step_fraction if not neuron_phase_done else _post_neuron_frac
        pre_step_model    = copy.deepcopy(net.model)

        if do_neuron:
            kwargs = dict(neuron_step_kwargs)
            kwargs["max_neurons"] = neurons_per_step
            n_removed, _ = neuron_step_fn(net, **kwargs)
            if n_removed == 0:
                neuron_exhausted = True
                net._model       = pre_step_model
                net.resync_graph()
                print(f"[{label}] Step {step}: neuron exhausted.")
                continue
            net.remove_lonely_neurons()
            neuron_steps_done += 1
            phase_label = "neuron"
        else:
            ordered     = _get_ordered_layers(net.model)
            G           = net.graph
            spatial_sizes = G.graph["spatial_sizes"]
            total_pairs = sum(
                (m.in_channels * m.out_channels) if isinstance(m, nn.Conv2d)
                else (G.graph["layers"][li] * m.out_features)
                for li, (_, m) in enumerate(ordered)
            )
            active_pairs = total_pairs - sum(
                int(_conv_edge_weight(m, j, i, spatial_sizes[li]) == 0.0)
                for li, (_, m) in enumerate(ordered)
                for j in range(m.out_channels if isinstance(m, nn.Conv2d) else m.out_features)
                for i in range(m.in_channels if isinstance(m, nn.Conv2d) else G.graph["layers"][li])
            )
            pairs_to_prune = max(1, int(active_pairs * active_edge_frac))
            target_pairs   = active_pairs - pairs_to_prune
            if target_pairs <= 0:
                edge_exhausted = True
                continue
            amount = _compute_edge_prune_amount(net, target_pairs)
            if not (0.0 < amount < 1.0):
                edge_exhausted = True
                continue
            edge_step_fn(net, amount=amount, **edge_step_kwargs)
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
              f"fc_neurons={_count_fc_neurons(net.model)}  "
              f"graph_edges={net.graph.number_of_edges():,}")

        _traj_record(label, total_steps, phase_label, acc, drop, net)
        total_steps += 1

        if drop <= accuracy_tolerance:
            best_model = copy.deepcopy(net.model)
            best_acc   = acc

            if (not neuron_tol_exhausted and neuron_tolerance is not None
                    and drop >= _neuron_tol):
                neuron_tol_exhausted = True
                print(f"[{label}] Step {step}: neuron tol cap "
                      f"(drop={drop:.4f} ≥ {_neuron_tol:.4f}).")

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

    print(f"\n[{label}] Done. {total_steps} steps. "
          f"acc={best_acc:.4f}  fc_neurons={_count_fc_neurons(net.model)}")
    eval_nn(net.model, X_test=X_test, y_test=y_test,
            device=device, timings=timings, evaluate_fn=evaluate_model)
    _noc_and_record(f"{label} [final]", net, X_test, y_test, device, timings)


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


def _print_table(title_lines, row_stages, base_stage):
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
          f"{'Graph Edges':^{w_val}}|{'Graph Nodes':^{w_val}}|"
          f"{'Batch Inf. (ms)':^{w_val}}|{'FLOPs (FC)':^{w_val}}|"
          f"{'Model Size (KB)':^{w_val}}|"
          f"{'F1':^{w_val}}|{'Precision':^{w_val}}|{'Recall':^{w_val}}|")
    print(sep)

    b = base_stage
    for s in row_stages:
        is_b   = (s is b)
        tag    = "  (baseline)" if is_b else ""
        acc_s  = f'{s["accuracy"]:>10.4f}{tag}'        if is_b else _fmt_delta(s["accuracy"],  b["accuracy"])
        cost_s = f'{s["cost"]:>10.4f}{tag}'             if is_b else _fmt_delta(s["cost"],      b["cost"])
        spc_s  = f'{s["sp_cost"]:>10.4f}{tag}'          if is_b else _fmt_delta(s["sp_cost"],   b["sp_cost"])
        edge_s = f'{s["edges"]:>10,}{tag}'               if is_b else _fmt_int(s["edges"],       b["edges"])
        neur_s = f'{s["neurons"]:>10,}{tag}'             if is_b else _fmt_int(s["neurons"],     b["neurons"])
        bat_s  = f'{s["batch_ms"]:>7.2f} ms{tag}'       if is_b else _fmt_ms(s["batch_ms"],     b["batch_ms"])
        flop_s = f'{s["flops"]:>10,}{tag}'               if is_b else _fmt_int(s["flops"],       b["flops"])
        size_s = f'{s["state_size_kb"]:>5.2f} KB{tag}'  if is_b else _fmt_kb(s["state_size_kb"],b["state_size_kb"])
        f1_s   = f'{s["f1"]:>10.4f}{tag}'               if is_b else _fmt_delta(s["f1"],         b["f1"])
        prec_s = f'{s["precision"]:>10.4f}{tag}'        if is_b else _fmt_delta(s["precision"],  b["precision"])
        rec_s  = f'{s["recall"]:>10.4f}{tag}'           if is_b else _fmt_delta(s["recall"],     b["recall"])

        print(f"|{s['name']:<{w_name}}|{acc_s:>{w_val}}|{cost_s:>{w_val}}|"
              f"{spc_s:>{w_val}}|{edge_s:>{w_val}}|{neur_s:>{w_val}}|"
              f"{bat_s:>{w_val}}|{flop_s:>{w_val}}|{size_s:>{w_val}}|"
              f"{f1_s:>{w_val}}|{prec_s:>{w_val}}|{rec_s:>{w_val}}|")
        print(sep)
    print()


def print_summary_tables():
    if not stages:
        return
    base = stages[0]
    snap_rows   = [s for s in stages if "@ drop≤" in s["name"]]
    final_rows  = [s for s in stages if s["name"].endswith("[final]")]
    anchor_row  = next((s for s in stages if s["name"] == "After L1 25% pre-pruning"), None)

    title_common = [
        "Model: LeNet-5 (conv1/c3/c5/f6/output) | Dataset: CIFAR-10",
        f"Tolerance: ≤{ACCURACY_TOLERANCE*100:.1f}% drop from POST-L1 anchor",
        "Graph: channel-level (Conv2d channels + FC neurons)",
        "Edge step: kernel-level for Conv2d, weight-block for Linear-after-Conv",
        "A=IterCC+Bridge  B=IterCC+L2  D=Taylor+L2  G=Taylor+Bridge",
        "Deltas relative to Baseline.",
    ]

    if snap_rows:
        _print_table(
            title_lines=["ISO-ACCURACY TABLE — snapshot at each drop threshold"]
                        + title_common,
            row_stages=[base] + snap_rows,
            base_stage=base,
        )

    _print_table(
        title_lines=["FINAL TABLE — each branch at its stopping point"]
                    + title_common,
        row_stages=[base] + ([anchor_row] if anchor_row else []) + final_rows,
        base_stage=base,
    )


def save_trajectories(path_csv="trajectory_LeNet5_CIFAR10.csv",
                      path_json="trajectory_LeNet5_CIFAR10.json"):
    if not trajectory:
        return
    import csv, json
    fields = list(trajectory[0].keys())
    with open(path_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(trajectory)
    print(f"[trajectory] {len(trajectory)} rows → {path_csv}")
    with open(path_json, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"[trajectory] {len(trajectory)} rows → {path_json}")


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

net = PrunableNetworkConv(model, eps=0.0)
print(net)
print(f"Graph layers: {net.graph.graph['layers']}")
print(f"Layer types : {net.graph.graph['layer_types']}")
print(f"Spatial sizes: {net.graph.graph['spatial_sizes']}")

# ==================================================================
# 2. Baseline evaluation
# ==================================================================

print("\n=== Baseline ===")
eval_nn(net.model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
_noc_and_record("Baseline", net, X_test, y_test, device, timings)

# ==================================================================
# 3. Iterative L1 edge pre-pruning (25%)
# ==================================================================

original_edge_count = net.graph.number_of_edges()
total_weights       = sum(p.numel() for name, p in net.model.named_parameters()
                          if "weight" in name and p.dim() >= 2)
target_removals     = round(total_weights * l1_target)

for step in range(1, l1_steps + 1):
    print(f"L1 step {step}/{l1_steps}")
    already_removed   = total_weights - sum(
        int((p != 0).sum().item())
        for name, p in net.model.named_parameters() if "weight" in name and p.dim() >= 2
    )
    remaining = target_removals - already_removed
    amount    = (already_removed + round(remaining / (l1_steps - step + 1))) / total_weights
    if amount >= 1.0:
        break
    net.prune_links_l1_global(amount=amount)
    fine_tune(net.model, X_train, y_train, device=device, lr=lr,
              epochs=ft_epochs, seed=seed + step)
    active = sum(int((p != 0).sum().item()) for name, p in net.model.named_parameters()
                 if "weight" in name and p.dim() >= 2)
    print(f"  Active weights after step {step}: {active:,} / {total_weights:,} "
          f"({active/total_weights:.1%})")

net.commit_pruning()
net.remove_lonely_neurons()

print("\n=== After L1 25% pre-pruning ===")
eval_nn(net.model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
_noc_and_record("After L1 25% pre-pruning", net, X_test, y_test, device, timings)

# ==================================================================
# 4. Checkpoint + anchor
# ==================================================================

l1_model_copy   = copy.deepcopy(net.model)
anchor_accuracy = stages[-1]["accuracy"]

print(f"\n[Checkpoint] anchor (POST-L1) = {anchor_accuracy:.4f}")
print(f"[Checkpoint] fc_neurons = {_count_fc_neurons(net.model)}")
print(f"[Checkpoint] graph nodes = {net.graph.number_of_nodes()}, "
      f"edges = {net.graph.number_of_edges():,}")

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
    edge_step_fn=_conv_bridge_edge_step,
    edge_step_kwargs={},
    edge_step_fraction=BRIDGE_STEP_FRACTION,
    post_neuron_edge_fraction=0.05,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)

# ==================================================================
# 6. Branch B — IterCC Neuron + L2
# ==================================================================

run_accuracy_budget_branch(
    label="B: IterCC Neuron + L2",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=iterative_commcost_neuron_pruning,
    neuron_step_kwargs=_iterCC_kwargs,
    edge_step_fn=_conv_l2_edge_step,
    edge_step_kwargs={},
    post_neuron_edge_fraction=0.05,
    neuron_steps_per_edge_step=1,
    neurons_per_step=3,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
)

# ==================================================================
# 7. Branch D — Taylor Neuron + L2
# ==================================================================

run_accuracy_budget_branch(
    label="D: Taylor Neuron + L2",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=_taylor_neuron_step,
    neuron_step_kwargs=_taylor_kwargs,
    edge_step_fn=_conv_l2_edge_step,
    edge_step_kwargs={},
    post_neuron_edge_fraction=0.05,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)

# ==================================================================
# 8. Branch G — Taylor Neuron + Bridge-Link
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
    edge_step_fn=_conv_bridge_edge_step,
    edge_step_kwargs={},
    edge_step_fraction=BRIDGE_STEP_FRACTION,
    post_neuron_edge_fraction=0.05,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)

# ==================================================================
# GA branches (Q/V/W/V2) — commented out, uncomment to enable
# ==================================================================
"""
from pruning_meta import ga_neuron_step_biased_v2, ga_neuron_step_memetic_biased

_ga_kwargs = dict(
    N=N, grouping_cap=10_000.0,
    low_mag_percentile=iterCC_low_mag_pct,
    pop_size=30, n_generations=20,
    tournament_k=3, mutation_rate=0.2,
    elite_frac=0.1, ga_seed=seed, debug=False,
    alpha=0.6, beta=0.4,
)

run_accuracy_budget_branch(
    label="Q: GA CC_Biased v2 + Bridge-Link",
    l1_model_copy=l1_model_copy, anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    device=device, timings=timings,
    neuron_step_fn=ga_neuron_step_biased_v2,
    neuron_step_kwargs=dict(**_ga_kwargs, rescore_interval=5),
    edge_step_fn=_conv_bridge_edge_step, edge_step_kwargs={},
    edge_step_fraction=BRIDGE_STEP_FRACTION, post_neuron_edge_fraction=0.05,
    neuron_steps_per_edge_step=1, micro_ft_epochs=2,
    neuron_tolerance=0.005, neurons_per_step=3,
)
"""

# ==================================================================
# 9. Summary
# ==================================================================

print_summary_tables()
#save_trajectories()
