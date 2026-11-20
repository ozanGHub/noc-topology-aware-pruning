"""
main_AlexNet_CIFAR10_e2e_AccuracyBudget_Metaheuristic_v4.py
============================================================
Accuracy-budget pruning for AlexNet (CIFAR-10 adapted) on CIFAR-10.

Architecture (channel-level graph)
  Input(3) → conv1(64) → conv2(192) → conv3(384) → conv4(256) → conv5(256)
           → c6(1024) → c7(512) → output(10)

  Hidden nodes: 64+192+384+256+256+1024+512 = 2688
  spatial_size for c6 (Linear 4096→1024 after conv5 out 256): 4096//256 = 16
  NoC mesh: 11×11 = 121 cores  (~5.6 hidden nodes/core)

Pruning strategy (middle path)
  Conv2d  transitions: kernel-level  — zero entire W[j,i,:,:]
  Linear-after-Conv   : block-level  — zero W[j, i*16:(i+1)*16]
  Linear-after-Linear : weight-level — zero W[j,i]

  Neuron pruning physically removes only Linear nodes (c6, c7).
  Conv channels (6 layers) remain structurally intact.

Anchor: BASELINE accuracy (stages[0]).   ← v2: was POST-L1

v2 changes (internal review outcomes/fixes):
  1. Anchor = BASELINE(was POST-L1); tolerance = 3 pp
  2. Accuracy/F1/precision/recall deltas in absolute pp
  3. Snapshots record the LAST state within the threshold (pre-step state),
     not the first state that crosses it.
  4. EDGE_STEP_FRACTION / BRIDGE_STEP_FRACTION 0.02 → 0.01; post_neuron 0.05 → 0.02
  5. L1 pre-pruning 25% → 10%

v3 change:
  6. Neuron candidates restricted to removable FC layers (c6, c7) for ALL
     branches. In v2, Taylor ranked conv channels too; PrunableNetworkConv
     silently ignores conv nodes, so D/G removed nothing and exhausted the
     neuron phase at step 1, while IterCC's pool also included conv nodes.
     IterCC now uses an FC-only wrapper that counts only real removals.

v4 change (budget consumption; pruning methods unchanged):
  7. A step that lands over the budget is not an immediate stop:
       a) up to RECOVERY_FT_ROUNDS extra fine-tune rounds, re-evaluated;
       b) if still over: revert the step, halve the step fraction (edge
          steps) and retry, down to MIN_STEP_SCALE of the nominal fraction;
          a neuron step over budget reverts and hands over to edge-only;
       c) stop after MAX_FAILS_AT_MIN failed retries at the minimum scale.
     The final state is always a measured state within tolerance.

Active branches: A (IterCC+Bridge), B (IterCC+L2), D (Taylor+L2), G (Taylor+Bridge)
GA branches Q/V/W/V2 commented out.
"""

from __future__ import annotations

import copy
import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from DNN_AlexNet_CIFAR10 import load_or_train, fine_tune, evaluate_model
from prunable_network_conv import PrunableNetworkConv
from eval_helper import eval_noc, eval_nn
from mapping_helpers import build_wctg_structural, compute_structural_cost
from pruning import select_neurons_high_indegree_low_magnitude, _commcost_score_neuron
from grouping_heuristic import indegree_grouping

# ==================================================================
# Configuration
# ==================================================================

data_dir:     str   = "./data"
model_path:   str   = "DNN_AlexNet_CIFAR10.pth"
seed:         int   = 42
batch_size:   int   = 128
lr:           float = 1e-3
epochs:       int   = 200
weight_decay: float = 1e-4
force_train:  bool  = False

# --- Shared L1 pre-pruning ---
# v2: reduced from 0.25 → 0.10 to limit pre-pruning accuracy cost (internal review pt.5)
l1_target: float = 0.10   # was 0.25
l1_steps:  int   = 5
ft_epochs: int   = 10

# v2: ACCURACY_TOLERANCE = 3 pp from BASELINE (internal review pt.1)
# v2: SNAPSHOT_INTERVAL  = 1 pp → thresholds [1pp, 2pp, 3pp] (internal review pt.3)
# v2: EDGE/BRIDGE step fractions halved for finer steps (internal review pt.4)
ACCURACY_TOLERANCE:         float = 0.03   # was 0.01 (3 pp from BASELINE)
SNAPSHOT_INTERVAL:          float = 0.01   # every 1 pp
MICRO_FT_EPOCHS:            int   = 2
NEURON_STEPS_PER_EDGE_STEP: int   = 1
NEURONS_PER_STEP:           int   = 3
EDGE_STEP_FRACTION:         float = 0.01   # was 0.02
BRIDGE_STEP_FRACTION:       float = 0.01   # was 0.02
MAX_STEPS:                  int   = 2000
RECOVERY_FT_ROUNDS:         int   = 2       # v4: extra fine-tunes before calling a step a violation
MIN_STEP_SCALE:             float = 0.125   # v4: smallest step multiplier after backtracking
MAX_FAILS_AT_MIN:           int   = 3       # v4: failed retries at min scale before stopping

# NoC — 11×11 mesh (2688 hidden nodes, ~5.6/core)
P:         int = 121
N:         int = 121
mesh_rows: int = int(math.sqrt(P))
mesh_cols: int = int(math.sqrt(P))

iterCC_low_mag_pct:   float = 90.0 #75.0?
iterCC_degree_q:      float = 0.25
iterCC_min_in_degree: int   = 0

_L1_LABEL = f"After L1 {l1_target:.0%} pre-pruning"

# ==================================================================
# Metrics
# ==================================================================

stages:     list[dict] = []
trajectory: list[dict] = []


def _traj_record(branch, step, phase, acc, drop, net):
    linears = [m for m in net.model.modules() if isinstance(m, nn.Linear)]
    n_edges = sum(int((m.weight.data != 0).sum().item()) for m in linears)
    trajectory.append({
        "branch": branch, "step": step, "phase": phase,
        "accuracy": acc, "accuracy_drop": drop,
        "fc_neurons": _count_fc_neurons(net.model),
        "fc_edges": n_edges,
        "graph_edges": net.graph.number_of_edges(),
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
    total = 0
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            total += mod.in_features * mod.out_features
    return 2 * total


def _count_fc_neurons(model: nn.Module) -> int:
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return sum(m.out_features for m in linears[:-1])


def _get_ordered_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
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
          f"sp_cost={sp_cost:.0f}  graph_edges={net.graph.number_of_edges():,}  "
          f"fc_neurons={_count_fc_neurons(net.model)}")


# ==================================================================
# Conv-aware edge pruning
# ==================================================================

def _conv_edge_weight(m: nn.Module, out_j: int, in_i: int, sp: int) -> float:
    if isinstance(m, nn.Conv2d):
        return float(m.weight.data[out_j, in_i].abs().sum().item())
    return float(m.weight.data[out_j, in_i * sp : (in_i + 1) * sp].abs().sum().item())


def _conv_zero_edge(m: nn.Module, out_j: int, in_i: int, sp: int) -> None:
    with torch.no_grad():
        if isinstance(m, nn.Conv2d):
            m.weight.data[out_j, in_i] = 0.0
        else:
            m.weight.data[out_j, in_i * sp : (in_i + 1) * sp] = 0.0


def _conv_bridge_edge_step(net: PrunableNetworkConv, *, amount: float, **_) -> None:
    noc_tmp = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                       title="_bridge_tmp", visualize=False)
    node_to_group = {node: gid for gid, members in enumerate(noc_tmp.groups)
                     for node in members}

    G             = net.graph
    spatial_sizes = G.graph["spatial_sizes"]
    ordered       = _get_ordered_layers(net.model)

    total_weights = sum(m.weight.numel() for _, m in ordered)
    already_zero  = sum(int((m.weight.data == 0).sum().item()) for _, m in ordered)
    active_edges  = total_weights - already_zero
    edges_to_prune = max(1, int(round(amount * total_weights)) - already_zero)
    edges_to_prune = min(edges_to_prune, active_edges - 1)
    if edges_to_prune <= 0:
        return

    bridge_candidates: list[tuple[float, int, int, int]] = []
    for (src_node, dst_node, _) in G.edges(data=True):
        src_grp = node_to_group.get(src_node)
        dst_grp = node_to_group.get(dst_node)
        if src_grp is None or dst_grp is None or src_grp == dst_grp:
            continue
        layer_idx = src_node[0]
        in_i, out_j = src_node[1], dst_node[1]
        m  = ordered[layer_idx][1]
        sp = spatial_sizes[layer_idx]
        bridge_candidates.append((_conv_edge_weight(m, out_j, in_i, sp),
                                   layer_idx, out_j, in_i))

    if not bridge_candidates:
        _conv_l2_edge_step(net, amount=amount)
        return

    bridge_candidates.sort(key=lambda x: x[0])
    pruned = 0
    for w_scalar, layer_idx, out_j, in_i in bridge_candidates:
        if pruned >= edges_to_prune:
            break
        _conv_zero_edge(ordered[layer_idx][1], out_j, in_i, spatial_sizes[layer_idx])
        pruned += 1

    net.remove_lonely_neurons()
    net.commit_pruning()
    net.resync_graph()


def _conv_l2_edge_step(net: PrunableNetworkConv, *, amount: float, **_) -> None:
    G             = net.graph
    spatial_sizes = G.graph["spatial_sizes"]
    ordered       = _get_ordered_layers(net.model)

    all_edges: list[tuple[float, int, int, int]] = []
    for layer_idx, (name, m) in enumerate(ordered):
        sp       = spatial_sizes[layer_idx]
        src_size = G.graph["layers"][layer_idx]
        out_size = (m.out_channels if isinstance(m, nn.Conv2d) else m.out_features)
        for out_j in range(out_size):
            for in_i in range(src_size):
                w = _conv_edge_weight(m, out_j, in_i, sp)
                all_edges.append((w, layer_idx, out_j, in_i))

    total   = len(all_edges)
    n_zero  = sum(1 for (w, *_) in all_edges if w == 0.0)
    n_prune = max(1, int(round(amount * total)) - n_zero)
    n_prune = min(n_prune, total - n_zero - 1)
    if n_prune <= 0:
        return

    all_edges.sort(key=lambda x: x[0])
    pruned = 0
    for w_scalar, layer_idx, out_j, in_i in all_edges:
        if pruned >= n_prune:
            break
        if w_scalar == 0.0:
            continue
        _conv_zero_edge(ordered[layer_idx][1], out_j, in_i, spatial_sizes[layer_idx])
        pruned += 1

    net.commit_pruning()
    net.resync_graph()


# ==================================================================
# FC-only neuron candidates (conv channels cannot be removed)
# ==================================================================

def _removable_fc_node_layers(net: PrunableNetworkConv) -> set[int]:
    n_lin = len(net.linear_names)
    nl2li = net.graph.graph["node_layer_to_linear_idx"]
    return {nl for nl, li in nl2li.items() if li <= n_lin - 2}   # excludes output


def _itercc_fc_neuron_step(net: PrunableNetworkConv, *, max_neurons: int,
                           N: int, grouping_cap: float = 10_000.0,
                           low_mag_percentile: float = 75.0,
                           degree_quantile: float | None = 0.25,
                           min_in_degree: int = 0,
                           min_score: float = 0.0,
                           debug: bool = False) -> tuple[int, list]:
    removed = 0
    history: list = []
    while removed < max_neurons:
        G = net.graph
        try:
            groups = indegree_grouping(G, N, grouping_cap)
        except RuntimeError:
            break
        node_to_group = {n: gid for gid, members in enumerate(groups) for n in members}
        group_to_nodes = {gid: members for gid, members in enumerate(groups)}

        fc_layers  = _removable_fc_node_layers(net)
        candidates = [n for n in select_neurons_high_indegree_low_magnitude(
                          G, low_mag_percentile=low_mag_percentile,
                          degree_quantile=degree_quantile,
                          min_in_degree=min_in_degree, debug=False)
                      if n[0] in fc_layers]
        if not candidates:
            break

        scored = sorted(((n, _commcost_score_neuron(G, n, node_to_group, group_to_nodes))
                         for n in candidates), key=lambda t: t[1], reverse=True)
        best_node, best_score = scored[0]
        if best_score <= min_score:
            break

        before = _count_fc_neurons(net.model)
        net.prune_neurons([best_node])
        if _count_fc_neurons(net.model) >= before:
            break
        history.append((best_node, best_score))
        removed += 1
    return removed, history


# ==================================================================
# Neuron pruning — Taylor (FC-only candidates)
# ==================================================================

def _taylor_neuron_step(net: PrunableNetworkConv, *, max_neurons: int,
                        X_calib, y_calib, device,
                        n_batches: int = 10, batch_sz: int = 128) -> tuple[int, list]:
    if isinstance(X_calib, np.ndarray):
        X_calib = torch.from_numpy(X_calib.astype(np.float32))
    if isinstance(y_calib, np.ndarray):
        y_calib = torch.from_numpy(y_calib.astype(np.int64))

    model   = net.model
    ordered = _get_ordered_layers(model)
    G       = net.graph

    importance: dict[tuple[int, int], float] = {}
    for layer_idx, (name, m) in enumerate(ordered):
        node_layer = layer_idx + 1
        out_size = m.out_channels if isinstance(m, nn.Conv2d) else m.out_features
        for j in range(out_size):
            importance[(node_layer, j)] = 0.0

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
            for layer_idx, (name, m) in enumerate(ordered):
                node_layer = layer_idx + 1
                if m.weight.grad is None:
                    continue
                if isinstance(m, nn.Conv2d):
                    scores = (m.weight.data * m.weight.grad).abs().sum(dim=(1, 2, 3))
                else:
                    scores = (m.weight.data * m.weight.grad).abs().sum(dim=1)
                for j in range(scores.shape[0]):
                    importance[(node_layer, j)] += float(scores[j].item()) / n_batches

    model.zero_grad()
    if not was_training:
        model.eval()

    fc_layers  = _removable_fc_node_layers(net)
    candidates = {node: score for node, score in importance.items()
                  if node[0] in fc_layers}
    if not candidates:
        return 0, []

    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1])
    to_prune     = [node for node, _ in sorted_nodes[:max_neurons]]

    before = _count_fc_neurons(model)
    net.prune_neurons(to_prune)
    net.remove_lonely_neurons()
    net.commit_pruning()
    net.resync_graph()
    return before - _count_fc_neurons(net.model), []


# ==================================================================
# Edge amount helper
# ==================================================================

def _compute_edge_prune_amount(net: PrunableNetworkConv, target_pairs: int) -> float:
    ordered       = _get_ordered_layers(net.model)
    G             = net.graph
    spatial_sizes = G.graph["spatial_sizes"]
    total = zero = 0
    for layer_idx, (name, m) in enumerate(ordered):
        sp       = spatial_sizes[layer_idx]
        src_size = G.graph["layers"][layer_idx]
        out_size = m.out_channels if isinstance(m, nn.Conv2d) else m.out_features
        n_pairs  = src_size * out_size
        total   += n_pairs
        for out_j in range(out_size):
            for in_i in range(src_size):
                if _conv_edge_weight(m, out_j, in_i, sp) == 0.0:
                    zero += 1
    active   = total - zero
    to_zero  = max(0, active - target_pairs)
    return float((zero + to_zero) / total) if total else 0.0


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
    accuracy_tolerance:        float = ACCURACY_TOLERANCE,
    snapshot_interval:         float = SNAPSHOT_INTERVAL,
    edge_step_fraction:        float = EDGE_STEP_FRACTION,
    post_neuron_edge_fraction: float | None = None,
    neuron_steps_per_edge_step: int  = NEURON_STEPS_PER_EDGE_STEP,
    micro_ft_epochs:           int   = MICRO_FT_EPOCHS,
    neuron_tolerance:          float | None = None,
    neurons_per_step:          int   = 1,
    max_steps:                 int   = MAX_STEPS,
    recovery_ft_rounds:        int   = RECOVERY_FT_ROUNDS,
    min_step_scale:            float = MIN_STEP_SCALE,
    max_fails_at_min:          int   = MAX_FAILS_AT_MIN,
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
          f"Tolerance={accuracy_tolerance:.4f} ({accuracy_tolerance*100:.1f} pp)")
    if neuron_tolerance is not None:
        print(f"  Neuron phase cap: {_neuron_tol:.4f}")
    if thresholds:
        print(f"  Snapshot thresholds: {[f'{t*100:.1f} pp' for t in thresholds]}")
    print(f"  Edge frac (interleaved/post): {edge_step_fraction:.4f}/{_post_neuron_frac:.4f}  "
          f"N:E ratio={_n_per_edge}:1  FT epochs={micro_ft_epochs}  Max steps={max_steps}")
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
    step_scale           = 1.0
    fails_at_min         = 0

    def _ft_eval(ft_seed: int) -> float:
        reapply_zero_masks(net)
        fine_tune(net.model, X_train, y_train, device=device, lr=lr,
                  epochs=micro_ft_epochs, seed=ft_seed)
        net.commit_pruning()
        net.resync_graph()
        return _extract_accuracy(
            evaluate_model(net.model, X_test, y_test, device=device, timings=timings))

    for step in range(1, max_steps + 1):
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
        active_edge_frac  = (edge_step_fraction if not neuron_phase_done
                             else _post_neuron_frac) * step_scale
        pre_step_model    = copy.deepcopy(net.model)   # saved BEFORE step

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
            G             = net.graph
            spatial_sizes = G.graph["spatial_sizes"]
            ordered       = _get_ordered_layers(net.model)
            total_pairs   = sum(
                G.graph["layers"][li] * (m.out_channels if isinstance(m, nn.Conv2d)
                                         else m.out_features)
                for li, (_, m) in enumerate(ordered)
            )
            active_pairs = total_pairs - sum(
                int(_conv_edge_weight(m, j, i, spatial_sizes[li]) == 0.0)
                for li, (_, m) in enumerate(ordered)
                for j in range(m.out_channels if isinstance(m, nn.Conv2d) else m.out_features)
                for i in range(G.graph["layers"][li])
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

        acc  = _ft_eval(seed + step)
        drop = anchor_accuracy - acc

        # v4: recovery fine-tune rounds before declaring a violation
        for r in range(1, recovery_ft_rounds + 1):
            if drop <= accuracy_tolerance:
                break
            print(f"[{label}] Step {step:3d} ({phase_label:6s}): drop={drop:+.4f} "
                  f"over budget -> recovery round {r}/{recovery_ft_rounds}")
            acc  = _ft_eval(seed + step + 1000 * r)
            drop = anchor_accuracy - acc

        # v4: still over budget -> revert this step, shrink step size, retry
        if drop > accuracy_tolerance:
            net._model = pre_step_model
            net.resync_graph()
            if do_neuron:
                neuron_exhausted = True
                print(f"[{label}] Step {step:3d}: neuron step over budget "
                      f"(drop={drop:+.4f}); reverted, switching to edge-only.")
            elif step_scale > min_step_scale:
                step_scale = max(min_step_scale, step_scale * 0.5)
                print(f"[{label}] Step {step:3d}: over budget (drop={drop:+.4f}); "
                      f"reverted, step scale -> {step_scale:.4f}")
            else:
                fails_at_min += 1
                print(f"[{label}] Step {step:3d}: over budget at min step scale "
                      f"({fails_at_min}/{max_fails_at_min}); reverted.")
                if fails_at_min >= max_fails_at_min:
                    print(f"[{label}] Budget consumed. Stopping.")
                    break
            continue

        fails_at_min = 0
        print(f"[{label}] Step {step:3d} ({phase_label:6s}): "
              f"acc={acc:.4f}  drop={drop:+.4f}  "
              f"fc_neurons={_count_fc_neurons(net.model)}  "
              f"graph_edges={net.graph.number_of_edges():,}  "
              f"scale={step_scale:.3f}")

        _traj_record(label, total_steps, phase_label, acc, drop, net)
        total_steps += 1

        best_model = copy.deepcopy(net.model)
        best_acc   = acc

        if (not neuron_tol_exhausted and neuron_tolerance is not None
                and drop >= _neuron_tol):
            neuron_tol_exhausted = True
            remaining = accuracy_tolerance - drop
            print(f"[{label}] Step {step}: neuron tol cap "
                  f"(drop={drop:.4f} ≥ {_neuron_tol:.4f}). "
                  f"Reserving {remaining:.4f} for edge-only ({_post_neuron_frac:.4f}).")

        # v2 fix (reviewer pt.3): record pre_step_model, not current state
        while (next_snap_idx < len(thresholds)
               and drop >= thresholds[next_snap_idx]):
            snap_pct   = thresholds[next_snap_idx] * 100
            snap_label = f"{label} @ drop≤{snap_pct:.1f}%"
            print(f"[{label}] → Snapshot at drop≤{snap_pct:.1f}%: '{snap_label}'")
            snap_net = PrunableNetworkConv(copy.deepcopy(pre_step_model), eps=0.0)
            _noc_and_record(snap_label, snap_net, X_test, y_test, device, timings)
            next_snap_idx += 1

    print(f"\n[{label}] Done. {total_steps} steps. "
          f"acc={best_acc:.4f}  fc_neurons={_count_fc_neurons(net.model)}")
    eval_nn(net.model, X_test=X_test, y_test=y_test,
            device=device, timings=timings, evaluate_fn=evaluate_model)
    _noc_and_record(f"{label} [final]", net, X_test, y_test, device, timings)


# ==================================================================
# Summary tables
# ==================================================================

def _fmt_acc_delta(cur, bv):
    """Absolute pp delta — used for accuracy, F1, precision, recall (internal review pt.2)."""
    pp   = (cur - bv) * 100
    sign = "+" if pp >= 0 else ""
    return f"{cur:>10.4f}  ({sign}{pp:.2f} pp)"

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
        # accuracy-like columns: absolute pp  (internal review pt.2)
        acc_s  = f'{s["accuracy"]:>10.4f}{tag}'        if is_b else _fmt_acc_delta(s["accuracy"],  b["accuracy"])
        cost_s = f'{s["cost"]:>10.4f}{tag}'             if is_b else _fmt_delta(s["cost"],          b["cost"])
        spc_s  = f'{s["sp_cost"]:>10.4f}{tag}'          if is_b else _fmt_delta(s["sp_cost"],       b["sp_cost"])
        edge_s = f'{s["edges"]:>10,}{tag}'               if is_b else _fmt_int(s["edges"],           b["edges"])
        neur_s = f'{s["neurons"]:>10,}{tag}'             if is_b else _fmt_int(s["neurons"],         b["neurons"])
        bat_s  = f'{s["batch_ms"]:>7.2f} ms{tag}'       if is_b else _fmt_ms(s["batch_ms"],         b["batch_ms"])
        flop_s = f'{s["flops"]:>10,}{tag}'               if is_b else _fmt_int(s["flops"],           b["flops"])
        size_s = f'{s["state_size_kb"]:>5.2f} KB{tag}'  if is_b else _fmt_kb(s["state_size_kb"],    b["state_size_kb"])
        f1_s   = f'{s["f1"]:>10.4f}{tag}'               if is_b else _fmt_acc_delta(s["f1"],         b["f1"])
        prec_s = f'{s["precision"]:>10.4f}{tag}'        if is_b else _fmt_acc_delta(s["precision"],  b["precision"])
        rec_s  = f'{s["recall"]:>10.4f}{tag}'           if is_b else _fmt_acc_delta(s["recall"],     b["recall"])

        print(f"|{s['name']:<{w_name}}|{acc_s:>{w_val}}|{cost_s:>{w_val}}|"
              f"{spc_s:>{w_val}}|{edge_s:>{w_val}}|{neur_s:>{w_val}}|"
              f"{bat_s:>{w_val}}|{flop_s:>{w_val}}|{size_s:>{w_val}}|"
              f"{f1_s:>{w_val}}|{prec_s:>{w_val}}|{rec_s:>{w_val}}|")
        print(sep)
    print()


def print_summary_tables():
    if not stages:
        return
    base       = stages[0]
    snap_rows  = [s for s in stages if "@ drop≤" in s["name"]]
    final_rows = [s for s in stages if s["name"].endswith("[final]")]
    anchor_row = next((s for s in stages if s["name"] == _L1_LABEL), None)

    title_common = [
        f"Snapshot interval: every {SNAPSHOT_INTERVAL*100:.1f} pp",
        "Model: AlexNet (conv1/conv2/conv3/conv4/conv5/c6/c7/output) | Dataset: CIFAR-10",
        f"Tolerance: ≤{ACCURACY_TOLERANCE*100:.1f} pp drop from BASELINE",
        "Graph: channel-level (Conv2d channels + FC neurons) | Mesh: 22×22=484 cores",
        "Edge step: kernel-level for Conv2d, weight-block (sp=16) for c6, weight for c7",
        "A=IterCC+Bridge  B=IterCC+L2  D=Taylor+L2  G=Taylor+Bridge",
        "Deltas: accuracy/F1/precision/recall in pp; cost/edges/FLOPs/size relative to Baseline.",
    ]

    if snap_rows:
        _print_table(
            title_lines=["ISO-ACCURACY TABLE — snapshot at each accuracy-drop threshold"]
                        + title_common,
            row_stages=[base] + snap_rows,
            base_stage=base,
        )

    _print_table(
        title_lines=["FINAL TABLE — each branch at its stopping point (last within tolerance)"]
                    + title_common,
        row_stages=[base] + ([anchor_row] if anchor_row else []) + final_rows,
        base_stage=base,
    )


def save_trajectories(path_csv="trajectory_AlexNet_CIFAR10.csv",
                      path_json="trajectory_AlexNet_CIFAR10.json"):
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
print(f"Graph layers     : {net.graph.graph['layers']}")
print(f"Layer types      : {net.graph.graph['layer_types']}")
print(f"Spatial sizes    : {net.graph.graph['spatial_sizes']}")
print(f"Graph nodes      : {net.graph.number_of_nodes():,}")
print(f"Graph edges      : {net.graph.number_of_edges():,}")

# ==================================================================
# 2. Baseline
# ==================================================================

print("\n=== Baseline ===")
eval_nn(net.model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
_noc_and_record("Baseline", net, X_test, y_test, device, timings)

# ==================================================================
# 3. Iterative L1 edge pre-pruning (10%)
# ==================================================================

original_edge_count = net.graph.number_of_edges()
total_weights = sum(p.numel() for name, p in net.model.named_parameters()
                    if "weight" in name and p.dim() >= 2)
target_removals = round(total_weights * l1_target)

for step in range(1, l1_steps + 1):
    print(f"L1 step {step}/{l1_steps}")
    already_removed = total_weights - sum(
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
    active = sum(int((p != 0).sum().item())
                 for name, p in net.model.named_parameters() if "weight" in name and p.dim() >= 2)
    print(f"  Active weights: {active:,} / {total_weights:,} ({active/total_weights:.1%})")

net.commit_pruning()
net.remove_lonely_neurons()

print(f"\n=== {_L1_LABEL} ===")
eval_nn(net.model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
_noc_and_record(_L1_LABEL, net, X_test, y_test, device, timings)

# ==================================================================
# 4. Checkpoint + anchor
#    v2: anchor = BASELINE (stages[0]), not post-L1 (internal review pt.1)
# ==================================================================

l1_model_copy   = copy.deepcopy(net.model)
anchor_accuracy = stages[0]["accuracy"]   # v2: BASELINE anchor

print(f"\n[Checkpoint] BASELINE anchor = {anchor_accuracy:.4f}")
l1_acc = stages[-1]["accuracy"]
print(f"[Checkpoint] Post-L1 accuracy = {l1_acc:.4f}  "
      f"(drop from baseline: {(anchor_accuracy - l1_acc)*100:.2f} pp)")
print(f"[Checkpoint] Remaining budget: {(ACCURACY_TOLERANCE - (anchor_accuracy - l1_acc))*100:.2f} pp")
print(f"[Checkpoint] fc_neurons = {_count_fc_neurons(net.model)}")
print(f"[Checkpoint] graph nodes = {net.graph.number_of_nodes():,}, "
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
# 5. Branch A — IterCC Neuron + Bridge
# ==================================================================

run_accuracy_budget_branch(
    label="A: IterCC Neuron + Bridge",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=_itercc_fc_neuron_step,
    neuron_step_kwargs=_iterCC_kwargs,
    edge_step_fn=_conv_bridge_edge_step,
    edge_step_kwargs={},
    edge_step_fraction=BRIDGE_STEP_FRACTION,
    post_neuron_edge_fraction=0.02,
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
    neuron_step_fn=_itercc_fc_neuron_step,
    neuron_step_kwargs=_iterCC_kwargs,
    edge_step_fn=_conv_l2_edge_step,
    edge_step_kwargs={},
    post_neuron_edge_fraction=0.02,
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
    post_neuron_edge_fraction=0.02,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)

# ==================================================================
# 8. Branch G — Taylor Neuron + Bridge
# ==================================================================

run_accuracy_budget_branch(
    label="G: Taylor Neuron + Bridge",
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
    post_neuron_edge_fraction=0.02,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2,
    neuron_tolerance=0.005,
    neurons_per_step=3,
)

# ==================================================================
# GA branches — uncomment to enable
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
    label="Q: GA CC_Biased v2 + Bridge",
    l1_model_copy=l1_model_copy, anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    device=device, timings=timings,
    neuron_step_fn=ga_neuron_step_biased_v2,
    neuron_step_kwargs=dict(**_ga_kwargs, rescore_interval=5),
    edge_step_fn=_conv_bridge_edge_step, edge_step_kwargs={},
    edge_step_fraction=BRIDGE_STEP_FRACTION, post_neuron_edge_fraction=0.02,
    neuron_steps_per_edge_step=1, micro_ft_epochs=2,
    neuron_tolerance=0.005, neurons_per_step=3,
)
"""

# ==================================================================
# 9. Summary
# ==================================================================

print_summary_tables()
save_trajectories()
