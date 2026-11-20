"""
main_LeNet_500_300_CIFAR10_e2e_AccuracyBudget_Metaheuristic_v3_fast.py
======================================================================
Accuracy-budget pruning comparison for a flat MLP on CIFAR-10.

v3_fast = v3_timing + speed-ups that do not change the pruning results
(verified on seed 42: identical accuracy/edges/neurons/steps for A, B, D, G):
  - _traj_record vectorised (always on).
  - read-only graph access WITHOUT the networkx copy made by `net.graph`
    (_graph_ro; always on, used only where the graph is just read).
  - BRIDGE_FAST=1         vectorised Bridge candidate selection
    BRIDGE_CHECK=1        (verification only) also runs the original loop and asserts equality
  - BRIDGE_GROUPS_FAST=1  Bridge groups via indegree_grouping instead of eval_noc (skips the SA)
    BRIDGE_GROUPS_CHECK=1 (verification only) asserts both group lists are identical
  Fast run:  BRIDGE_FAST=1 BRIDGE_GROUPS_FAST=1 python <this file>
v3 changes (on top of v2):
  A. Fine-tuning uses the SAME augmentation as baseline training (random crop
     pad=4 + horizontal flip, applied batch-wise on the device).  The 0% L1
     control run showed that the plain (non-augmented) fine-tune alone costs
     ~3.7 pp, i.e. the pre-pruning "cost" was a fine-tune artifact, not a
     pruning cost.  Switch with FT_AUGMENT.
     - fine_tune_lr: updated to 1e-4 from 5e-4 (lowered by 2.5x) to preserve accuracy (internal review pt.5)
  B. Noise-robust stop rule (ported from AlexNet v4).  A step that lands over
     the budget is not an immediate stop:
       a) up to RECOVERY_FT_ROUNDS extra fine-tune rounds, re-evaluated;
       b) if still over: revert the step, halve the step fraction (edge
          steps) and retry, down to MIN_STEP_SCALE of the nominal fraction;
          a neuron step over budget reverts and hands over to edge-only;
       c) stop after MAX_FAILS_AT_MIN failed retries at the minimum scale.
     The final state is always a measured state within tolerance.
  NOTE: the baseline checkpoint is still the best-test-accuracy epoch of
  DNN_LeNet_500_300_CIFAR10.py (no validation split); retraining is needed
  to change that.

v2 changes (internal review outcomes):
  1. Anchor: BASELINE (was POST-L1); tolerance = 5 pp for CIFAR-10.
  2. Accuracy deltas printed as absolute pp; other metrics stay relative %.
  3. Snapshots record the LAST state within the threshold (pre-step state),
     not the first state that crosses it.
  4. EDGE_STEP_FRACTION / BRIDGE_STEP_FRACTION reduced 0.02 → 0.01 for
     finer pruning steps; post_neuron_edge_fraction 0.05 → 0.02.
  5. L1 pre-pruning reduced 25% → 10% to limit pre-pruning accuracy cost.
  6. iterCC_low_mag_pct 90 → 75: wider candidate pool for dense CIFAR-10 graphs.

Architecture : LeNet 3072-500-300-10  (same hidden topology as LeNet 500-300)
Dataset      : CIFAR-10  (50k train / 10k test, 3×32×32 → 3072-dim flat)
NoC mapping  : P=196, N=196  (14×14 mesh)

Branches
--------
  A  : IterCC Neuron + Bridge-Link    (heuristic, primary)
  B  : IterCC Neuron + L2 edge
  #C  : IterCC Neuron + L1 edge
  D  : Taylor Neuron + L2 edge
  #E  : Taylor Neuron + L1 edge
  #F  : L1 Neuron + L2 edge
  G  : Taylor Neuron + Bridge-Link
  H  : Random Neuron + L2 edge
  I  : Activation Neuron + L2 edge
  K  : L2 Neuron + L2 edge
  Q  : GA CC_Biased v2 + Bridge-Link  (metaheuristic, bridge 0.01)
  V  : GA Memetic Biased + Bridge     (metaheuristic, bridge 0.01)
  W  : GA Memetic Biased + L2 edge    (metaheuristic)
  V2 : GA Memetic Biased + Bridge     (metaheuristic, bridge 0.005)
"""

from __future__ import annotations

import copy
import math
import os
import time
from collections import defaultdict
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import torch.nn.functional as F

from DNN_LeNet_500_300_CIFAR10 import (
    load_or_train, fine_tune as _fine_tune_plain, evaluate_model, set_seed,
)
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
model_path:   str   = "DNN_LeNet_500_300_CIFAR10.pth"
seed:         int   = 42
test_size:    float = 0.2
batch_size:   int   = 128
lr:           float = 1e-3
#fine_tune_lr: float = 5e-5  #was 5e-4
fine_tune_lr: float = 1e-4  #was 5e-4
epochs:       int   = 300
weight_decay: float = 1e-4
threshold:    float = 0.5
force_train:  bool  = False

# --- Shared L1 pre-pruning ---
# v2: reduced from 0.25 → 0.10 to limit pre-pruning accuracy cost (internal review pt.5)
l1_target: float = 0.10 #was 0.25
l1_steps:  int   = 5
ft_epochs: int   = 10

# --- Accuracy-budget settings ---
# v2: ACCURACY_TOLERANCE = 2 pp from BASELINE (internal review pt.1)
# v2: SNAPSHOT_INTERVAL  = 1 pp → thresholds [1pp, 2pp, 3pp] (internal review pt.3)
# v2: EDGE/BRIDGE step fractions halved for finer steps (internal review pt.4)
ACCURACY_TOLERANCE: float = 0.02 #was 0.03
SNAPSHOT_INTERVAL:  float = 0.01

MICRO_FT_EPOCHS:            int   = 5
NEURON_STEPS_PER_EDGE_STEP: int   = 3
NEURONS_PER_STEP:           int   = 3
EDGE_STEP_FRACTION:         float = 0.01   # was 0.02 (internal review pt.3)
BRIDGE_STEP_FRACTION:       float = 0.01   # was 0.02 (internal review pt.3)
MAX_STEPS:                  int   = 2000   # increased to allow more fine-grained steps

NEURON_FIRST_FRACTION: float = 0.15

# v3: noise-robust stop rule 
RECOVERY_FT_ROUNDS: int   = 2       # extra fine-tunes before calling a step a violation
MIN_STEP_SCALE:     float = 0.125   # smallest step multiplier after backtracking
MAX_FAILS_AT_MIN:   int   = 3       # failed retries at min scale before stopping

# v3: fine-tune with the baseline's augmentation (random crop + flip)
FT_AUGMENT: bool = False

# IterCC neuron filter settings
iterCC_low_mag_pct:   float = 75.0 # was 90.0
iterCC_degree_q:      float = 0.25
iterCC_min_in_degree: int   = 0

# NoC mapping parameters
P:         int = 196
N:         int = 196
mesh_rows: int = int(math.sqrt(P))
mesh_cols: int = int(math.sqrt(P))

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
BRIDGE_POST_NEURON_SMALL:   float = 0.005   # was 0.01 (finer for v2)
BRIDGE_POST_NEURON_SMALLER: float = 0.005
V2B_MAX_STEPS:              int   = 5000

# L1 label used in stages list — derived from l1_target so it stays consistent
_L1_LABEL = f"After L1 {l1_target:.0%} pre-pruning"

# ==================================================================
# Metrics tracker
# ==================================================================

stages:      list[dict] = []
trajectory:  list[dict] = []
_anchor_accuracy: float = 0.0

# ------------------------------------------------------------------
# Wall-clock timing (timing version)
#   branch_timings   : one record per branch (see run_accuracy_budget_branch)
#   _bridge_noc_s    : seconds spent in eval_noc (grouping + SA mapping) INSIDE
#                      the Bridge edge step, reset at the start of every branch
#   _last_record_s   : timings of the most recent _noc_and_record call
#   shared_pre_s     : baseline-independent pre-pruning (L1 stage) time, shared by
#                      all branches; filled in at the L1 stage below
# All durations use time.perf_counter(); CUDA is synchronised before reading it.
# ------------------------------------------------------------------
branch_timings: list[dict] = []
_bridge_noc_s:  list[float] = [0.0]
_last_record_s: dict = {}
shared_pre_s:   dict = {}


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _graph_ro(net):
    """Read-only access to the net's graph WITHOUT the networkx copy made by `net.graph`
    (PrunableNetwork.graph copies ~1.5M edges on every access).  Use only where the graph
    is just read, never modified."""
    g = getattr(net, "_graph", None)
    return g if g is not None else net.graph

# ==================================================================
# Fine-tune (v3: optional baseline-matching augmentation)
# ==================================================================

_AUG_MEAN = (0.4914, 0.4822, 0.4465)   # same constants as AugmentedCIFARDataset
_AUG_STD  = (0.2470, 0.2435, 0.2616)
_aug_cache: dict = {}


def _augment_batch(xb: torch.Tensor) -> torch.Tensor:
    """Random crop (pad=4) + horizontal flip on a flat normalised (B, 3072) batch."""
    dev  = xb.device
    B    = xb.size(0)
    mean = torch.tensor(_AUG_MEAN, device=dev).view(1, 3, 1, 1)
    std  = torch.tensor(_AUG_STD,  device=dev).view(1, 3, 1, 1)

    x = xb.view(B, 3, 32, 32) * std + mean              # back to [0, 1]
    x = F.pad(x, (4, 4, 4, 4))                          # zero (black) padding
    i = torch.randint(0, 9, (B,), device=dev)
    j = torch.randint(0, 9, (B,), device=dev)
    ar   = torch.arange(32, device=dev)
    rows = (i[:, None] + ar)[:, None, :, None]          # (B,1,32,1)
    cols = (j[:, None] + ar)[:, None, None, :]          # (B,1,1,32)
    bidx = torch.arange(B, device=dev)[:, None, None, None]
    cidx = torch.arange(3, device=dev)[None, :, None, None]
    x = x[bidx, cidx, rows, cols]                       # (B,3,32,32)
    flip = torch.rand(B, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    x = (x - mean) / std
    return x.reshape(B, 3072)


def fine_tune(model, X_train, y_train, *, device, lr, epochs, seed=None,
              batch_size: int = 128, weight_decay: float = 1e-4):
    """Drop-in for DNN_LeNet_500_300_CIFAR10.fine_tune; augments when FT_AUGMENT."""
    if not FT_AUGMENT:
        return _fine_tune_plain(model, X_train, y_train, device=device, lr=lr,
                                epochs=epochs, seed=seed)
    if seed is not None:
        set_seed(seed)

    key = id(X_train)
    if key not in _aug_cache:
        _aug_cache.clear()
        _aug_cache[key] = (
            torch.from_numpy(X_train.astype(np.float32)).to(device),
            torch.from_numpy(y_train.astype(np.int64)).to(device),
        )
    Xg, yg = _aug_cache[key]
    n = Xg.size(0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            Xb  = _augment_batch(Xg[idx])
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(Xb), yg[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    return model


# ------------------------------------------------------------------
# Vectorised cross-group edge helpers (same results as the per-edge Python loops:
# an edge is "cross-group" if both endpoint nodes have a group and the groups differ).
# ------------------------------------------------------------------
def _cross_group_masks(layers, node_to_group):
    """Per Linear layer: bool mask (out x in) of non-zero cross-group edges."""
    masks = []
    for li, layer in enumerate(layers):
        w = layer.weight.data
        n_out, n_in = w.shape
        src = torch.tensor([node_to_group.get((li, c), -1) for c in range(n_in)],
                           dtype=torch.long, device=w.device)
        dst = torch.tensor([node_to_group.get((li + 1, r), -1) for r in range(n_out)],
                           dtype=torch.long, device=w.device)
        valid = (dst[:, None] >= 0) & (src[None, :] >= 0)
        masks.append((w != 0) & valid & (dst[:, None] != src[None, :]))
    return masks


def _traj_record(branch: str, step: int, phase: str,
                 acc: float, drop: float, net: PrunableNetwork) -> None:
    try:
        from grouping_heuristic import indegree_grouping as _ig
        groups = _ig(_graph_ro(net), N, 10_000.0)
        node_to_grp: dict = {}
        for gid, members in enumerate(groups):
            for nd in members:
                node_to_grp[nd] = gid
        linears = [m for m in net.model.modules() if isinstance(m, nn.Linear)]
        masks   = _cross_group_masks(linears, node_to_grp)      # vectorised (was a per-edge loop)
        n_cross = int(sum(int(m.sum().item()) for m in masks))
        w_cross = float(sum(float(l.weight.data.abs()[m].double().sum().item())
                            for l, m in zip(linears, masks)))
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

    _t0 = time.perf_counter()
    noc = eval_noc(_graph_ro(net), P=P, N=N,
                   mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                   title=name, visualize=False)
    _t_noc = time.perf_counter() - _t0          # grouping + SA mapping

    _t1 = time.perf_counter()
    sp_wctg = build_wctg_structural(_graph_ro(net), noc.groups, net.model, eps=1e-8)
    sp_cost = compute_structural_cost(sp_wctg, noc.spg)
    _t_cost = time.perf_counter() - _t1         # structural cost evaluation
    _last_record_s.clear()
    _last_record_s.update({"grouping_sa_s": _t_noc, "cost_eval_s": _t_cost})

    stages.append({
        "name": name, "accuracy": acc,
        "f1": metrics["f1"], "precision": metrics["precision"],
        "recall": metrics["recall"],
        "cost": float(noc.cost), "sp_cost": sp_cost,
        "flops": compute_model_flops(net.model),
        "edges": _graph_ro(net).number_of_edges(),
        "neurons": _graph_ro(net).number_of_nodes(),
        "batch_ms": inf["batch_ms"], "state_size_kb": metrics["state_size_kb"],
    })
    print(f"  [recorded] '{name}'  acc={acc:.4f}  cost={noc.cost:.0f}  "
          f"sp_cost={sp_cost:.0f}  edges={_graph_ro(net).number_of_edges():,}  "
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


def repair_bn_sizes(model: nn.Module) -> None:
    all_mods = list(model.named_modules())
    for idx, (_, mod) in enumerate(all_mods):
        if not isinstance(mod, nn.Linear):
            continue
        expected = mod.out_features
        for _, next_mod in all_mods[idx + 1:]:
            if isinstance(next_mod, nn.BatchNorm1d):
                if next_mod.num_features != expected:
                    dev, dtype = next_mod.weight.device, next_mod.weight.dtype
                    new_bn = nn.BatchNorm1d(expected).to(device=dev, dtype=dtype)
                    keep = min(expected, next_mod.num_features)
                    with torch.no_grad():
                        new_bn.weight[:keep].copy_(next_mod.weight[:keep])
                        new_bn.bias[:keep].copy_(next_mod.bias[:keep])
                        new_bn.running_mean[:keep].copy_(next_mod.running_mean[:keep])
                        new_bn.running_var[:keep].copy_(next_mod.running_var[:keep])
                        new_bn.num_batches_tracked.copy_(next_mod.num_batches_tracked)
                    _replace_module(model, next_mod, new_bn)
                break
            if isinstance(next_mod, nn.Linear):
                break


def physically_remove_zeroed_neurons(model: nn.Module) -> None:
    while True:
        all_mods = list(model.named_modules())
        linears  = [m for m in model.modules() if isinstance(m, nn.Linear)]
        hidden   = linears[:-1]
        replaced_any = False

        for i, layer_out in enumerate(hidden):
            row_norms = layer_out.weight.data.abs().sum(dim=1)
            live_idx  = (row_norms > 0).nonzero(as_tuple=False).squeeze(1)
            if live_idx.numel() == layer_out.out_features:
                continue

            dev, dtype = layer_out.weight.device, layer_out.weight.dtype

            bn_to_trim = None
            found_linear = False
            for _, mod in all_mods:
                if mod is layer_out:
                    found_linear = True
                    continue
                if found_linear:
                    if isinstance(mod, nn.BatchNorm1d):
                        bn_to_trim = mod
                        break
                    if isinstance(mod, nn.Linear):
                        break

            new_out = nn.Linear(layer_out.in_features, live_idx.numel(),
                                bias=layer_out.bias is not None,
                                dtype=dtype, device=dev)
            with torch.no_grad():
                new_out.weight.copy_(layer_out.weight[live_idx])
                if layer_out.bias is not None:
                    new_out.bias.copy_(layer_out.bias[live_idx])
            _replace_module(model, layer_out, new_out)

            if bn_to_trim is not None:
                new_bn = nn.BatchNorm1d(live_idx.numel()).to(device=dev, dtype=dtype)
                with torch.no_grad():
                    new_bn.weight.copy_(bn_to_trim.weight[live_idx])
                    new_bn.bias.copy_(bn_to_trim.bias[live_idx])
                    new_bn.running_mean.copy_(bn_to_trim.running_mean[live_idx])
                    new_bn.running_var.copy_(bn_to_trim.running_var[live_idx])
                    new_bn.num_batches_tracked.copy_(bn_to_trim.num_batches_tracked)
                _replace_module(model, bn_to_trim, new_bn)

            linears_now   = [m for m in model.modules() if isinstance(m, nn.Linear)]
            layer_in_next = linears_now[i + 1]
            new_in = nn.Linear(live_idx.numel(), layer_in_next.out_features,
                               bias=layer_in_next.bias is not None,
                               dtype=dtype, device=dev)
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


def prune_neurons_l2_structured(model: nn.Module, amount: float) -> None:
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    hidden  = linears[:-1]
    for layer in hidden:
        scores  = layer.weight.data.pow(2).sum(dim=1).sqrt()
        n_prune = max(1, int(round(amount * scores.numel())))
        if n_prune >= scores.numel():
            continue
        threshold = torch.topk(scores, n_prune, largest=False).values.max()
        mask = (scores > threshold).float().to(layer.weight.device)
        layer.weight.data *= mask.unsqueeze(1)
        if layer.bias is not None:
            layer.bias.data *= mask


def prune_neurons_random(model: nn.Module, amount: float, seed: int = 0) -> None:
    rng     = torch.Generator()
    rng.manual_seed(seed)
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    hidden  = linears[:-1]
    total   = sum(m.out_features for m in hidden)
    n_prune = max(1, int(round(amount * total)))
    perm      = torch.randperm(total, generator=rng)
    prune_set = set(perm[:n_prune].tolist())
    offset = 0
    for layer in hidden:
        n_out = layer.out_features
        mask  = torch.ones(n_out, device=layer.weight.device)
        for local_i in range(n_out):
            if (offset + local_i) in prune_set:
                mask[local_i] = 0.0
        layer.weight.data *= mask.unsqueeze(1)
        if layer.bias is not None:
            layer.bias.data *= mask
        offset += n_out


def prune_neurons_activation(
    model: nn.Module,
    amount: float,
    X_calib,
    device,
    n_batches: int = 10,
    batch_sz: int  = 128,
) -> None:
    if isinstance(X_calib, np.ndarray):
        X_calib = torch.from_numpy(X_calib).float()
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    hidden  = linears[:-1]
    activation_sums   = {id(m): torch.zeros(m.out_features) for m in hidden}
    activation_counts = {id(m): 0 for m in hidden}
    hooks = []
    def make_hook(layer_id):
        def hook(module, inp, out):
            activation_sums[layer_id]   += out.detach().abs().mean(dim=0).cpu()
            activation_counts[layer_id] += 1
        return hook
    for layer in hidden:
        hooks.append(layer.register_forward_hook(make_hook(id(layer))))
    was_training = model.training
    model.eval()
    n = X_calib.shape[0]
    with torch.no_grad():
        for b in range(n_batches):
            idx_b = torch.randperm(n, generator=torch.Generator().manual_seed(b))[:batch_sz]
            model(X_calib[idx_b].to(device))
    for h in hooks:
        h.remove()
    if was_training:
        model.train()
    scores_per_layer = []
    for layer in hidden:
        cnt = max(1, activation_counts[id(layer)])
        scores_per_layer.append(activation_sums[id(layer)] / cnt)
    all_scores    = torch.cat(scores_per_layer)
    n_prune       = max(1, int(round(amount * all_scores.numel())))
    prune_indices = torch.topk(all_scores, n_prune, largest=False).indices
    prune_set     = set(prune_indices.tolist())
    offset = 0
    for layer, scores in zip(hidden, scores_per_layer):
        n_out = scores.numel()
        mask  = torch.ones(n_out, device=layer.weight.device)
        for local_i in range(n_out):
            if (offset + local_i) in prune_set:
                mask[local_i] = 0.0
        layer.weight.data *= mask.unsqueeze(1)
        if layer.bias is not None:
            layer.bias.data *= mask
        offset += n_out


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
                tied = (~zero_masks[mod]) & (layer_scores[mod] == threshold) \
                       & (mod.weight.data != 0.0)
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


def _l2_neuron_step(net: PrunableNetwork, *, max_neurons: int) -> tuple[int, list]:
    before   = _count_hidden_neurons(net.model)
    fraction = max_neurons / max(before, 1)
    prune_neurons_l2_structured(net.model, fraction)
    physically_remove_zeroed_neurons(net.model)
    net.commit_pruning()
    net.resync_graph()
    return before - _count_hidden_neurons(net.model), []


def _random_neuron_step(net: PrunableNetwork, *, max_neurons: int,
                        seed: int = 42) -> tuple[int, list]:
    before   = _count_hidden_neurons(net.model)
    fraction = max_neurons / max(before, 1)
    prune_neurons_random(net.model, fraction, seed=seed)
    physically_remove_zeroed_neurons(net.model)
    net.commit_pruning()
    net.resync_graph()
    return before - _count_hidden_neurons(net.model), []


def _activation_neuron_step(net: PrunableNetwork, *, max_neurons: int,
                             X_calib, device,
                             n_batches: int = 10,
                             batch_sz: int  = 128) -> tuple[int, list]:
    before   = _count_hidden_neurons(net.model)
    fraction = max_neurons / max(before, 1)
    prune_neurons_activation(net.model, fraction, X_calib=X_calib, device=device,
                             n_batches=n_batches, batch_sz=batch_sz)
    physically_remove_zeroed_neurons(net.model)
    net.commit_pruning()
    net.resync_graph()
    return before - _count_hidden_neurons(net.model), []


# ==================================================================
# Edge-step wrappers
# ==================================================================

def _l2_edge_step(net: PrunableNetwork, *, amount: float, **_) -> None:
    prune_l2_global(net.model, amount)


def _l1_edge_step(net: PrunableNetwork, *, amount: float, **_) -> None:
    _prune_l1_global_by_amount(net.model, amount)


# ------------------------------------------------------------------
# Groups used by the Bridge step.
#   default                 : eval_noc(...).groups  (grouping + SA mapping; original behaviour)
#   BRIDGE_GROUPS_FAST=1    : indegree_grouping(graph, N, 10_000.0) directly.  NoCMapper sets
#                             its groups with exactly this call (noc_mapper._group) and never
#                             changes them afterwards, so the groups are identical; only the
#                             unused SA mapping is skipped.  bridge_grouping_sa_s then holds
#                             grouping time only.
#   BRIDGE_GROUPS_CHECK=1   : (verification only, slow; implies FAST) also runs eval_noc and
#                             asserts both group lists are identical.  Do not use for timing.
# ------------------------------------------------------------------
BRIDGE_GROUPS_CHECK: bool = os.environ.get("BRIDGE_GROUPS_CHECK") == "1"
BRIDGE_GROUPS_FAST:  bool = BRIDGE_GROUPS_CHECK or os.environ.get("BRIDGE_GROUPS_FAST") == "1"


def _bridge_groups(net) -> list:
    graph = _graph_ro(net)
    if BRIDGE_GROUPS_FAST:
        from grouping_heuristic import indegree_grouping as _ig
        _t0 = time.perf_counter()
        groups = _ig(graph, N, 10_000.0)
        _bridge_noc_s[0] += time.perf_counter() - _t0          # grouping only
        if BRIDGE_GROUPS_CHECK:
            ref = eval_noc(graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                           title="_bridge_tmp", visualize=False).groups
            ok = (groups == ref)
            print(f"[bridge-groups-check] {len(groups)} groups -> {'IDENTICAL' if ok else 'MISMATCH'}")
            assert ok, "indegree_grouping groups differ from eval_noc groups"
        return groups
    _t0 = time.perf_counter()
    noc_tmp = eval_noc(graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                       title="_bridge_tmp", visualize=False)
    _bridge_noc_s[0] += time.perf_counter() - _t0              # grouping + SA mapping
    return noc_tmp.groups


# ------------------------------------------------------------------
# Bridge selection, vectorised.  BRIDGE_FAST=1 uses it; BRIDGE_CHECK=1 (verification
# only -- slow, do not use for timing runs) also runs the original per-edge loop and
# asserts that both pick exactly the same edges.  Default: original code path.
# ------------------------------------------------------------------
BRIDGE_CHECK: bool = os.environ.get("BRIDGE_CHECK") == "1"
BRIDGE_FAST:  bool = BRIDGE_CHECK or os.environ.get("BRIDGE_FAST") == "1"


def _bridge_select_fast(layers, node_to_group, k):
    """The k lowest-|w| cross-group edges (stable, row-major tie order = original list.sort).
    Returns [(layer_idx, row_idx_tensor, col_idx_tensor), ...] or None if no candidates."""
    vals, idxs = [], []
    for li, (layer, m) in enumerate(zip(layers, _cross_group_masks(layers, node_to_group))):
        rr, cc = m.nonzero(as_tuple=True)                     # row-major order
        vals.append(layer.weight.data[rr, cc].abs())
        idxs.append((rr, cc))
    sizes = [v.numel() for v in vals]
    if sum(sizes) == 0:
        return None
    allv    = torch.cat(vals)
    order   = torch.sort(allv, stable=True).indices[:k]
    layer_of = torch.repeat_interleave(
        torch.arange(len(sizes), device=allv.device),
        torch.tensor(sizes, device=allv.device))
    sel, off = [], 0
    for li, (rr, cc) in enumerate(idxs):
        pick = order[layer_of[order] == li] - off
        sel.append((li, rr[pick], cc[pick]))
        off += sizes[li]
    return sel


def _bridge_check_selection(layers, node_to_group, k, sel):
    ref = []
    for li, layer in enumerate(layers):
        w = layer.weight.data
        nz_r, nz_c = (w != 0).nonzero(as_tuple=True)
        for r, c in zip(nz_r.tolist(), nz_c.tolist()):
            sg = node_to_group.get((li,     c))
            dg = node_to_group.get((li + 1, r))
            if sg is None or dg is None or sg == dg:
                continue
            ref.append((float(w[r, c].abs()), li, r, c))
    ref.sort(key=lambda x: x[0])
    ref_set  = {(li, r, c) for _, li, r, c in ref[:k]}
    fast_set = set() if sel is None else {
        (li, r, c) for li, rr, cc in sel for r, c in zip(rr.tolist(), cc.tolist())}
    ok = (ref_set == fast_set) and ((len(ref) > 0) == (sel is not None))
    print(f"[bridge-check] candidates={len(ref):,}  k={k:,}  selected(ref/fast)="
          f"{len(ref_set):,}/{len(fast_set):,}  -> {'IDENTICAL' if ok else 'MISMATCH'}")
    assert ok, "Bridge fast selection differs from the original loop"


def _bridge_edge_step_slow(net: PrunableNetwork, *, amount: float, **_) -> None:
    groups = _bridge_groups(net)

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

    linears = [m for m in net.model.modules() if isinstance(m, nn.Linear)]
    bridge_candidates: list[tuple[float, int, int, int]] = []

    for li, layer in enumerate(linears):
        w = layer.weight.data
        nz_rows, nz_cols = (w != 0).nonzero(as_tuple=True)
        for r, c in zip(nz_rows.tolist(), nz_cols.tolist()):
            src_node = (li,     c)
            dst_node = (li + 1, r)
            src_grp  = node_to_group.get(src_node)
            dst_grp  = node_to_group.get(dst_node)
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

    physically_remove_zeroed_neurons(net.model)
    net.commit_pruning()
    net.resync_graph()


def _bridge_edge_step_fast(net: PrunableNetwork, *, amount: float, **_) -> None:
    groups = _bridge_groups(net)

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

    sel = _bridge_select_fast(layers, node_to_group, edges_to_prune)
    if BRIDGE_CHECK:
        _bridge_check_selection(layers, node_to_group, edges_to_prune, sel)
    if sel is None:
        prune_l2_global(net.model, amount)
        net.commit_pruning()
        net.resync_graph()
        return
    with torch.no_grad():
        for li, rr, cc in sel:
            layers[li].weight.data[rr, cc] = 0.0

    physically_remove_zeroed_neurons(net.model)
    net.commit_pruning()
    net.resync_graph()


_bridge_edge_step = _bridge_edge_step_fast if BRIDGE_FAST else _bridge_edge_step_slow


# ==================================================================
# Summary tables
# ==================================================================

def _fmt_delta(cur, bv, spec=".4f"):
    """Relative % delta — used for cost, edges, FLOPs, model size, inference time."""
    if bv == 0:
        return f"{cur:>10}"
    pct  = (cur - bv) / bv * 100
    sign = "+" if pct >= 0 else ""
    return f"{cur:>10{spec}}  ({sign}{pct:.2f}%)"

def _fmt_acc_delta(cur, bv):
    """Absolute pp delta — used for accuracy, F1, precision, recall (internal review pt.2)."""
    pp   = (cur - bv) * 100
    sign = "+" if pp >= 0 else ""
    return f"{cur:>10.4f}  ({sign}{pp:.2f} pp)"

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
        # accuracy-like columns: absolute pp  (internal review pt.2)
        acc_s  = f'{s["accuracy"]:>10.4f}{tag}'       if is_b else _fmt_acc_delta(s["accuracy"],  b["accuracy"])
        cost_s = f'{s["cost"]:>10.4f}{tag}'            if is_b else _fmt_delta(s["cost"],      b["cost"])
        spc_s  = f'{s["sp_cost"]:>10.4f}{tag}'         if is_b else _fmt_delta(s["sp_cost"],   b["sp_cost"])
        edge_s = f'{s["edges"]:>10,}{tag}'              if is_b else _fmt_int(s["edges"],       b["edges"])
        neur_s = f'{s["neurons"]:>10,}{tag}'            if is_b else _fmt_int(s["neurons"],     b["neurons"])
        bat_s  = f'{s["batch_ms"]:>7.2f} ms{tag}'      if is_b else _fmt_ms(s["batch_ms"],     b["batch_ms"])
        flop_s = f'{s["flops"]:>10,}{tag}'              if is_b else _fmt_int(s["flops"],       b["flops"])
        size_s = f'{s["state_size_kb"]:>5.2f} KB{tag}' if is_b else _fmt_kb(s["state_size_kb"],b["state_size_kb"])
        f1_s   = f'{s["f1"]:>10.4f}{tag}'              if is_b else _fmt_acc_delta(s["f1"],         b["f1"])
        prec_s = f'{s["precision"]:>10.4f}{tag}'       if is_b else _fmt_acc_delta(s["precision"],  b["precision"])
        rec_s  = f'{s["recall"]:>10.4f}{tag}'          if is_b else _fmt_acc_delta(s["recall"],     b["recall"])

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
    anchor_row = next((s for s in stages if s["name"] == _L1_LABEL), None)

    title_common = [
        "Model: LeNet 3072-500-300-10 | Dataset: CIFAR-10",
        f"Tolerance: ≤{ACCURACY_TOLERANCE*100:.1f} pp drop from BASELINE",
        "A=IterCC+Bridge  B=IterCC+L2  C=IterCC+L1  "
        "D=Taylor+L2  E=Taylor+L1  F=L1Neuron+L2  G=Taylor+Bridge  "
        "H=Random+L2  I=Activation+L2  K=L2Neuron+L2",
        "Q=GABiasedv2+Bridge  V=GAMemBiased+Bridge  W=GAMemBiased+L2  "
        "V2=GAMemBiased+Bridge(0.005)",
        "Deltas: accuracy/F1/precision/recall in pp; cost/edges/FLOPs/size relative to Baseline.",
    ]

    if snap_rows:
        _print_table(
            title_lines=["ISO-ACCURACY TABLE — snapshot at each accuracy-drop threshold",
                         f"Snapshot interval: every {SNAPSHOT_INTERVAL*100:.1f} pp"] + title_common,
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


def save_trajectories(
    path_csv:  str = "trajectory_LeNet500_300_CIFAR10_Meta_v2.csv",
    path_json: str = "trajectory_LeNet500_300_CIFAR10_Meta_v2.json",
) -> None:
    if not trajectory:
        print("[trajectory] Nothing to save.")
        return
    import csv, json
    fields = list(trajectory[0].keys())
    with open(path_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
        csv.DictWriter(f, fieldnames=fields).writerows(trajectory)
    print(f"[trajectory] Saved {len(trajectory)} rows → {path_csv}")
    with open(path_json, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"[trajectory] Saved {len(trajectory)} rows → {path_json}")


# ==================================================================
# Core accuracy-budget loop
# ==================================================================

def run_accuracy_budget_branch(
    label: str,
    l1_model_copy: nn.Module,
    anchor_accuracy: float,
    X_train, y_train, X_test, y_test,
    device, timings,
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
    max_steps: int = MAX_STEPS,
    recovery_ft_rounds: int   = RECOVERY_FT_ROUNDS,
    min_step_scale:     float = MIN_STEP_SCALE,
    max_fails_at_min:   int   = MAX_FAILS_AT_MIN,
) -> None:
    _sync()
    _t_branch = time.perf_counter()          # branch wall-clock starts here
    tm = defaultdict(float)                  # per-component seconds / counters
    _bridge_noc_s[0] = 0.0
    floor = anchor_accuracy - accuracy_tolerance
    _post_frac  = post_neuron_edge_fraction if post_neuron_edge_fraction is not None \
                  else edge_step_fraction
    _n_per_edge = max(0, neuron_steps_per_edge_step)
    _neuron_tol = neuron_tolerance if neuron_tolerance is not None else accuracy_tolerance

    if snapshot_interval and snapshot_interval > 0:
        n_snaps    = max(1, int(round(accuracy_tolerance / snapshot_interval)))
        thresholds = [round(snapshot_interval * (i + 1), 8) for i in range(n_snaps)]
    else:
        thresholds = []

    print("\n" + "=" * 70)
    print(f"  BRANCH: {label}")
    print(f"  Anchor={anchor_accuracy:.4f}  Tolerance={accuracy_tolerance:.4f} ({accuracy_tolerance*100:.1f} pp)  "
          f"Floor={floor:.4f}")
    if neuron_tolerance is not None:
        print(f"  Neuron phase cap: {_neuron_tol:.4f}")
    if neuron_first_steps > 0:
        print(f"  Neuron-first phase: {neuron_first_steps} steps")
    if thresholds:
        print(f"  Snapshot thresholds: {[f'{t*100:.1f} pp' for t in thresholds]}")
    print(f"  Edge frac (interleaved/post): {edge_step_fraction:.4f}/{_post_frac:.4f}  "
          f"N:E ratio={_n_per_edge}:1  FT epochs={micro_ft_epochs}  Max steps={max_steps}")
    print("=" * 70)

    net = PrunableNetwork(copy.deepcopy(l1_model_copy), eps=0.0)

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
    step_scale           = 1.0
    fails_at_min         = 0

    def _ft_eval(ft_seed: int) -> float:
        _sync(); _t0 = time.perf_counter()
        reapply_zero_masks(net)
        fine_tune(net.model, X_train, y_train, device=device, lr=fine_tune_lr,
                  epochs=micro_ft_epochs, seed=ft_seed)
        _sync(); _t1 = time.perf_counter()
        net.commit_pruning()
        net.resync_graph()
        _t2 = time.perf_counter()
        acc_ = _extract_accuracy(
            evaluate_model(net.model, X_test, y_test, device=device, timings=timings))
        _sync(); _t3 = time.perf_counter()
        tm["finetune_s"]   += _t1 - _t0
        tm["graph_sync_s"] += _t2 - _t1
        tm["eval_s"]       += _t3 - _t2
        tm["n_finetune_rounds"] += 1
        return acc_

    # ------------------------------------------------------------------
    # Phase 0 — Neuron-first
    # ------------------------------------------------------------------
    if neuron_first_steps > 0:
        print(f"[{label}] === Neuron-first phase: up to {neuron_first_steps} steps ===")
        for nf_step in range(1, neuron_first_steps + 1):
            pre_step_model = copy.deepcopy(net.model)   # saved BEFORE step
            kwargs = dict(neuron_step_kwargs)
            kwargs["max_neurons"] = neurons_per_step
            n_removed, _ = neuron_step_fn(net, **kwargs)

            if n_removed == 0:
                neuron_exhausted = True
                net._model = pre_step_model
                net.resync_graph()
                print(f"[{label}] Neuron-first exhausted at step {nf_step}.")
                break

            repair_bn_sizes(net.model)
            physically_remove_zeroed_neurons(net.model)
            reapply_zero_masks(net)
            fine_tune(net.model, X_train, y_train, device=device, lr=fine_tune_lr,
                      epochs=micro_ft_epochs, seed=seed + nf_step)
            net.commit_pruning()
            net.resync_graph()

            acc  = _extract_accuracy(
                evaluate_model(net.model, X_test, y_test, device=device, timings=timings))
            drop = anchor_accuracy - acc
            neuron_steps_done += 1
            total_steps       += 1
            print(f"[{label}] NF-step {nf_step:3d}: acc={acc:.4f}  drop={drop:+.4f} pp  "
                  f"neurons={_count_hidden_neurons(net.model)}  "
                  f"edges={_graph_ro(net).number_of_edges():,}")
            _traj_record(label, nf_step, "neuron-first", acc, drop, net)

            if drop <= accuracy_tolerance:
                best_model = copy.deepcopy(net.model)
                best_acc   = acc
                # v2 fix (internal review pt.3): record pre_step_model, not current state
                while (next_snap_idx < len(thresholds)
                       and drop >= thresholds[next_snap_idx]):
                    snap_pct   = thresholds[next_snap_idx] * 100
                    snap_label = f"{label} @ drop≤{snap_pct:.1f}%"
                    print(f"[{label}] → Snapshot at drop≤{snap_pct:.1f}%")
                    snap_net = PrunableNetwork(copy.deepcopy(pre_step_model), eps=0.0)
                    _noc_and_record(snap_label, snap_net, X_test, y_test, device, timings)
                    next_snap_idx += 1
            else:
                net._model = best_model
                net.resync_graph()
                print(f"[{label}] Neuron-first: tolerance exceeded, reverting.")
                break

        print(f"[{label}] === Neuron-first complete. Neurons: "
              f"{_count_hidden_neurons(net.model)} ===")

    # ------------------------------------------------------------------
    # Phase 1 — Interleaved loop
    # ------------------------------------------------------------------
    for step in range(1, max_steps + 1):
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
        active_edge_frac  = (edge_step_fraction if not neuron_phase_done
                             else _post_frac) * step_scale

        pre_step_model = copy.deepcopy(net.model)   # saved BEFORE step

        if do_neuron:
            kwargs = dict(neuron_step_kwargs)
            kwargs["max_neurons"] = neurons_per_step
            _sync(); _t0 = time.perf_counter()
            n_removed, _ = neuron_step_fn(net, **kwargs)
            _sync(); tm["neuron_step_s"] += time.perf_counter() - _t0
            if n_removed == 0:
                neuron_exhausted = True
                net._model = pre_step_model
                net.resync_graph()
                remaining = accuracy_tolerance - (anchor_accuracy - best_acc)
                print(f"[{label}] Step {step}: neuron exhausted. "
                      f"Remaining headroom: {remaining:.4f}. "
                      f"Switching to edge-only ({_post_frac:.4f}).")
                continue
            repair_bn_sizes(net.model)
            physically_remove_zeroed_neurons(net.model)
            neuron_steps_done += 1
            phase_label = "neuron"
        else:
            layers_now    = [m for m in net.model.modules() if isinstance(m, nn.Linear)]
            current_edges = sum(int((m.weight.data != 0).sum().item()) for m in layers_now)
            edges_to_prune = max(1, int(current_edges * active_edge_frac))
            target_edges   = current_edges - edges_to_prune
            if target_edges <= 0:
                edge_exhausted = True
                print(f"[{label}] Step {step}: edge exhausted (no edges left).")
                continue
            amount = _compute_edge_prune_amount(net, target_edges)
            if not (0.0 < amount < 1.0):
                edge_exhausted = True
                print(f"[{label}] Step {step}: edge exhausted (amount={amount:.4f}).")
                continue
            _sync(); _t0 = time.perf_counter()
            edge_step_fn(net, amount=amount, **edge_step_kwargs)
            net.commit_pruning()
            net.resync_graph()
            _sync(); tm["edge_step_s"] += time.perf_counter() - _t0
            phase_label = "edge" if not neuron_phase_done else "edge*"

        acc  = _ft_eval(seed + step)
        drop = anchor_accuracy - acc

        # v3: recovery fine-tune rounds before declaring a violation
        for r in range(1, recovery_ft_rounds + 1):
            if drop <= accuracy_tolerance:
                break
            print(f"[{label}] Step {step:3d} ({phase_label:6s}): drop={drop:+.4f} "
                  f"over budget -> recovery round {r}/{recovery_ft_rounds}")
            acc  = _ft_eval(seed + step + 1000 * r)
            drop = anchor_accuracy - acc

        # v3: still over budget -> revert this step, shrink step size, retry
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
              f"acc={acc:.4f}  drop={drop:+.4f} pp  "
              f"neurons={_count_hidden_neurons(net.model)}  "
              f"edges={_graph_ro(net).number_of_edges():,}  "
              f"scale={step_scale:.3f}")

        _t0 = time.perf_counter()
        _traj_record(label, total_steps, phase_label, acc, drop, net)
        tm["trajectory_log_s"] += time.perf_counter() - _t0   # measurement overhead
        total_steps += 1

        if True:   # within tolerance (violations handled above)
            best_model = copy.deepcopy(net.model)
            best_acc   = acc

            if (not neuron_tol_exhausted
                    and neuron_tolerance is not None
                    and drop >= _neuron_tol):
                neuron_tol_exhausted = True
                remaining = accuracy_tolerance - drop
                print(f"[{label}] Step {step}: neuron tolerance cap "
                      f"(drop={drop:.4f} ≥ {_neuron_tol:.4f}). "
                      f"Reserving {remaining:.4f} for edge-only ({_post_frac:.4f}).")

            # v2 fix (internal review pt.3): record pre_step_model, not current state
            while (next_snap_idx < len(thresholds)
                   and drop >= thresholds[next_snap_idx]):
                snap_pct   = thresholds[next_snap_idx] * 100
                snap_label = f"{label} @ drop≤{snap_pct:.1f}%"
                print(f"[{label}] → Snapshot at drop≤{snap_pct:.1f}%: '{snap_label}'")
                _t0 = time.perf_counter()
                snap_net = PrunableNetwork(copy.deepcopy(pre_step_model), eps=0.0)
                _noc_and_record(snap_label, snap_net, X_test, y_test, device, timings)
                tm["snapshot_s"] += time.perf_counter() - _t0   # measurement overhead
                next_snap_idx += 1

    _sync()
    _t_loop_end = time.perf_counter()
    final_label = f"{label} [final]"
    print(f"\n[{label}] Finished. {total_steps} micro-steps completed.")
    print(f"  Final: acc={best_acc:.4f}  "
          f"neurons={_count_hidden_neurons(net.model)}  "
          f"edges={_graph_ro(net).number_of_edges():,}")
    print(f"\n=== Final evaluation: {label} ===")
    _t0 = time.perf_counter()
    eval_nn(net.model, X_test=X_test, y_test=y_test, device=device,
            timings=timings, evaluate_fn=evaluate_model)
    _t_final_nn = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    _noc_and_record(final_label, net, X_test, y_test, device, timings)
    _t_final_rec = time.perf_counter() - _t0
    _sync()
    _t_end = time.perf_counter()

    # ---- timing record for this branch -------------------------------------
    total_wall = _t_end - _t_branch
    loop_wall  = _t_loop_end - _t_branch
    overhead   = tm["snapshot_s"] + tm["trajectory_log_s"]   # not part of the method
    rec = {
        "branch": label,
        "seed": seed,
        # (i) final grouping + SA mapping, on the final model
        "final_grouping_sa_s": _last_record_s.get("grouping_sa_s", float("nan")),
        "final_cost_eval_s":   _last_record_s.get("cost_eval_s", float("nan")),
        # (ii) whole pipeline for this branch
        "total_wall_s":        total_wall,                    # incl. logging overhead + final eval
        "core_wall_s":         total_wall - overhead,         # excl. snapshots + trajectory logging
        "loop_wall_s":         loop_wall,
        "final_eval_s":        _t_final_nn + _t_final_rec,
        # components of the loop
        "neuron_step_s":       tm["neuron_step_s"],
        "edge_step_s":         tm["edge_step_s"],
        "bridge_grouping_sa_s": _bridge_noc_s[0],             # part of edge_step_s (Bridge only)
        "finetune_s":          tm["finetune_s"],
        "eval_s":              tm["eval_s"],
        "graph_sync_s":        tm["graph_sync_s"],
        "snapshot_s":          tm["snapshot_s"],
        "trajectory_log_s":    tm["trajectory_log_s"],
        # counters
        "accepted_steps":      total_steps,
        "finetune_rounds":     int(tm["n_finetune_rounds"]),
    }
    branch_timings.append(rec)
    print(f"[{label}] TIMING: total={rec['total_wall_s']:.1f}s  core={rec['core_wall_s']:.1f}s  "
          f"final grouping+SA={rec['final_grouping_sa_s']:.1f}s  "
          f"neuron={rec['neuron_step_s']:.1f}s  edge={rec['edge_step_s']:.1f}s "
          f"(bridge grouping+SA {rec['bridge_grouping_sa_s']:.1f}s)  "
          f"finetune={rec['finetune_s']:.1f}s  eval={rec['eval_s']:.1f}s")


# ==================================================================
# 1. Train / load
# ==================================================================
### ADDED TO TEST FIXED SEED BETWEEN RUNS ###
### INTERNAL REVIEW OUTCOME: FIXED SEED FOR REPRODUCIBILITY ###                                            
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
### ADDED TO TEST FIXED SEED BETWEEN RUNS ###

model, scaler, (X_train, X_test, y_train, y_test), timings, device = load_or_train(
    data_dir, model_path,
    seed=seed, test_size=test_size, batch_size=batch_size,
    lr=lr, epochs=epochs, weight_decay=weight_decay,
    threshold=threshold, force_train=force_train,
    patience=30, min_delta=1e-4,
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
# 3. Iterative unstructured L1 edge pre-pruning  (5 × 2% = 10%)
# ==================================================================

_sync()
_t_pre0 = time.perf_counter()               # shared L1 pre-pruning timing starts
original_edge_count = _graph_ro(net).number_of_edges()
target_removals     = round(original_edge_count * l1_target)
total_weights       = sum(p.numel() for name, p in net.model.named_parameters()
                          if "weight" in name and p.dim() == 2)

for step in range(1, l1_steps + 1):
    print(f"L1 loop: {step}")
    already_removed   = original_edge_count - _graph_ro(net).number_of_edges()
    remaining         = target_removals - already_removed
    cumulative_target = already_removed + round(remaining / (l1_steps - step + 1))
    amount            = cumulative_target / total_weights

    if amount >= 1.0:
        print(f"  [Step {step}/{l1_steps}] Skipping.")
        break

    net.prune_links_l1_global(amount=amount)
    fine_tune(net.model, X_train, y_train, device=device, lr=fine_tune_lr,
              epochs=ft_epochs, seed=seed + step)

    removed = original_edge_count - _graph_ro(net).number_of_edges()
    print(f"  [Step {step}/{l1_steps}] Edges removed: {removed}/{original_edge_count} "
          f"({removed / original_edge_count:.1%})")

net.commit_pruning()
physically_remove_zeroed_neurons(net.model)
net.resync_graph()

print(f"\n=== After iterative L1 pruning ({l1_target:.0%} removed) ===")
eval_nn(net.model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
_noc_and_record(_L1_LABEL, net, X_test, y_test, device, timings)
_sync()
shared_pre_s.update({
    "l1_stage_total_s": time.perf_counter() - _t_pre0,       # prune + fine-tune + record
    "l1_grouping_sa_s": _last_record_s.get("grouping_sa_s", float("nan")),
})
print(f"[timing] Shared L1 pre-pruning stage: {shared_pre_s['l1_stage_total_s']:.1f}s "
      f"(grouping+SA of the L1 model: {shared_pre_s['l1_grouping_sa_s']:.1f}s)")

# ==================================================================
# 4. Anchor accuracy + checkpoint
#    v2: anchor = BASELINE (stages[0]), not post-L1 (internal review pt.1)
# ==================================================================

l1_model_copy   = copy.deepcopy(net.model)
l1_neuron_count = _count_hidden_neurons(net.model)
l1_edge_count   = _graph_ro(net).number_of_edges()
anchor_accuracy  = stages[0]["accuracy"]   # BASELINE anchor (v2)
_anchor_accuracy = anchor_accuracy

neuron_first_steps_bc = round(l1_neuron_count * NEURON_FIRST_FRACTION / NEURONS_PER_STEP)

post_l1_acc  = stages[-1]["accuracy"]
post_l1_drop = anchor_accuracy - post_l1_acc

print(f"\n[Checkpoint] L1 model: {l1_neuron_count} neurons, {l1_edge_count:,} edges.")
print(f"[Checkpoint] Tolerance anchor : BASELINE = {anchor_accuracy:.4f}")
print(f"[Checkpoint] Post-L1 accuracy : {post_l1_acc:.4f}  "
      f"(already {post_l1_drop*100:.2f} pp below baseline)")
print(f"[Checkpoint] Hard stop floor  : {anchor_accuracy - ACCURACY_TOLERANCE:.4f} "
      f"(baseline - {ACCURACY_TOLERANCE*100:.1f} pp)")
remaining_budget = ACCURACY_TOLERANCE - post_l1_drop
print(f"[Checkpoint] Remaining budget for pruning: {remaining_budget*100:.2f} pp")

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

# GA kwargs
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
# 5. Branch A — IterativeCC Neuron + Bridge-Link
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
    post_neuron_edge_fraction=0.02,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2, #was 2
    neuron_tolerance=0.005,
    neurons_per_step=3,
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
    post_neuron_edge_fraction=0.02,
    neuron_tolerance=0.005,
    neurons_per_step=3,
    micro_ft_epochs=2, #was 2
)

# ==================================================================
# 7. Branch C — IterativeCC Neuron + L1 edge  (commented out)
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
    post_neuron_edge_fraction=0.02,
    neuron_tolerance=0.005,
    neurons_per_step=3,
    micro_ft_epochs=2, #was 2
)
"""

# ==================================================================
# 8. Branch D — Taylor Neuron + L2 edge
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
    post_neuron_edge_fraction=0.02,
    micro_ft_epochs=2, #was 2
    neuron_tolerance=0.005,
    neurons_per_step=3,
)

# ==================================================================
# 9. Branch E — Taylor Neuron + L1 edge  (commented out)
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
    post_neuron_edge_fraction=0.02,
)
"""

# ==================================================================
# 10. Branch F — L1 Neuron + L2 edge  (commented out)
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
    post_neuron_edge_fraction=0.02,
    micro_ft_epochs=2, #was 2
    neuron_tolerance=0.005,
    neurons_per_step=3,
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
    post_neuron_edge_fraction=0.02,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2, #was 2
    neuron_tolerance=0.005,
    neurons_per_step=3,
)

# ==================================================================
# 12. Branch H — Random Neuron + L2 edge
# ==================================================================
"""
run_accuracy_budget_branch(
    label="H: Random Neuron + L2 edge",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=_random_neuron_step,
    neuron_step_kwargs={"seed": seed},
    edge_step_fn=_l2_edge_step,
    edge_step_kwargs={},
    neuron_steps_per_edge_step=1,
    post_neuron_edge_fraction=0.02,
    micro_ft_epochs=2, #was 2
    neuron_tolerance=0.005,
    neurons_per_step=3,
)
"""
# ==================================================================
# 13. Branch I — Activation Neuron + L2 edge
# ==================================================================
"""
run_accuracy_budget_branch(
    label="I: Activation Neuron + L2 edge",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=_activation_neuron_step,
    neuron_step_kwargs={"X_calib": X_train, "device": device,
                        "n_batches": 10, "batch_sz": 128},
    edge_step_fn=_l2_edge_step,
    edge_step_kwargs={},
    neuron_steps_per_edge_step=1,
    post_neuron_edge_fraction=0.02,
    micro_ft_epochs=2, #was 2
    neuron_tolerance=0.005,
    neurons_per_step=3,
)
"""
# ==================================================================
# 14. Branch K — L2 Neuron + L2 edge
# ==================================================================
"""
run_accuracy_budget_branch(
    label="K: L2 Neuron + L2 edge",
    l1_model_copy=l1_model_copy,
    anchor_accuracy=anchor_accuracy,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
    neuron_step_fn=_l2_neuron_step,
    neuron_step_kwargs={},
    edge_step_fn=_l2_edge_step,
    edge_step_kwargs={},
    neuron_steps_per_edge_step=1,
    post_neuron_edge_fraction=0.02,
    micro_ft_epochs=2, #was 2
    neuron_tolerance=0.005,
    neurons_per_step=3,
)
"""
# ==================================================================
# 15. Branch Q — GA CC_Biased v2 + Bridge-Link
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
    post_neuron_edge_fraction=0.02,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2, #was 2
    neuron_tolerance=0.005,
    neurons_per_step=3,
)
"""
# ==================================================================
# 16. Branch V — Memetic + Biased GA + Bridge-Link
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
    post_neuron_edge_fraction=0.02,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2, #was 2
    neuron_tolerance=0.005,
    neurons_per_step=3,
)
"""
# ==================================================================
# 17. Branch W — Memetic + Biased GA + L2 Edge
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
    post_neuron_edge_fraction=0.02,
    neuron_steps_per_edge_step=1,
    micro_ft_epochs=2, #was 2
    neuron_tolerance=0.005,
    neurons_per_step=3,
)
"""
# ==================================================================
# 18. Branch V2 — Memetic + Biased GA + Bridge-Link (small step 0.005)
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
    micro_ft_epochs=2, #was 2
    neuron_tolerance=0.005,
    neurons_per_step=3,
)
"""
# ==================================================================
# 19. Summary
# ==================================================================

print_summary_tables()


def print_and_save_timing() -> None:
    """Per-branch wall-clock table (seconds) + JSON/CSV next to the other outputs."""
    import csv, json
    if not branch_timings:
        print("[timing] Nothing recorded.")
        return
    cols = [("branch", "Branch"),
            ("total_wall_s", "Total"), ("core_wall_s", "Core"),
            ("final_grouping_sa_s", "Final grp+SA"),
            ("neuron_step_s", "Neuron"), ("edge_step_s", "Edge"),
            ("bridge_grouping_sa_s", "(Bridge grp+SA)"),
            ("finetune_s", "Fine-tune"), ("eval_s", "Eval"),
            ("snapshot_s", "Snapshots"), ("trajectory_log_s", "Traj log"),
            ("accepted_steps", "Steps"), ("finetune_rounds", "FT rounds")]
    print("\n" + "=" * 150)
    print(f"  WALL-CLOCK TIMING per branch (seconds, seed {seed}).  "
          "Total = whole branch incl. snapshots/logging/final eval;  "
          "Core = Total minus snapshots and trajectory logging.")
    print(f"  Shared L1 pre-pruning (not in any branch): "
          f"{shared_pre_s.get('l1_stage_total_s', float('nan')):.1f}s")
    print("=" * 150)
    print("  ".join(f"{h:>15}" if i else f"{h:<40}" for i, (_, h) in enumerate(cols)))
    for r in branch_timings:
        cells = []
        for i, (k, _) in enumerate(cols):
            v = r[k]
            if i == 0:
                cells.append(f"{v:<40}")
            elif isinstance(v, int):
                cells.append(f"{v:>15d}")
            else:
                cells.append(f"{v:>15.1f}")
        print("  ".join(cells))
    stem = f"timing_LeNet500_300_CIFAR10_v3_seed{seed}"
    with open(stem + ".json", "w") as f:
        json.dump({"seed": seed, "shared_pre": shared_pre_s, "branches": branch_timings},
                  f, indent=2, default=float)
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(branch_timings[0].keys()))
        w.writeheader()
        w.writerows(branch_timings)
    print(f"[timing] Saved {stem}.json / .csv")


print_and_save_timing()

save_trajectories(
    path_csv="trajectory_LeNet500_300_CIFAR10_Meta_v3.csv",
    path_json="trajectory_LeNet500_300_CIFAR10_Meta_v3.json",
)
