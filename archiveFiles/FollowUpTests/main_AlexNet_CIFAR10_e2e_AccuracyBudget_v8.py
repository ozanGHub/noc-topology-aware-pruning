"""
main_AlexNet_CIFAR10_e2e_AccuracyBudget_Metaheuristic_v8.py
============================================================
v8 (built from v7; every new switch defaults to the v7 behaviour, Bridge is unchanged):
  14. NEURON_TOL=<float>        override the neuron-phase cap passed by every branch call
                                (hard-coded 0.005 = 0.5 pp in v7).
  15. NEURON_CAP_FROM_L1=1      measure the neuron cap from the post-L1 accuracy instead of from the
                                baseline: the neuron phase ends when (drop - L1_drop) >= cap, with
                                L1_drop = max(0, baseline - post-L1 accuracy).  In v7 the L1 drop
                                counts against the cap, so the neuron phase ends after 1-2 steps
                                (3-6 neurons removed).  The total budget (2 pp from baseline) is unchanged.
  16. NEURONS_PER_STEP=<int>    override neurons_per_step (3 in v7).
  17. RUN_TAG=<text>            suffix for the output files (trajectory_/timing_ ..._v8<tag>_seed<seed>),
                                so variant runs do not overwrite each other.
  18. Observation (not a bug, but different from the end-to-end design): in v7 (and the v3 flat
                                scripts) the neuron/edge choice uses neuron_steps_done % (N+1) != N,
                                and neuron_steps_done only increases on neuron steps.  After the first
                                N neuron steps the run is therefore edge-only for good (AlexNet 1:1 ->
                                exactly ONE neuron step, 3 neurons), so the neuron phase never uses
                                its share of the accuracy budget.
      INTERLEAVE_FIX=1          real N:1 alternation (cycle counter advances on every accepted step);
                                the neuron cap then ends the neuron phase.
  19. NEURON_SEQUENTIAL=1       original end-to-end design: ONLY neuron steps until the neuron cap is
                                reached (NEURON_TOL, optionally NEURON_CAP_FROM_L1=1) or the criterion
                                is exhausted, then ONLY edge steps with the remaining budget.  The N:1
                                ratio is ignored.  Takes precedence over INTERLEAVE_FIX.
  20. NEURON_MAX_FRAC=<f>       size-independent neuron phase: end the neuron phase when f * (removable
                                FC neurons at the start of the branch) have been removed (counted from the
                                actual neuron count, lonely-neuron cleanup included).  The accuracy cap
                                NEURON_TOL and the budget revert still apply; whichever comes first ends it.
                                Use a loose NEURON_TOL (e.g. 0.02) so the count decides.
  21. NEURON_STEP_FRAC=<f>      neurons removed per neuron step = max(1, round(f * base)); overrides
                                NEURONS_PER_STEP so the number of neuron steps stays similar across
                                model sizes.  The last step is trimmed to hit the NEURON_MAX_FRAC target.
  23. NEURON_SHARE=<f>          neuron-phase cap as a FRACTION of the total budget: cap = f * ACCURACY_TOLERANCE
                                (e.g. 0.8 with ACCURACY_TOLERANCE=0.01 -> 0.8 pp; with 0.02 -> 1.6 pp).
                                Takes precedence over NEURON_TOL; combine with NEURON_CAP_FROM_L1=1 to
                                measure it from the post-L1 accuracy instead of from the baseline.
  24. NEURON_SHARE_BASIS=remaining   (with NEURON_SHARE; default "total" = unchanged behaviour)
                                the share applies to the budget left AFTER the L1 stage, while the cap
                                stays a cumulative drop from the BASELINE:
                                    cap = L1_drop + share * (ACCURACY_TOLERANCE - L1_drop).
                                A large L1 drop (e.g. AlexNet seed 10042, -1.49 pp) then does not eat the
                                neuron budget, and a small one changes little.  Overrides
                                NEURON_CAP_FROM_L1.
  22. ACCURACY_TOLERANCE=<f>    total accuracy budget from the BASELINE (default 0.02 = 2 pp).  E.g. 0.01 with
                                NEURON_TOL=0.005 reproduces the original end-to-end design (1 pp total,
                                neurons first up to 0.5 pp, edges with the rest; L1 is part of the budget
                                unless NEURON_CAP_FROM_L1=1).  Snapshot thresholds and stop rules follow it.
  25. BRANCHES=A,D              run only the listed branches (letters A,B,D,G); default all.  With L2_LONELY=1,
                                B==A and G==D, so A,D loses nothing and halves the runtime.
  26. BASELINE_REANNEAL=1        (default off) fine-tune the dense model once with a cosine LR decay (BASELINE_REANNEAL_LR, default 2e-4,
                                to 0 over BASELINE_REANNEAL_EPOCHS, default 10) and use it as the baseline; saved to
                                DNN_AlexNet_CIFAR10_annealed.pth and reused by all seeds.  Original .pth untouched.
  27. SNAPSHOT_NEURON_END=1      (default off) record the full metric row "<branch> [neuron end]" (WCTG, cross-group, edges, FLOPs...)
                                at the moment the neuron phase ends, and show it in the FINAL TABLE.  Isolates neuron selection
                                (same neuron count, before the edge phase) from the edge phase, which saturates near -97% on AlexNet.
  28. DEFAULTS (v8): the settings used in the AlexNet experiments are now ON by default, so no env variables are needed:
                                MODEL_PATH=DNN_AlexNet_CIFAR10_full200.pth  FT_LR=1e-4  L1_TARGET=0.10  L2_LONELY=1  NEURON_SEQUENTIAL=1
                                SNAPSHOT_INTERVAL=0.005  SEED=42  EVAL_FAST=1  GRAPH_FAST=1  EDGE_FAST=1  BRIDGE_GROUPS_FAST=1
                                matched-count neuron phase: NEURON_MAX_FRAC=0.30  NEURON_STEP_FRAC=0.015  NEURON_TOL=0.05 (safety cap)
                                SNAPSHOT_NEURON_END=1.  ACCURACY_TOLERANCE stays 0.02 (2 pp).  BRANCHES stays "all".
                                To run the share-based variant: NEURON_MAX_FRAC= NEURON_STEP_FRAC=0.015 NEURON_SHARE=0.8 NEURON_SHARE_BASIS=remaining
                                (empty NEURON_MAX_FRAC disables the count target).  Any default is overridden by setting the variable.
  Output files are named v8.  The neuron-phase end is now logged with the neuron count removed.
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

v5 change: fine_tune_lr decoupled from the training LR (env FT_LR), env L1_TARGET, STOP_AFTER_L1.

v7 change (stop rule only; everything else identical to v6):
  13. Diminishing-returns stop, env MIN_GAIN_STOP (default 0 = OFF = exactly v6).
      With MIN_GAIN_STOP=0.03 and MIN_GAIN_WINDOW=10 an edge-only phase stops when the last
      10 accepted steps removed < 3% of the graph edges (the tail where the step scale is
      0.125 and each step removes ~0.25% edges but still costs fine-tune + recovery rounds).
      The last accepted model is kept, so the result is a valid within-budget point; it
      stops EARLIER than v6, so v6 and v7 are two different stopping rules to compare.
      Output files are named v7, so v6 outputs are not overwritten.

v6 changes (speed + timing; pruning methods and results unchanged):
  8. Per-branch wall-clock timing (same fields as the LeNet 500-300 v3_timing file) ->
     timing_AlexNet_CIFAR10_v6_seed<seed>.json/.csv.  graph_sync_s (the library's
     graph rebuild after every fine-tune) and edge_prep_s are reported separately.
  9. Vectorised edge helpers (the per-pair Python loops with a GPU .item() per
     channel pair were the main cost of the edge steps):
       EDGE_FAST=1   -> vectorised L2 edge step, Bridge candidate selection and zero-pair counts
       EDGE_CHECK=1  -> (verification only, slow; implies EDGE_FAST) additionally runs the
                        original loops and asserts the same edges / counts are selected.
     Default (no env var) = original code path.  Edge selection uses the same candidate
     order as the original (L2: layer, out, in; Bridge: graph-edge order = layer, in, out).
     Weights are summed on the GPU in a different reduction order than the per-pair
     loop, so values can differ by ~1 ulp: edges tied to within that are the only
     possible difference (the check allows rtol=1e-4 on the selected weights).
 10. Always-on, results-neutral: read-only graph access without the networkx copy that
     `net.graph` makes (_graph_ro), and Taylor importance accumulation via .tolist().
 11. Env SEED; seed-suffixed output names.
 12. Bridge edge step FIX: edges_to_prune is now counted in channel pairs (the unit of
     `amount`).  v5 and earlier mixed pair-level `amount` with element counts, so a single
     Bridge step could prune 10-100x the nominal step and collapse the network to 0 edges
     (seen in branches A and G of earlier logs).  Branch A/G results from v5 and earlier
     are not comparable with v6.

Active branches: A (IterCC+Bridge), B (IterCC+L2), D (Taylor+L2), G (Taylor+Bridge)
GA branches Q/V/W/V2 commented out.
"""

from __future__ import annotations

import copy
import os
import math
import time
from collections import defaultdict
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# v8 defaults: GRAPH_FAST is read inside graph.py / graph_conv.py at import time (shared with other scripts), so it is
# switched on here, BEFORE they are imported, without changing those modules.  GRAPH_FAST=0 turns it off again.
os.environ.setdefault("GRAPH_FAST", "1")

from DNN_AlexNet_CIFAR10 import load_or_train, fine_tune, evaluate_model, NumpyDataset, predict
from prunable_network_conv import PrunableNetworkConv
from eval_helper import eval_noc, eval_nn
from mapping_helpers import build_wctg_structural, compute_structural_cost
from pruning import select_neurons_high_indegree_low_magnitude, _commcost_score_neuron
from grouping_heuristic import indegree_grouping

# ==================================================================
# Configuration
# ==================================================================

data_dir:     str   = "./data"
model_path:   str   = os.environ.get("MODEL_PATH", "DNN_AlexNet_CIFAR10_full200.pth")   # v8 default: the fully annealed baseline (env override)
seed:         int   = int(os.environ.get("SEED", "42"))   # env SEED overrides
batch_size:   int   = 128
lr:           float = 1e-3
fine_tune_lr: float = float(os.environ.get("FT_LR", "1e-4"))   # v5: decoupled from training LR (1e-3); env FT_LR overrides for LR sweeps
epochs:       int   = 200
weight_decay: float = 1e-4
force_train:  bool  = os.environ.get("FORCE_TRAIN") == "1"   # v8: env switch (default off)

# --- Shared L1 pre-pruning ---
# v2: reduced from 0.25 → 0.10 to limit pre-pruning accuracy cost (internal review pt.5)
l1_target: float = float(os.environ.get("L1_TARGET", "0.10"))   # env L1_TARGET overrides
l1_steps:  int   = 5
ft_epochs: int   = 10

# v2: ACCURACY_TOLERANCE = 2 pp from BASELINE (internal review pt.1)
# v2: SNAPSHOT_INTERVAL  = 1 pp → thresholds [1pp, 2pp, 3pp] (internal review pt.3)
# v2: EDGE/BRIDGE step fractions halved for finer steps (internal review pt.4)
ACCURACY_TOLERANCE:         float = float(os.environ.get("ACCURACY_TOLERANCE", "0.02"))   # 2 pp from BASELINE; v8: env override (e.g. 0.01)
SNAPSHOT_INTERVAL:          float = float(os.environ.get("SNAPSHOT_INTERVAL", "0.005"))   # v8 default: every 0.5 pp (env override)
MICRO_FT_EPOCHS:            int   = 2
NEURON_STEPS_PER_EDGE_STEP: int   = 1
NEURONS_PER_STEP:           int   = 3
EDGE_STEP_FRACTION:         float = 0.01   # was 0.02
BRIDGE_STEP_FRACTION:       float = 0.01   # was 0.02
MAX_STEPS:                  int   = 2000
RECOVERY_FT_ROUNDS:         int   = 2       # v4: extra fine-tunes before calling a step a violation
MIN_STEP_SCALE:             float = 0.125   # v4: smallest step multiplier after backtracking
MAX_FAILS_AT_MIN:           int   = 3       # v4: failed retries at min scale before stopping
MIN_GAIN_STOP:              float = float(os.environ.get("MIN_GAIN_STOP", "0"))    # v7: 0 = off
MIN_GAIN_WINDOW:            int   = int(os.environ.get("MIN_GAIN_WINDOW", "10"))  # v7
# v8 switches (all default to the v7 behaviour)
NEURON_TOL_ENV:      float | None = (float(os.environ.get("NEURON_TOL", "0.05"))   # v8 default: loose 5 pp safety cap, the neuron COUNT decides; NEURON_TOL= disables
                                     if os.environ.get("NEURON_TOL", "0.05") else None)
NEURON_SHARE_ENV:    float | None = (float(os.environ["NEURON_SHARE"])
                                     if os.environ.get("NEURON_SHARE") else None)
NEURON_SHARE_BASIS:  str          = os.environ.get("NEURON_SHARE_BASIS", "total")   # "total" (default) | "remaining"
NEURON_CAP_FROM_L1:  bool         = os.environ.get("NEURON_CAP_FROM_L1") == "1"
NEURONS_PER_STEP_ENV: int | None  = (int(os.environ["NEURONS_PER_STEP"])
                                     if os.environ.get("NEURONS_PER_STEP") else None)
RUN_TAG:             str          = os.environ.get("RUN_TAG", "")
INTERLEAVE_FIX:      bool         = os.environ.get("INTERLEAVE_FIX") == "1"
NEURON_SEQUENTIAL:   bool         = os.environ.get("NEURON_SEQUENTIAL", "1") == "1"
NEURON_MAX_FRAC:     float | None = (float(os.environ.get("NEURON_MAX_FRAC", "0.30"))   # v8 default: remove 30% of the FC neurons; NEURON_MAX_FRAC= (empty) disables
                                     if os.environ.get("NEURON_MAX_FRAC", "0.30") else None)
BRANCHES:            str          = os.environ.get("BRANCHES", "").replace(",", "").upper()   # e.g. "AD"; empty = all
SNAPSHOT_NEURON_END: bool         = os.environ.get("SNAPSHOT_NEURON_END", "1") == "1"   # v8 default ON: record the [neuron end] row (SNAPSHOT_NEURON_END=0 disables)
NEURON_STEP_FRAC:    float | None = (float(os.environ.get("NEURON_STEP_FRAC", "0.015"))   # v8 default: 1.5% of the FC neurons per step; NEURON_STEP_FRAC= disables
                                     if os.environ.get("NEURON_STEP_FRAC", "0.015") else None)

# NoC — 11×11 mesh (2688 hidden nodes, ~5.6/core)
P:         int = 121
N:         int = 121
mesh_rows: int = int(math.sqrt(P))
mesh_cols: int = int(math.sqrt(P))

iterCC_low_mag_pct:   float = 75.0 #90.0?
iterCC_degree_q:      float = 0.25
iterCC_min_in_degree: int   = 0

_L1_LABEL = f"After L1 {l1_target:.0%} pre-pruning"

# ==================================================================
# Metrics
# ==================================================================

stages:     list[dict] = []
trajectory: list[dict] = []

# ------------------------------------------------------------------
# Wall-clock timing (v6) -- same definitions as the LeNet 500-300 v3_timing file
#   branch_timings : one record per branch (see run_accuracy_budget_branch)
#   _bridge_noc_s  : seconds in eval_noc (grouping + SA mapping) INSIDE the Bridge edge
#                    step, reset at the start of every branch
#   _last_record_s : timings of the most recent _noc_and_record call
#   shared_pre_s   : L1 pre-pruning stage time (shared by all branches)
# ------------------------------------------------------------------
branch_timings: list[dict] = []
_bridge_noc_s:  list[float] = [0.0]
_last_record_s: dict = {}
shared_pre_s:   dict = {}


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _traj_record(branch, step, phase, acc, drop, net):
    linears = [m for m in net.model.modules() if isinstance(m, nn.Linear)]
    n_edges = sum(int((m.weight.data != 0).sum().item()) for m in linears)
    trajectory.append({
        "branch": branch, "step": step, "phase": phase,
        "accuracy": acc, "accuracy_drop": drop,
        "fc_neurons": _count_fc_neurons(net.model),
        "fc_edges": n_edges,
        "graph_edges": _graph_ro(net).number_of_edges(),
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

    _t0 = time.perf_counter()
    noc = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                   title=name, visualize=False)
    _t_noc = time.perf_counter() - _t0          # grouping + SA mapping

    _t1 = time.perf_counter()
    _Gw = net.graph
    try:
        sp_wctg = build_wctg_structural(_Gw, noc.groups, net.model, eps=1e-8)
    except Exception:
        # v8 diagnostic: a node that passes has_node() but is missing from _succ means an inconsistent graph copy
        import networkx as _nx
        _bad = [n for n in _Gw._node if n not in _Gw._succ][:5]
        _miss = [u for g in noc.groups for u in g if not _Gw.has_node(u)][:5]
        print(f"[DIAG] networkx={_nx.__version__} nodes={len(_Gw._node)} succ={len(_Gw._succ)} pred={len(_Gw._pred)} "
              f"in_node_not_in_succ={_bad} group_members_missing={_miss} "
              f"member_types={[(type(a).__name__, type(b).__name__) for a, b in (_miss or _bad)[:2]]}")
        raise
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
BRIDGE_GROUPS_FAST:  bool = BRIDGE_GROUPS_CHECK or os.environ.get("BRIDGE_GROUPS_FAST", "1") == "1"


def _bridge_groups(net) -> list:
    graph = net.graph
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


def _conv_bridge_edge_step_slow(net: PrunableNetworkConv, *, amount: float, **_) -> None:
    node_to_group = {node: gid for gid, members in enumerate(_bridge_groups(net))
                     for node in members}

    G             = net.graph
    spatial_sizes = G.graph["spatial_sizes"]
    ordered       = _get_ordered_layers(net.model)

    # v6 fix: `amount` is a fraction of channel PAIRS (see _compute_edge_prune_amount), so the
    # number of pairs to prune must be counted in pairs too.  v5 mixed it with element counts
    # (m.weight.numel()), which let one Bridge step prune hundreds of thousands of pairs.
    total_pairs, zero_pairs = _zero_pairs(net)
    active_edges   = total_pairs - zero_pairs
    edges_to_prune = max(1, int(round(amount * total_pairs)) - zero_pairs)
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


# L2_LONELY=1 (ablation, default OFF): the L2 edge step also calls net.remove_lonely_neurons()
# before commit_pruning(), exactly as the Bridge edge step does.  Bridge itself is unchanged.
L2_LONELY: bool = os.environ.get("L2_LONELY", "1") == "1"


def _conv_l2_edge_step_slow(net: PrunableNetworkConv, *, amount: float, **_) -> None:
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

    if L2_LONELY:
        net.remove_lonely_neurons()
    net.commit_pruning()
    net.resync_graph()


# ------------------------------------------------------------------
# Vectorised edge helpers (v6).  EDGE_FAST=1 uses them; EDGE_CHECK=1 (verification only,
# slow -- never for timing runs) also runs the original per-pair loops and asserts that
# the same edges / counts result.  Default: original code path.
# ------------------------------------------------------------------
EDGE_CHECK: bool = os.environ.get("EDGE_CHECK") == "1"
EDGE_FAST:  bool = EDGE_CHECK or os.environ.get("EDGE_FAST", "1") == "1"


def _graph_ro(net):
    """Read-only access to the net's graph WITHOUT the networkx copy made by `net.graph`."""
    g = getattr(net, "_graph", None)
    return g if g is not None else net.graph


def _pair_weight_mats(net) -> list:
    """Per Conv/Linear layer an (out, in) matrix with the channel-pair weight used by the
    edge steps (sum of |w| over the kernel / spatial block) = vectorised _conv_edge_weight."""
    G     = _graph_ro(net)
    sizes = G.graph["layers"]
    sps   = G.graph["spatial_sizes"]
    mats  = []
    for li, (_, m) in enumerate(_get_ordered_layers(net.model)):
        w = m.weight.data.abs()
        if isinstance(m, nn.Conv2d):
            mats.append(w.sum(dim=(2, 3)))
        else:
            mats.append(w.view(w.shape[0], sizes[li], sps[li]).sum(dim=-1))
    return mats


def _zero_pairs_slow(net) -> tuple[int, int]:
    G             = _graph_ro(net)
    spatial_sizes = G.graph["spatial_sizes"]
    total = zero = 0
    for li, (_, m) in enumerate(_get_ordered_layers(net.model)):
        sp       = spatial_sizes[li]
        src_size = G.graph["layers"][li]
        out_size = m.out_channels if isinstance(m, nn.Conv2d) else m.out_features
        total   += src_size * out_size
        for j in range(out_size):
            for i in range(src_size):
                if _conv_edge_weight(m, j, i, sp) == 0.0:
                    zero += 1
    return total, zero


def _zero_pairs(net) -> tuple[int, int]:
    """(total channel pairs, pairs whose block is entirely zero)."""
    if not EDGE_FAST:
        return _zero_pairs_slow(net)
    mats  = _pair_weight_mats(net)
    fast  = (sum(M.numel() for M in mats), sum(int((M == 0).sum().item()) for M in mats))
    if EDGE_CHECK:
        slow = _zero_pairs_slow(net)
        print(f"[edge-check] zero pairs fast/slow = {fast}/{slow} -> "
              f"{'IDENTICAL' if fast == slow else 'MISMATCH'}")
        assert fast == slow, "zero-pair counts differ"
    return fast


def _lowest_k_pairs(mats, keep_masks, k: int, col_major: bool):
    """k lowest-weight pairs among keep_masks (stable; candidate order = row-major over
    (out,in), or over (in,out) when col_major).  Returns [(layer, out_idx, in_idx, w)] or None."""
    vals, idxs, sizes = [], [], []
    for M, mask in zip(mats, keep_masks):
        if col_major:
            cc, rr = mask.t().nonzero(as_tuple=True)
        else:
            rr, cc = mask.nonzero(as_tuple=True)
        vals.append(M[rr, cc])
        idxs.append((rr, cc))
        sizes.append(rr.numel())
    if sum(sizes) == 0:
        return None
    allv     = torch.cat(vals)
    order    = torch.sort(allv, stable=True).indices[:k]
    layer_of = torch.repeat_interleave(torch.arange(len(sizes), device=allv.device),
                                       torch.tensor(sizes, device=allv.device))
    sel, off = [], 0
    for li, (rr, cc) in enumerate(idxs):
        pick = order[layer_of[order] == li] - off
        sel.append((li, rr[pick], cc[pick], vals[li][pick]))
        off += sizes[li]
    return sel


def _zero_pairs_apply(net, sel) -> None:
    ordered = _get_ordered_layers(net.model)
    G       = _graph_ro(net)
    sizes, sps = G.graph["layers"], G.graph["spatial_sizes"]
    with torch.no_grad():
        for li, rr, cc, _ in sel:
            if rr.numel() == 0:
                continue
            m = ordered[li][1]
            if isinstance(m, nn.Conv2d):
                m.weight.data[rr, cc] = 0.0
            else:
                w = m.weight.data
                w.view(w.shape[0], sizes[li], sps[li])[rr, cc] = 0.0


def _edge_check(name: str, ref_sorted, k: int, sel) -> None:
    """ref_sorted: [(w, layer, out, in)] stable-sorted reference candidates."""
    import numpy as _np
    ref_k    = ref_sorted[:k]
    ref_set  = {(li, j, i) for _, li, j, i in ref_k}
    if sel is None:
        ok = len(ref_sorted) == 0
        print(f"[edge-check] {name}: no candidates (ref has {len(ref_sorted)}) -> {'IDENTICAL' if ok else 'MISMATCH'}")
        assert ok
        return
    fast_set = {(li, int(j), int(i)) for li, rr, cc, _ in sel for j, i in zip(rr.tolist(), cc.tolist())}
    ref_w    = _np.sort(_np.array([w for w, *_ in ref_k], dtype=_np.float64))
    fast_w   = _np.sort(_np.concatenate([v.double().cpu().numpy() for *_, v in sel]))
    ok = len(ref_k) == len(fast_set) and _np.allclose(ref_w, fast_w, rtol=1e-4, atol=0.0)
    diff = len(ref_set ^ fast_set)
    print(f"[edge-check] {name}: candidates={len(ref_sorted):,}  k={k:,}  "
          f"selected ref/fast={len(ref_set):,}/{len(fast_set):,}  differing edges={diff}  "
          f"-> {'OK' if ok else 'MISMATCH'}{'' if diff == 0 else ' (ties within 1e-4)'}")
    assert ok, f"{name}: fast selection differs from the original loop"


def _conv_l2_edge_step_fast(net: PrunableNetworkConv, *, amount: float, **_) -> None:
    mats    = _pair_weight_mats(net)
    total   = sum(M.numel() for M in mats)
    n_zero  = sum(int((M == 0).sum().item()) for M in mats)
    n_prune = max(1, int(round(amount * total)) - n_zero)
    n_prune = min(n_prune, total - n_zero - 1)
    if n_prune <= 0:
        return
    sel = _lowest_k_pairs(mats, [M != 0 for M in mats], n_prune, col_major=False)
    if EDGE_CHECK:
        G, ordered = _graph_ro(net), _get_ordered_layers(net.model)
        ref = []
        for li, (_, m) in enumerate(ordered):
            sp, src_size = G.graph["spatial_sizes"][li], G.graph["layers"][li]
            out_size = m.out_channels if isinstance(m, nn.Conv2d) else m.out_features
            for j in range(out_size):
                for i in range(src_size):
                    w = _conv_edge_weight(m, j, i, sp)
                    if w != 0.0:
                        ref.append((w, li, j, i))
        ref.sort(key=lambda x: x[0])
        _edge_check("L2", ref, n_prune, sel)
    _zero_pairs_apply(net, sel)
    if L2_LONELY:
        net.remove_lonely_neurons()
    net.commit_pruning()
    net.resync_graph()


def _conv_bridge_edge_step_fast(net: PrunableNetworkConv, *, amount: float, **_) -> None:
    node_to_group = {node: gid for gid, members in enumerate(_bridge_groups(net))
                     for node in members}

    G       = _graph_ro(net)
    ordered = _get_ordered_layers(net.model)
    # v6 fix: pair units (see the slow version)
    mats           = _pair_weight_mats(net)
    total_pairs    = sum(M.numel() for M in mats)
    zero_pairs     = sum(int((M == 0).sum().item()) for M in mats)
    active_edges   = total_pairs - zero_pairs
    edges_to_prune = max(1, int(round(amount * total_pairs)) - zero_pairs)
    edges_to_prune = min(edges_to_prune, active_edges - 1)
    if edges_to_prune <= 0:
        return

    sizes, masks = G.graph["layers"], []
    for li, M in enumerate(mats):
        src = torch.tensor([node_to_group.get((li, i), -1) for i in range(M.shape[1])],
                           dtype=torch.long, device=M.device)
        dst = torch.tensor([node_to_group.get((li + 1, j), -1) for j in range(M.shape[0])],
                           dtype=torch.long, device=M.device)
        valid = (dst[:, None] >= 0) & (src[None, :] >= 0)
        masks.append((M != 0) & valid & (dst[:, None] != src[None, :]))
    sel = _lowest_k_pairs(mats, masks, edges_to_prune, col_major=True)   # graph-edge order: (layer, in, out)

    if EDGE_CHECK:
        sp_sizes = G.graph["spatial_sizes"]
        ref = []
        for (src_node, dst_node, _) in G.edges(data=True):
            sg, dg = node_to_group.get(src_node), node_to_group.get(dst_node)
            if sg is None or dg is None or sg == dg:
                continue
            li, in_i, out_j = src_node[0], src_node[1], dst_node[1]
            ref.append((_conv_edge_weight(ordered[li][1], out_j, in_i, sp_sizes[li]), li, out_j, in_i))
        ref.sort(key=lambda x: x[0])
        _edge_check("Bridge", ref, edges_to_prune, sel)

    if sel is None:
        _conv_l2_edge_step(net, amount=amount)
        return
    _zero_pairs_apply(net, sel)
    net.remove_lonely_neurons()
    net.commit_pruning()
    net.resync_graph()


_conv_bridge_edge_step = _conv_bridge_edge_step_fast if EDGE_FAST else _conv_bridge_edge_step_slow
_conv_l2_edge_step     = _conv_l2_edge_step_fast     if EDGE_FAST else _conv_l2_edge_step_slow


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
                for j, s_j in enumerate(scores.tolist()):
                    importance[(node_layer, j)] += float(s_j) / n_batches

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
    total, zero = _zero_pairs(net)
    active   = total - zero
    to_zero  = max(0, active - target_pairs)
    return float((zero + to_zero) / total) if total else 0.0


# ==================================================================
# Core accuracy-budget loop
# ==================================================================

# ------------------------------------------------------------------
# Accuracy-only evaluation inside the pruning loop.
#   evaluate_model() = predict + metrics + timed_inference (10 warm-up + 200 full-test-set
#   passes + 1000 single-sample passes).  The loop only needs the accuracy, but the timing
#   part took ~80% of the AlexNet branch wall-clock (~23 s per evaluation).
#   default            : evaluate_model (original behaviour)
#   EVAL_FAST=1        : predict + accuracy_score only -- the same predictions, so the same
#                        accuracy; inference timing is still measured for the tables
#                        (eval_nn / _noc_and_record are unchanged).
#   EVAL_CHECK=1       : (verification only, slow; implies FAST) also runs evaluate_model and
#                        asserts both accuracies are equal.
# ------------------------------------------------------------------
EVAL_CHECK: bool = os.environ.get("EVAL_CHECK") == "1"
EVAL_FAST:  bool = EVAL_CHECK or os.environ.get("EVAL_FAST", "1") == "1"


def _loop_accuracy(model, X_test, y_test, device, timings) -> float:
    if not EVAL_FAST:
        return _extract_accuracy(
            evaluate_model(model, X_test, y_test, device=device, timings=timings))
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score
    loader = DataLoader(NumpyDataset(X_test, y_test), batch_size=128, shuffle=False)
    acc = float(accuracy_score(y_test, predict(model, loader, device)))
    if EVAL_CHECK:
        ref = _extract_accuracy(
            evaluate_model(model, X_test, y_test, device=device, timings=timings))
        print(f"[eval-check] fast={acc:.6f} ref={ref:.6f} -> {'IDENTICAL' if acc == ref else 'MISMATCH'}")
        assert acc == ref, "accuracy-only evaluation differs from evaluate_model"
    return acc


def _layer_widths(model: nn.Module) -> str:
    """v8 log helper: output width of every Conv2d / Linear layer, e.g. 'conv 6-16 | fc 120-84-10'."""
    conv = [m.out_channels for m in model.modules() if isinstance(m, nn.Conv2d)]
    fc   = [m.out_features for m in model.modules() if isinstance(m, nn.Linear)]
    return "conv " + "-".join(map(str, conv)) + " | fc " + "-".join(map(str, fc))


def _neuron_end_snapshot(label: str, net, tm) -> None:
    """v8: SNAPSHOT_NEURON_END=1 -- record a NoC stage '<label> [neuron end]' when the neuron phase ends.
    It isolates the neuron-selection effect (same neuron count, before the edge phase) from the edge phase."""
    if not SNAPSHOT_NEURON_END:
        return
    _t0 = time.perf_counter()
    snap_net = PrunableNetworkConv(copy.deepcopy(net.model), eps=0.0)
    _noc_and_record(f"{label} [neuron end]", snap_net, X_test, y_test, device, timings)
    tm["snapshot_s"] += time.perf_counter() - _t0   # measurement overhead


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
    if BRANCHES and label[0].upper() not in BRANCHES:        # v8: BRANCHES=A,D runs only those branches
        print(f"[{label}] skipped (BRANCHES={BRANCHES}).")
        return
    _sync()
    _t_branch = time.perf_counter()          # branch wall-clock starts here
    tm = defaultdict(float)                  # per-component seconds / counters
    _bridge_noc_s[0] = 0.0
    floor             = anchor_accuracy - accuracy_tolerance
    _post_neuron_frac = post_neuron_edge_fraction if post_neuron_edge_fraction is not None \
                        else edge_step_fraction
    _n_per_edge       = max(0, neuron_steps_per_edge_step)
    # v8: env overrides (unset = the values passed by the branch call, i.e. v7 behaviour)
    if NEURON_TOL_ENV is not None and neuron_tolerance is not None:
        neuron_tolerance = NEURON_TOL_ENV
    if NEURON_SHARE_ENV is not None and neuron_tolerance is not None:
        neuron_tolerance = NEURON_SHARE_ENV * accuracy_tolerance      # takes precedence over NEURON_TOL
    if NEURONS_PER_STEP_ENV is not None:
        neurons_per_step = NEURONS_PER_STEP_ENV
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
        print(f"  Neuron phase cap: {_neuron_tol:.4f}"
              f"{' (measured from post-L1 accuracy)' if NEURON_CAP_FROM_L1 else ' (measured from baseline)'}"
              f"  neurons_per_step={neurons_per_step}")
    if thresholds:
        print(f"  Snapshot thresholds: {[f'{t*100:.1f} pp' for t in thresholds]}")
    print(f"  Edge frac (interleaved/post): {edge_step_fraction:.4f}/{_post_neuron_frac:.4f}  "
          f"N:E ratio={_n_per_edge}:1  FT epochs={micro_ft_epochs}  Max steps={max_steps}")
    print("=" * 70)

    net        = PrunableNetworkConv(copy.deepcopy(l1_model_copy), eps=0.0)
    best_model = copy.deepcopy(net.model)
    best_acc   = _loop_accuracy(net.model, X_test, y_test, device, timings)
    print(f"[{label}] Start: acc={best_acc:.4f}  "
          f"fc_neurons={_count_fc_neurons(net.model)}  "
          f"graph_edges={_graph_ro(net).number_of_edges():,}")
    _fc0 = _count_fc_neurons(net.model)                                  # v8: for the neuron-phase log
    _cap_base = max(0.0, anchor_accuracy - best_acc) if NEURON_CAP_FROM_L1 else 0.0   # v8
    if (NEURON_SHARE_ENV is not None and NEURON_SHARE_BASIS == "remaining"
            and neuron_tolerance is not None):
        # v8: the share applies to the budget LEFT after the L1 stage, but the cap stays a cumulative
        # drop measured from the BASELINE:  cap = L1_drop + share * (tolerance - L1_drop)
        _l1_drop  = max(0.0, anchor_accuracy - best_acc)
        _cap_base = 0.0
        _neuron_tol = _l1_drop + NEURON_SHARE_ENV * max(0.0, accuracy_tolerance - _l1_drop)
        print(f"  Neuron share basis=remaining: L1 drop={_l1_drop*100:.2f} pp, "
              f"share={NEURON_SHARE_ENV:.2f} of the remaining {max(0.0, accuracy_tolerance - _l1_drop)*100:.2f} pp "
              f"-> cumulative neuron cap {_neuron_tol*100:.2f} pp from baseline")
    # v8: size-independent neuron phase.  Base = removable FC neurons at the start of the branch.
    _target_removed = (max(1, int(round(NEURON_MAX_FRAC * _fc0)))
                       if NEURON_MAX_FRAC is not None else None)
    _step_n = (max(1, int(round(NEURON_STEP_FRAC * _fc0)))
               if NEURON_STEP_FRAC is not None else neurons_per_step)
    if _target_removed is not None or NEURON_STEP_FRAC is not None:
        print(f"  Neuron phase size: base={_fc0} removable FC neurons  "
              f"target={_target_removed if _target_removed is not None else 'none (cap only)'}"
              f"{'' if NEURON_MAX_FRAC is None else f' ({NEURON_MAX_FRAC:.0%})'}  "
              f"neurons per step={_step_n}"
              f"{'' if NEURON_STEP_FRAC is None else f' ({NEURON_STEP_FRAC:.1%} of base)'}")

    next_snap_idx        = 0
    total_steps          = 0
    neuron_steps_done    = 0
    neuron_exhausted     = False
    edge_exhausted       = False
    neuron_tol_exhausted = False
    step_scale           = 1.0
    fails_at_min         = 0
    edge_hist: list[int] = []     # v7: graph edges after each accepted edge-only step
    cycle_steps          = 0      # v8: accepted steps, used by INTERLEAVE_FIX

    def _ft_eval(ft_seed: int) -> float:
        _sync(); _t0 = time.perf_counter()
        reapply_zero_masks(net)
        fine_tune(net.model, X_train, y_train, device=device, lr=fine_tune_lr,
                  epochs=micro_ft_epochs, seed=ft_seed)
        _sync(); _t1 = time.perf_counter()
        net.commit_pruning()
        net.resync_graph()
        _t2 = time.perf_counter()
        acc_ = _loop_accuracy(net.model, X_test, y_test, device, timings)
        _sync(); _t3 = time.perf_counter()
        tm["finetune_s"]   += _t1 - _t0
        tm["graph_sync_s"] += _t2 - _t1
        tm["eval_s"]       += _t3 - _t2
        tm["n_finetune_rounds"] += 1
        return acc_

    for step in range(1, max_steps + 1):
        if neuron_exhausted and edge_exhausted:
            print(f"[{label}] Both phases exhausted.")
            break

        if NEURON_SEQUENTIAL:
            # v8: original end-to-end design -- neuron steps only until the neuron cap is reached
            # (or the criterion is exhausted), then edge steps only.  Ratio is ignored.
            do_neuron = (not neuron_exhausted) and (not neuron_tol_exhausted)
        elif _n_per_edge == 0:
            do_neuron = False
        elif INTERLEAVE_FIX:
            # v8: cycle position advances on EVERY accepted step -> true N:1 interleaving
            do_neuron = (
                not neuron_exhausted
                and not neuron_tol_exhausted
                and (cycle_steps % (_n_per_edge + 1) < _n_per_edge)
            )
        else:
            # v7 behaviour: neuron_steps_done only grows on neuron steps, so after the first
            # N neuron steps this stays False and the rest of the run is edge-only
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
            _n_this = _step_n
            if _target_removed is not None:
                _n_this = min(_n_this, _target_removed - (_fc0 - _count_fc_neurons(net.model)))
                if _n_this <= 0:                        # target already reached
                    neuron_tol_exhausted = True
                    print(f"[{label}] Step {step}: neuron count target reached "
                          f"({_fc0 - _count_fc_neurons(net.model)}/{_target_removed}).")
                    continue
            kwargs = dict(neuron_step_kwargs)
            kwargs["max_neurons"] = _n_this
            _sync(); _t0 = time.perf_counter()
            n_removed, _ = neuron_step_fn(net, **kwargs)
            _sync(); tm["neuron_step_s"] += time.perf_counter() - _t0
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
            _sync(); _t0 = time.perf_counter()
            total_pairs, zero_pairs = _zero_pairs(net)
            active_pairs = total_pairs - zero_pairs
            pairs_to_prune = max(1, int(active_pairs * active_edge_frac))
            target_pairs   = active_pairs - pairs_to_prune
            if target_pairs <= 0:
                edge_exhausted = True
                continue
            amount = _compute_edge_prune_amount(net, target_pairs)
            _sync(); tm["edge_prep_s"] += time.perf_counter() - _t0
            if not (0.0 < amount < 1.0):
                edge_exhausted = True
                continue
            _sync(); _t0 = time.perf_counter()
            edge_step_fn(net, amount=amount, **edge_step_kwargs)
            _sync(); tm["edge_step_s"] += time.perf_counter() - _t0
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
        cycle_steps += 1
        print(f"[{label}] Step {step:3d} ({phase_label:6s}): "
              f"acc={acc:.4f}  drop={drop:+.4f}  "
              f"fc_neurons={_count_fc_neurons(net.model)}  "
              f"graph_edges={_graph_ro(net).number_of_edges():,}  "
              f"scale={step_scale:.3f}")

        _t0 = time.perf_counter()
        _traj_record(label, total_steps, phase_label, acc, drop, net)
        tm["trajectory_log_s"] += time.perf_counter() - _t0   # measurement overhead
        total_steps += 1

        best_model = copy.deepcopy(net.model)
        best_acc   = acc

        if (not neuron_tol_exhausted and neuron_tolerance is not None
                and drop - _cap_base >= _neuron_tol):
            neuron_tol_exhausted = True
            remaining = accuracy_tolerance - drop
            print(f"[{label}] Step {step}: neuron tol cap "
                  f"(drop={drop:.4f}, L1 part={_cap_base:.4f}, cap={_neuron_tol:.4f}). "
                  f"Reserving {remaining:.4f} for edge-only ({_post_neuron_frac:.4f}).")
            print(f"[{label}] Neuron phase ended: {neuron_steps_done} neuron steps, "
                  f"{_fc0 - _count_fc_neurons(net.model)} FC neurons removed so far "
                  f"({_fc0} -> {_count_fc_neurons(net.model)}).")
            print(f"[{label}] Layer widths at end of neuron phase: {_layer_widths(net.model)}  "
                  f"graph_edges={_graph_ro(net).number_of_edges():,}")
            _neuron_end_snapshot(label, net, tm)

        # v8: neuron-count target (NEURON_MAX_FRAC) reached -> end the neuron phase
        if (not neuron_tol_exhausted and _target_removed is not None and do_neuron
                and (_fc0 - _count_fc_neurons(net.model)) >= _target_removed):
            neuron_tol_exhausted = True
            print(f"[{label}] Step {step}: neuron count target reached "
                  f"(drop={drop:.4f}). Reserving {accuracy_tolerance - drop:.4f} for edge-only "
                  f"({_post_neuron_frac:.4f}).")
            print(f"[{label}] Neuron phase ended: {neuron_steps_done} neuron steps, "
                  f"{_fc0 - _count_fc_neurons(net.model)} FC neurons removed so far "
                  f"({_fc0} -> {_count_fc_neurons(net.model)}).")
            print(f"[{label}] Layer widths at end of neuron phase: {_layer_widths(net.model)}  "
                  f"graph_edges={_graph_ro(net).number_of_edges():,}")
            _neuron_end_snapshot(label, net, tm)

        # v2 fix (reviewer pt.3): record pre_step_model, not current state
        while (next_snap_idx < len(thresholds)
               and drop >= thresholds[next_snap_idx]):
            snap_pct   = thresholds[next_snap_idx] * 100
            snap_label = f"{label} @ drop≤{snap_pct:.1f}%"
            print(f"[{label}] → Snapshot at drop≤{snap_pct:.1f}%: '{snap_label}'")
            _t0 = time.perf_counter()
            snap_net = PrunableNetworkConv(copy.deepcopy(pre_step_model), eps=0.0)
            _noc_and_record(snap_label, snap_net, X_test, y_test, device, timings)
            tm["snapshot_s"] += time.perf_counter() - _t0   # measurement overhead
            next_snap_idx += 1

        # v7: diminishing-returns stop (edge-only phase only; off unless MIN_GAIN_STOP > 0)
        if MIN_GAIN_STOP > 0 and not do_neuron and neuron_phase_done:
            edge_hist.append(_graph_ro(net).number_of_edges())
            if len(edge_hist) > MIN_GAIN_WINDOW:
                old_e = edge_hist[-MIN_GAIN_WINDOW - 1]
                gain  = (old_e - edge_hist[-1]) / max(1, old_e)
                if gain < MIN_GAIN_STOP:
                    print(f"[{label}] Diminishing returns: last {MIN_GAIN_WINDOW} edge steps removed "
                          f"{gain*100:.2f}% < {MIN_GAIN_STOP*100:.2f}% of edges. Stopping.")
                    break

    _sync()
    _t_loop_end = time.perf_counter()
    print(f"\n[{label}] Done. {total_steps} steps. "
          f"acc={best_acc:.4f}  fc_neurons={_count_fc_neurons(net.model)}")
    print(f"[{label}] Final layer widths: {_layer_widths(net.model)}  "
          f"graph_edges={_graph_ro(net).number_of_edges():,}")
    _t0 = time.perf_counter()
    eval_nn(net.model, X_test=X_test, y_test=y_test,
            device=device, timings=timings, evaluate_fn=evaluate_model)
    _t_final_nn = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    _noc_and_record(f"{label} [final]", net, X_test, y_test, device, timings)
    _t_final_rec = time.perf_counter() - _t0
    _sync()
    _t_end = time.perf_counter()

    # ---- timing record for this branch -------------------------------------
    total_wall = _t_end - _t_branch
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
        "loop_wall_s":         _t_loop_end - _t_branch,
        "final_eval_s":        _t_final_nn + _t_final_rec,
        # components of the loop
        "neuron_step_s":       tm["neuron_step_s"],
        "edge_step_s":         tm["edge_step_s"],
        "edge_prep_s":         tm["edge_prep_s"],
        "bridge_grouping_sa_s": _bridge_noc_s[0],             # part of edge_step_s (Bridge only)
        "finetune_s":          tm["finetune_s"],
        "eval_s":              tm["eval_s"],
        "graph_sync_s":        tm["graph_sync_s"],
        "snapshot_s":          tm["snapshot_s"],
        "trajectory_log_s":    tm["trajectory_log_s"],
        "accepted_steps":      total_steps,
        "finetune_rounds":     int(tm["n_finetune_rounds"]),
    }
    branch_timings.append(rec)
    print(f"[{label}] TIMING: total={rec['total_wall_s']:.1f}s  core={rec['core_wall_s']:.1f}s  "
          f"final grouping+SA={rec['final_grouping_sa_s']:.1f}s  "
          f"neuron={rec['neuron_step_s']:.1f}s  edge={rec['edge_step_s']:.1f}s "
          f"(bridge grouping+SA {rec['bridge_grouping_sa_s']:.1f}s, prep {rec['edge_prep_s']:.1f}s)  "
          f"finetune={rec['finetune_s']:.1f}s  graph_sync={rec['graph_sync_s']:.1f}s  "
          f"eval={rec['eval_s']:.1f}s")


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
    nend_rows  = [s for s in stages if s["name"].endswith("[neuron end]")]
    anchor_row = next((s for s in stages if s["name"] == _L1_LABEL), None)

    title_common = [
        f"Snapshot interval: every {SNAPSHOT_INTERVAL*100:.1f} pp",
        "Model: AlexNet (conv1/conv2/conv3/conv4/conv5/c6/c7/output) | Dataset: CIFAR-10",
        f"Tolerance: ≤{ACCURACY_TOLERANCE*100:.1f} pp drop from BASELINE",
        "Graph: channel-level (Conv2d channels + FC neurons) | Mesh: 11×11=121 cores",
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
        row_stages=[base] + ([anchor_row] if anchor_row else []) + nend_rows + final_rows,
        base_stage=base,
    )


def save_trajectories(path_csv=f"trajectory_AlexNet_CIFAR10_v8{RUN_TAG}_seed{seed}.csv",
                      path_json=f"trajectory_AlexNet_CIFAR10_v8{RUN_TAG}_seed{seed}.json"):
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
    patience=int(os.environ.get("TRAIN_PATIENCE", "20")), min_delta=1e-4,
    restore_best=os.environ.get("RESTORE_BEST", "1") != "0",
)

# ------------------------------------------------------------------
# v8: BASELINE_REANNEAL=1 (default OFF) -- the saved dense AlexNet appears to be an un-annealed snapshot
# (cosine schedule over 200 epochs cut by early stopping), so a low-LR fine-tune adds ~+1 pp by itself and the
# L1-stage accuracy change is dominated by the fine-tune, not by the pruning.  With this switch the dense model is
# first fine-tuned with a cosine LR decay (BASELINE_REANNEAL_LR -> 0 over BASELINE_REANNEAL_EPOCHS) and that model
# becomes the baseline.  It is built ONCE (fixed seed 42) and saved to DNN_AlexNet_CIFAR10_annealed.pth, which every
# later run/seed loads, so all seeds share one baseline.  The original .pth is not touched.
# ------------------------------------------------------------------
if os.environ.get("BASELINE_REANNEAL") == "1":
    _ra_path = os.environ.get("BASELINE_REANNEAL_PATH", "DNN_AlexNet_CIFAR10_annealed.pth")
    if os.path.exists(_ra_path):
        model.load_state_dict(torch.load(_ra_path, map_location=device))
        model.eval()
        print(f"[BaselineReanneal] loaded {_ra_path}")
    else:
        from torch.utils.data import DataLoader
        from DNN_AlexNet_CIFAR10 import set_seed as _set_seed
        _ra_lr = float(os.environ.get("BASELINE_REANNEAL_LR", "2e-4"))
        _ra_ep = int(os.environ.get("BASELINE_REANNEAL_EPOCHS", "10"))
        _set_seed(42)
        _ra_loader = DataLoader(NumpyDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        _ra_test   = DataLoader(NumpyDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
        model.eval()
        _acc = lambda: float((np.asarray(predict(model, _ra_test, device)) == np.asarray(y_test)).mean())
        _acc0 = _acc()
        _opt  = torch.optim.Adam(model.parameters(), lr=_ra_lr, weight_decay=weight_decay)
        _sch  = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=_ra_ep * len(_ra_loader))
        _crit = nn.CrossEntropyLoss()
        model.train()
        for _e in range(1, _ra_ep + 1):
            for _Xb, _yb in _ra_loader:
                _Xb, _yb = _Xb.to(device), _yb.to(device)
                _opt.zero_grad(set_to_none=True)
                _crit(model(_Xb), _yb).backward()
                _opt.step()
                _sch.step()
            model.eval()
            _a = _acc()
            print(f"[BaselineReanneal] epoch {_e}/{_ra_ep}  lr_end={_sch.get_last_lr()[0]:.2e}  test acc={_a:.4f}")
            model.train()
        model.eval()
        torch.save(model.state_dict(), _ra_path)
        print(f"[BaselineReanneal] accuracy {_acc0:.4f} -> {_a:.4f}  (lr {_ra_lr:g}->0, {_ra_ep} epochs); saved {_ra_path}")

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

_sync()
_t_pre0 = time.perf_counter()               # shared L1 pre-pruning timing starts
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
    fine_tune(net.model, X_train, y_train, device=device, lr=fine_tune_lr,
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
_sync()
shared_pre_s.update({
    "l1_stage_total_s": time.perf_counter() - _t_pre0,       # prune + fine-tune + record
    "l1_grouping_sa_s": _last_record_s.get("grouping_sa_s", float("nan")),
})
print(f"[timing] Shared L1 pre-pruning stage: {shared_pre_s['l1_stage_total_s']:.1f}s "
      f"(grouping+SA of the L1 model: {shared_pre_s['l1_grouping_sa_s']:.1f}s)")

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
print(f"[Checkpoint] fine_tune_lr = {fine_tune_lr:g}  l1_target = {l1_target:g}")
print(f"[Checkpoint] v8 neuron switches: NEURON_TOL={NEURON_TOL_ENV}  "
      f"NEURON_CAP_FROM_L1={NEURON_CAP_FROM_L1}  NEURONS_PER_STEP={NEURONS_PER_STEP_ENV}  "
      f"RUN_TAG='{RUN_TAG}'  L2_LONELY={L2_LONELY}  "
      f"INTERLEAVE_FIX={INTERLEAVE_FIX}  NEURON_SEQUENTIAL={NEURON_SEQUENTIAL}  "
      f"NEURON_MAX_FRAC={NEURON_MAX_FRAC}  NEURON_STEP_FRAC={NEURON_STEP_FRAC}  "
      f"BRANCHES={BRANCHES or 'all'}  "
      f"ACCURACY_TOLERANCE={ACCURACY_TOLERANCE}  NEURON_SHARE={NEURON_SHARE_ENV}  "
      f"NEURON_SHARE_BASIS={NEURON_SHARE_BASIS}  "
      f"NEURON_CAP_FROM_L1={NEURON_CAP_FROM_L1}")

# --- diagnostic (read-only): share of edges that cross group boundaries after L1 ---
try:
    _grp = indegree_grouping(net.graph, N, 10_000.0)
    _n2g = {n: g for g, ms in enumerate(_grp) for n in ms}
    _tot = _cross = _ung = 0
    for _u, _v in net.graph.edges():
        if _u in _n2g and _v in _n2g:
            _tot += 1
            _cross += (_n2g[_u] != _n2g[_v])
        else:
            _ung += 1
    print(f"[Checkpoint] cross-group edges after L1: {_cross:,} / {_tot:,} "
          f"({_cross/max(_tot,1):.1%}); groups = {len(_grp)}; edges with ungrouped endpoint = {_ung:,}")
except Exception as _e:
    print(f"[Checkpoint] cross-group diagnostic skipped: {_e}")

# LR-sweep mode: set STOP_AFTER_L1=1 to stop after the baseline + L1 stage.
if os.environ.get("STOP_AFTER_L1"):
    print_summary_tables()
    raise SystemExit(0)

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
            ("bridge_grouping_sa_s", "(Bridge grp+SA)"), ("edge_prep_s", "Edge prep"),
            ("finetune_s", "Fine-tune"), ("graph_sync_s", "Graph sync"), ("eval_s", "Eval"),
            ("snapshot_s", "Snapshots"), ("trajectory_log_s", "Traj log"),
            ("accepted_steps", "Steps"), ("finetune_rounds", "FT rounds")]
    print("\n" + "=" * 180)
    print(f"  WALL-CLOCK TIMING per branch (seconds, seed {seed}).  "
          "Total = whole branch incl. snapshots/logging/final eval;  "
          "Core = Total minus snapshots and trajectory logging.")
    print(f"  Shared L1 pre-pruning (not in any branch): "
          f"{shared_pre_s.get('l1_stage_total_s', float('nan')):.1f}s")
    print("=" * 180)
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
    stem = f"timing_AlexNet_CIFAR10_v8{RUN_TAG}_seed{seed}"
    with open(stem + ".json", "w") as f:
        json.dump({"seed": seed, "shared_pre": shared_pre_s, "branches": branch_timings},
                  f, indent=2, default=float)
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(branch_timings[0].keys()))
        w.writeheader()
        w.writerows(branch_timings)
    print(f"[timing] Saved {stem}.json / .csv")



print_and_save_timing()

save_trajectories()
