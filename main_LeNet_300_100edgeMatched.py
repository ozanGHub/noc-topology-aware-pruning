import math
import copy

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from DNN_LeNet_300_100 import load_or_train, fine_tune, evaluate_model
from prunable_network import PrunableNetwork
from eval_helper import *
from visualize import visualize_model
from mapping_helpers import build_wctg_structural, compute_structural_cost

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

data_dir: str       = "./data"
model_path: str     = "DNN_LeNet_300_100.pth"
seed: int           = 42
test_size: float    = 0.2
batch_size: int     = 64
lr: float           = 1e-3
epochs: int         = 200    # early stopping will halt well before this
weight_decay: float = 1e-4
threshold: float    = 0.5
force_train: bool   = True

l1_target: float  = 0.25    # step 2: iterative L1 target
l1_steps: int     = 5
ft_epochs: int    = 10

low_mag_percentile: float = 75.0
degree_quantile: float    = 0.25
min_in_degree: int        = 0

bridge_amount: float = 0.25

zmft_epochs: int = 10        # fine-tune epochs after neuron pruning

P: int         = 36
N: int         = 36
mesh_rows: int = int(math.sqrt(P))
mesh_cols: int = int(math.sqrt(P))

# ------------------------------------------------------------------
# Metrics tracker
# ------------------------------------------------------------------

stages: list[dict] = []


def _extract_accuracy(report):
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


def _extract_inference_time(report):
    if not hasattr(report, "batch_ms"):
        return {"batch_ms": float("nan")}
    return {"batch_ms": float(report.batch_ms)}


def _extract_metrics(report):
    result = {
        "f1":            float("nan"),
        "precision":     float("nan"),
        "recall":        float("nan"),
        "state_size_kb": float("nan"),
    }
    for attr in ("f1", "precision", "recall", "state_size_kb"):
        if hasattr(report, attr):
            result[attr] = float(getattr(report, attr))
    return result


def compute_model_flops(model: nn.Module) -> int:
    """Compute total multiply-accumulate operations (MACs) for one forward pass.

    For each nn.Linear layer: MACs = in_features × out_features.
    Total FLOPs = 2 × Σ MACs (multiply + add per operation).

    This uses the *dense* layer dimensions (after structural/neuron pruning),
    so it correctly reflects the actual compute cost on standard hardware
    regardless of weight sparsity — matching what batch inference time measures.
    """
    total_macs = 0
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            total_macs += mod.in_features * mod.out_features
    return 2 * total_macs


def record_stage(name, net, noc, X_test, y_test, device, timings):
    report  = evaluate_model(net.model, X_test, y_test, device=device, timings=timings)
    acc     = _extract_accuracy(report)
    inf     = _extract_inference_time(report)
    metrics = _extract_metrics(report)

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


def reapply_zero_masks(net: PrunableNetwork) -> None:
    for mod in net.model.modules():
        if isinstance(mod, nn.Linear):
            if hasattr(mod, "weight_orig"):
                prune.remove(mod, "weight")
            mask = (mod.weight.data != 0).float()
            if mask.all():
                continue
            prune.custom_from_mask(mod, "weight", mask)
            #prune.remove(mod, "weight")


def make_metrics(stages: list) -> dict:
    if len(stages) < 1:
        return None
    base = stages[0]
    cur  = stages[-1]
    return {
        "baseline": {
            "accuracy": base["accuracy"],
            "cost":     base["cost"],
            "sp_cost":  base["sp_cost"],
            "flops":    base["flops"],
            "edges":    base["edges"],
            "neurons":  base["neurons"],
            "batch_ms": base["batch_ms"],
        },
        "current": {
            "accuracy": cur["accuracy"],
            "cost":     cur["cost"],
            "sp_cost":  cur["sp_cost"],
            "flops":    cur["flops"],
            "edges":    cur["edges"],
            "neurons":  cur["neurons"],
            "batch_ms": cur["batch_ms"],
        },
    }


def print_summary_table():
    if not stages:
        return

    base     = stages[0]
    w_name   = max(len("Stage"), *(len(s["name"]) for s in stages)) + 2
    w_val    = 22
    num_cols = 11
    sep      = "+" + "-" * w_name + "+" + ("-" * w_val + "+") * num_cols

    def fmt_delta(cur, base_val):
        if base_val == 0:
            return f"{cur:>10}"
        pct  = (cur - base_val) / base_val * 100
        sign = "+" if pct >= 0 else ""
        return f"{cur:>10.4f}  ({sign}{pct:.2f}%)"

    def fmt_delta_int(cur, base_val):
        if base_val == 0:
            return f"{cur:>10}"
        pct  = (cur - base_val) / base_val * 100
        sign = "+" if pct >= 0 else ""
        return f"{cur:>10,}  ({sign}{pct:.2f}%)"

    def fmt_ms(cur, base_val):
        if base_val == 0 or math.isnan(base_val):
            return f"{cur:>10.2f} ms"
        pct  = (cur - base_val) / base_val * 100
        sign = "+" if pct >= 0 else ""
        return f"{cur:>7.2f} ms  ({sign}{pct:.2f}%)"

    def fmt_kb(cur, base_val):
        if base_val == 0 or math.isnan(base_val):
            return f"{cur:>8.2f} KB"
        pct  = (cur - base_val) / base_val * 100
        sign = "+" if pct >= 0 else ""
        return f"{cur:>5.2f} KB  ({sign}{pct:.2f}%)"

    print("\n")
    print("=" * (w_name + w_val * num_cols + num_cols + 1))
    print("  PIPELINE COMPARISON — Baseline vs L1(25%) vs Comm.Cost vs Bridge vs matched variants")
    print("  Model: LeNet 784-300-100-10 | Dataset: MNIST")
    print("  *-matched: saved 25%-L1 model pre-pruned to match Comm.Cost edge count (same fine-tune)")
    print("  FLOPs = 2 x sum(in_features x out_features) — dense layer cost, aligns with inference time")
    print("=" * (w_name + w_val * num_cols + num_cols + 1))
    print(sep)
    print(f"|{'Stage':^{w_name}}|{'Accuracy':^{w_val}}|{'Comm. Cost (WCTG n2n)':^{w_val}}|"
          f"{'Cross Group Comm. Cost':^{w_val}}|{'Edges':^{w_val}}|{'Neurons':^{w_val}}|"
          f"{'Batch Inf. (ms)':^{w_val}}|{'FLOPs':^{w_val}}|{'Model Size (KB)':^{w_val}}|"
          f"{'F1':^{w_val}}|{'Precision':^{w_val}}|{'Recall':^{w_val}}|")
    print(sep)

    base_batch = base["batch_ms"]
    base_size  = base["state_size_kb"]

    for s in stages:
        name    = s["name"]
        acc_s   = fmt_delta(s["accuracy"],  base["accuracy"])  if s is not base else f'{s["accuracy"]:>10.4f}  (baseline)'
        cost_s  = fmt_delta(s["cost"],      base["cost"])      if s is not base else f'{s["cost"]:>10.4f}  (baseline)'
        spc_s   = fmt_delta(s["sp_cost"],   base["sp_cost"])   if s is not base else f'{s["sp_cost"]:>10.4f}  (baseline)'
        edge_s  = fmt_delta_int(s["edges"],   base["edges"])   if s is not base else f'{s["edges"]:>10,}  (baseline)'
        neur_s  = fmt_delta_int(s["neurons"], base["neurons"]) if s is not base else f'{s["neurons"]:>10,}  (baseline)'
        batch_s = fmt_ms(s["batch_ms"],     base_batch)        if s is not base else f'{s["batch_ms"]:>7.2f} ms  (baseline)'
        flop_s  = fmt_delta_int(s["flops"],   base["flops"])   if s is not base else f'{s["flops"]:>10,}  (baseline)'
        size_s  = fmt_kb(s["state_size_kb"], base_size)        if s is not base else f'{s["state_size_kb"]:>5.2f} KB  (baseline)'
        f1_s    = fmt_delta(s["f1"],        base["f1"])        if s is not base else f'{s["f1"]:>10.4f}  (baseline)'
        prec_s  = fmt_delta(s["precision"], base["precision"]) if s is not base else f'{s["precision"]:>10.4f}  (baseline)'
        rec_s   = fmt_delta(s["recall"],    base["recall"])    if s is not base else f'{s["recall"]:>10.4f}  (baseline)'

        print(f"|{name:<{w_name}}|{acc_s:>{w_val}}|{cost_s:>{w_val}}|{spc_s:>{w_val}}|"
              f"{edge_s:>{w_val}}|{neur_s:>{w_val}}|{batch_s:>{w_val}}|{flop_s:>{w_val}}|"
              f"{size_s:>{w_val}}|{f1_s:>{w_val}}|{prec_s:>{w_val}}|{rec_s:>{w_val}}|")
        print(sep)

    print()



# ------------------------------------------------------------------
# 1. Train / load
# ------------------------------------------------------------------

model, scaler, (X_train, X_test, y_train, y_test), timings, device = load_or_train(
    data_dir, model_path,
    seed=seed, test_size=test_size, batch_size=batch_size,
    lr=lr, epochs=epochs, weight_decay=weight_decay,
    threshold=threshold, force_train=force_train,
    patience=15,
    min_delta=1e-4,
)

net = PrunableNetwork(model, eps=0.0)

# ------------------------------------------------------------------
# 2. Baseline evaluation
# ------------------------------------------------------------------

print("\n=== Initial model ===")
eval_nn(net.model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
noc = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
               title="Baseline", visualize=False)
record_stage("Baseline", net, noc, X_test, y_test, device, timings)
noc.visualize_all(title="Baseline", metrics=make_metrics(stages))

# ------------------------------------------------------------------
# 3. Iterative L1 link pruning + fine-tuning
# ------------------------------------------------------------------

original_edge_count = net.graph.number_of_edges()
target_removals = round(original_edge_count * l1_target)

total_weights = sum(
    p.numel() for name, p in net.model.named_parameters()
    if "weight" in name and p.dim() == 2
)

for step in range(1, l1_steps + 1):
    print("L1 loop:", step)
    already_removed     = original_edge_count - net.graph.number_of_edges()
    remaining_to_remove = target_removals - already_removed
    steps_left          = l1_steps - step + 1
    this_step_target    = round(remaining_to_remove / steps_left)

    cumulative_target = already_removed + this_step_target
    amount            = cumulative_target / total_weights

    if amount >= 1.0:
        print(f"  [Step {step}/{l1_steps}] Skipping — not enough edges remaining.")
        break

    net.prune_links_l1_global(amount=amount)
    fine_tune(net.model, X_train, y_train, device=device, lr=lr,
              epochs=ft_epochs, seed=seed + step)

    removed = original_edge_count - net.graph.number_of_edges()
    print(f"  [Step {step}/{l1_steps}] Edges removed: {removed}/{original_edge_count} "
          f"({removed / original_edge_count:.1%} of original)")

net.commit_pruning()
net.remove_lonely_neurons()

print(f"\n=== After iterative L1 pruning ({l1_target:.0%} of original links removed) ===")
eval_nn(net.model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
noc = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
               title="After L1 pruning", visualize=False)
record_stage("After L1 25% pre pruning", net, noc, X_test, y_test, device, timings)
noc.visualize_all(title="After L1 pruning", metrics=make_metrics(stages))

# ------------------------------------------------------------------
# Save L1 checkpoint — used later for the L1-matched comparison
# We deepcopy the full model object (not just state_dict) because
# remove_lonely_neurons has already structurally resized the layers
# (e.g. fc1: 300→299), so the original MLP architecture no longer
# matches. Deepcopying the model preserves the resized architecture.
# ------------------------------------------------------------------
l1_model_copy  = copy.deepcopy(net.model)
l1_edge_count  = net.graph.number_of_edges()
print(f"\n[Checkpoint] Saved L1 model — {l1_edge_count:,} edges remaining.")

# ------------------------------------------------------------------
# 4. Comm.Cost aware neuron pruning + fine-tune
# ------------------------------------------------------------------

net.prune_high_indegree_low_magnitude(
    low_mag_percentile=low_mag_percentile,
    degree_quantile=degree_quantile,
    min_in_degree=min_in_degree,
)
net.remove_lonely_neurons()

reapply_zero_masks(net)
fine_tune(net.model, X_train, y_train, device=device, lr=lr,
          epochs=zmft_epochs, seed=seed)


#TBD
net.commit_pruning()
net.resync_graph()  #added due to additional reapply_zero_masks(net) and fine_tune() step above
#TBD


print("\n=== After Comm.Cost aware neuron pruning ===")
eval_nn(net.model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
noc = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
               title="After Comm.Cost aware pruning", visualize=False)
record_stage("After Comm.Cost pruning", net, noc, X_test, y_test, device, timings)
noc.visualize_all(title="After Comm.Cost pruning", metrics=make_metrics(stages))

# Record edge count achieved by Comm.Cost pruning — used to calibrate step 6
commcost_edge_count = net.graph.number_of_edges()
print(f"\n[Comparison] Comm.Cost pruning edge count: {commcost_edge_count:,} "
      f"(started from {l1_edge_count:,} after L1)")

# ------------------------------------------------------------------
# 5. Heuristic bridge-link pruning
# ------------------------------------------------------------------

net.prune_links_heuristic(
    groups=noc.groups,
    amount=bridge_amount,
    debug=False,
)
net.commit_pruning()

print("\n=== After heuristic bridge-link pruning ===")
eval_nn(net.model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
noc_bridge = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                      title="After Bridge-Link pruning", visualize=False)
record_stage("After Bridge-Link pruning", net, noc_bridge, X_test, y_test, device, timings)
noc_bridge.visualize_all(title="After Bridge-Link pruning", metrics=make_metrics(stages))

# ------------------------------------------------------------------
# 6. L1-matched comparison
#    Reload saved L1 checkpoint → single-shot L1 to reach same edge
#    count as Comm.Cost step → evaluate immediately (with same fine-tune as Comm. Cost)
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("  STEP 6: L1-matched comparison")
print(f"  Target: match Comm.Cost edge count ({commcost_edge_count:,} edges)")
print(f"  Starting from L1 checkpoint ({l1_edge_count:,} edges)")
print("=" * 70)

# Wrap the deepcopied model directly — architecture already matches
# the resized layers from remove_lonely_neurons.
net_l1 = PrunableNetwork(copy.deepcopy(l1_model_copy), eps=0.0)

# Verify edge count matches our saved value
assert net_l1.graph.number_of_edges() == l1_edge_count, (
    f"Checkpoint edge count mismatch: "
    f"expected {l1_edge_count}, got {net_l1.graph.number_of_edges()}"
)

# Compute single-shot L1 amount needed to reach commcost_edge_count.
#
# Key insight: prune_links_l1_global(amount) zeroes the globally weakest
# `amount` fraction of ALL weight slots (including already-zero slots from
# the previous L1 step). After the L1 checkpoint, ~25% of weights are
# already zero, so a naive amount = edges_to_remove / total_weights
# underestimates — most of the zeroed slots were already zero.
#
# Correct formula: we need the cumulative fraction of total weights that
# must be zero AFTER this step. That equals:
#   (total_weights - commcost_edge_count) / total_weights
# because commcost_edge_count is the number of non-zero weights we want
# to keep (each active edge = one non-zero weight).
total_weights_l1 = sum(
    p.numel() for name, p in net_l1.model.named_parameters()
    if "weight" in name and p.dim() == 2
)

edges_to_remove = l1_edge_count - commcost_edge_count
# Cumulative amount: fraction of total weights that should be zero after pruning
amount_needed   = (total_weights_l1 - commcost_edge_count) / total_weights_l1

print(f"\n[L1-matched] Need to remove {edges_to_remove:,} more edges "
      f"({edges_to_remove / l1_edge_count:.1%} of L1 checkpoint edges)")
print(f"[L1-matched] Total weight slots: {total_weights_l1:,}")
print(f"[L1-matched] Cumulative L1 amount (fraction to zero): {amount_needed:.4f}")

if amount_needed <= 0:
    print("[L1-matched] WARNING: Comm.Cost edge count >= total weights — skipping.")
elif amount_needed >= 1.0:
    print("[L1-matched] WARNING: amount_needed >= 1.0, clamping to 0.99.")
    amount_needed = 0.99

if 0 < amount_needed < 1.0:
    net_l1.prune_links_l1_global(amount=amount_needed)
    net_l1.commit_pruning()
    #net_l1.remove_lonely_neurons()

matched_edge_count = net_l1.graph.number_of_edges()
print(f"[L1-matched] Edges after pruning: {matched_edge_count:,} "
      f"(target was {commcost_edge_count:,}, "
      f"diff={matched_edge_count - commcost_edge_count:+,})")

"""
Fine Tune Similar to Comm. Cost Fine Tuning
"""
reapply_zero_masks(net_l1)
fine_tune(net_l1.model, X_train, y_train, device=device, lr=lr,
          epochs=zmft_epochs, seed=seed)
net.remove_lonely_neurons()

print("\n=== After L1-matched pruning (with same fine-tune as Comm. Cost) ===")
eval_nn(net_l1.model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
noc_l1m = eval_noc(net_l1.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                   title="After L1-matched", visualize=False)
record_stage("After L1-matched", net_l1, noc_l1m, X_test, y_test, device, timings)
noc_l1m.visualize_all(title="After L1-matched", metrics=make_metrics(stages))

# ------------------------------------------------------------------
# Generic helper: run a matched-pruning comparison branch
# ------------------------------------------------------------------

def run_matched_branch(
    label: str,
    l1_model_copy: nn.Module,
    l1_edge_count: int,
    commcost_edge_count: int,
    prune_fn,
    prune_kwargs: dict,
    X_train, y_train, X_test, y_test,
    device, timings,
) -> None:
    """Reload the L1 checkpoint, prune to commcost_edge_count with prune_fn,
    fine-tune identically to the Comm.Cost branch, then record the stage."""
    print("\n" + "=" * 70)
    print(f"  STEP: {label}")
    print(f"  Target: match Comm.Cost edge count ({commcost_edge_count:,} edges)")
    print(f"  Starting from L1 checkpoint ({l1_edge_count:,} edges)")
    print("=" * 70)

    net_branch = PrunableNetwork(copy.deepcopy(l1_model_copy), eps=0.0)

    assert net_branch.graph.number_of_edges() == l1_edge_count, (
        f"Checkpoint edge count mismatch: "
        f"expected {l1_edge_count}, got {net_branch.graph.number_of_edges()}"
    )

    total_weights_branch = sum(
        p.numel() for name, p in net_branch.model.named_parameters()
        if "weight" in name and p.dim() == 2
    )
    amount_needed = (total_weights_branch - commcost_edge_count) / total_weights_branch

    print(f"\n[{label}] Total weight slots   : {total_weights_branch:,}")
    print(f"[{label}] Cumulative amount     : {amount_needed:.4f}")

    if amount_needed <= 0:
        print(f"[{label}] WARNING: Comm.Cost edge count >= total weights — skipping.")
        return
    if amount_needed >= 1.0:
        print(f"[{label}] WARNING: amount_needed >= 1.0, clamping to 0.99.")
        amount_needed = 0.99

    if 0 < amount_needed < 1.0:
        prune_fn(net_branch.model, amount_needed, **prune_kwargs)
        net_branch.commit_pruning()

    matched_edges = net_branch.graph.number_of_edges()
    print(f"[{label}] Edges after pruning   : {matched_edges:,} "
          f"(target was {commcost_edge_count:,}, diff={matched_edges - commcost_edge_count:+,})")

    reapply_zero_masks(net_branch)
    fine_tune(net_branch.model, X_train, y_train, device=device, lr=lr,
              epochs=zmft_epochs, seed=seed)
    
    print(f"\n=== {label} ===")
    eval_nn(net_branch.model, X_test=X_test, y_test=y_test, device=device,
            timings=timings, evaluate_fn=evaluate_model)
    noc_branch = eval_noc(net_branch.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                          title=label, visualize=False)
    record_stage(label, net_branch, noc_branch, X_test, y_test, device, timings)
    noc_branch.visualize_all(title=label, metrics=make_metrics(stages))


# ------------------------------------------------------------------
# Custom pruning criterion helpers (L2, Taylor, SNIP)
# ------------------------------------------------------------------

class L2Unstructured(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
        self.amount = amount

    def compute_mask(self, t, default_mask):
        nparams_toprune = int(round(self.amount * t.nelement()))
        mask = default_mask.clone()
        scores = (t ** 2).view(-1)
        active = mask.view(-1).bool()
        active_scores = scores.clone()
        active_scores[~active] = float("inf")
        if nparams_toprune > 0:
            threshold = torch.topk(active_scores, nparams_toprune, largest=False).values.max()
            mask.view(-1)[active_scores <= threshold] = 0
        return mask

    @classmethod
    def apply(cls, module, name, amount):
        return super().apply(module, name, amount=amount)


def prune_l2_global(model: nn.Module, amount: float) -> None:
    parameters = [(mod, "weight") for mod in model.modules() if isinstance(mod, nn.Linear)]
    prune.global_unstructured(parameters, pruning_method=L2Unstructured, amount=amount)


def _accumulate_taylor_scores(model, X_calib, y_calib, device, n_batches=10, batch_sz=128):
    import numpy as np
    was_training = model.training
    model.train()
    model.zero_grad()
    if isinstance(X_calib, np.ndarray):
        X_calib = torch.from_numpy(X_calib).float()
    if isinstance(y_calib, np.ndarray):
        y_calib = torch.from_numpy(y_calib).long()
    criterion = nn.CrossEntropyLoss()
    n = X_calib.shape[0]
    importance = {}
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            importance[mod] = torch.zeros_like(mod.weight.data)
    for b in range(n_batches):
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(b))[:batch_sz]
        xb  = X_calib[idx].to(device)
        yb  = y_calib[idx].to(device)
        out  = model(xb)
        loss = nn.CrossEntropyLoss()(out, yb)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for mod in model.modules():
                if isinstance(mod, nn.Linear) and mod.weight.grad is not None:
                    importance[mod] += (mod.weight.data * mod.weight.grad).abs()
    model.zero_grad()
    if not was_training:
        model.eval()
    for mod in importance:
        importance[mod] /= n_batches
    return importance


def prune_taylor_global(model, amount, X_calib, y_calib, device, n_batches=10, batch_sz=128):
    importance  = _accumulate_taylor_scores(model, X_calib, y_calib, device, n_batches, batch_sz)
    all_scores  = torch.cat([s.view(-1) for s in importance.values()])
    n_prune     = int(round(amount * all_scores.numel()))
    if n_prune <= 0:
        return
    threshold = torch.topk(all_scores, n_prune, largest=False).values.max()
    with torch.no_grad():
        for mod, scores in importance.items():
            mod.weight.data.mul_((scores > threshold).float())


def prune_snip_global(model, amount, X_calib, y_calib, device, n_batches=10, batch_sz=128):
    # Post-training SNIP = Taylor (same |w·g| formula)
    prune_taylor_global(model, amount, X_calib, y_calib, device, n_batches, batch_sz)


# ------------------------------------------------------------------
# 7. L2-matched comparison
# ------------------------------------------------------------------

run_matched_branch(
    label="After L2-matched",
    l1_model_copy=l1_model_copy,
    l1_edge_count=l1_edge_count,
    commcost_edge_count=commcost_edge_count,
    prune_fn=lambda model, amount: prune_l2_global(model, amount),
    prune_kwargs={},
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
)

# ------------------------------------------------------------------
# 8. Taylor-matched comparison
# ------------------------------------------------------------------

run_matched_branch(
    label="After Taylor-matched",
    l1_model_copy=l1_model_copy,
    l1_edge_count=l1_edge_count,
    commcost_edge_count=commcost_edge_count,
    prune_fn=prune_taylor_global,
    prune_kwargs={
        "X_calib":   X_train,
        "y_calib":   y_train,
        "device":    device,
        "n_batches": 10,
        "batch_sz":  128,
    },
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
)

# ------------------------------------------------------------------
# 9. SNIP-matched comparison
# ------------------------------------------------------------------

run_matched_branch(
    label="After SNIP-matched",
    l1_model_copy=l1_model_copy,
    l1_edge_count=l1_edge_count,
    commcost_edge_count=commcost_edge_count,
    prune_fn=prune_snip_global,
    prune_kwargs={
        "X_calib":   X_train,
        "y_calib":   y_train,
        "device":    device,
        "n_batches": 10,
        "batch_sz":  128,
    },
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
)

# ------------------------------------------------------------------
# 10. Final comparison summary
# ------------------------------------------------------------------

print_summary_table()
