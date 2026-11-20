import math
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from DNN_LeNet_500_300 import load_or_train, fine_tune, evaluate_model
from prunable_network import PrunableNetwork
from eval_helper import *
from visualize import visualize_model
from mapping_helpers import build_wctg_structural, compute_structural_cost

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

data_dir: str       = "./data"
model_path: str     = "DNN_LeNet_500_300.pth"
seed: int           = 42
test_size: float    = 0.2
batch_size: int     = 64
lr: float           = 1e-3
epochs: int         = 200    # early stopping will halt well before this
weight_decay: float = 1e-4
threshold: float    = 0.5
force_train: bool   = True

# Unstructured L1 edge pre-pruning target (same as original edge file)
l1_target: float = 0.25
l1_steps: int    = 5
ft_epochs: int   = 10

low_mag_percentile: float = 75.0
degree_quantile: float    = 0.25
min_in_degree: int        = 0

bridge_amount: float = 0.25

zmft_epochs: int = 10        # fine-tune epochs after neuron pruning

P: int         = 64
N: int         = 64
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

    After structural / neuron pruning the layer dimensions shrink, so this
    correctly reflects the actual compute cost on standard hardware.
					  
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
    print("  PIPELINE COMPARISON — Baseline vs L1(25% edges) vs Comm.Cost vs Bridge vs neuron-matched variants")
    print("  Model: LeNet 784-500-300-10 | Dataset: MNIST")
    print("  *-matched: saved unstructured-L1 model pre-pruned to match Comm.Cost neuron count (same fine-tune)")
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


# ==================================================================
# Structured neuron pruning helpers
# ==================================================================
# All helpers below operate at the NEURON level (structured pruning).
# They zero entire output rows of weight matrices (= output neurons).
# After pruning the caller must invoke:
#   physically_remove_zeroed_neurons(model)  — shrinks Linear dims
#   net.commit_pruning()                     — rebuilds the graph
#   net.remove_lonely_neurons()              — drops dangling nodes
# Calling physically_remove_zeroed_neurons is essential: without it
# _count_hidden_neurons still counts zeroed-out rows as live neurons,
# so the matched-branch neuron counts would not reach commcost_neuron_count.
# ==================================================================

# ------------------------------------------------------------------
# Helper: count hidden neurons across all intermediate Linear layers
# (excludes the final output layer)
# ------------------------------------------------------------------

def _count_hidden_neurons(model: nn.Module) -> int:
    """Return total hidden neuron count (all Linear layers except the last)."""
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return sum(m.out_features for m in linears[:-1])


# ------------------------------------------------------------------
# Structured L1 neuron pruning  (per-layer, row-wise L1 norm)
# ------------------------------------------------------------------

def prune_neurons_l1_structured(model: nn.Module, amount: float) -> None:
    """Zero the weakest `amount` fraction of output neurons in each hidden
    layer independently, ranked by L1 row-norm.

    Per-layer pruning prevents a smaller layer from being disproportionately
    gutted by a global threshold skewed by a wider layer's magnitude range.
    """
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    hidden  = linears[:-1]   # exclude the classifier head

															
						 
    for layer in hidden:
        scores  = layer.weight.data.abs().sum(dim=1)   # [out_features]
									   

																						
        n_prune = max(1, int(round(amount * scores.numel())))
        if n_prune >= scores.numel():
            continue   # safety: never zero all neurons in a layer
        threshold = torch.topk(scores, n_prune, largest=False).values.max()

						
												   
        mask = (scores > threshold).float().to(layer.weight.device)
        layer.weight.data *= mask.unsqueeze(1)
        if layer.bias is not None:
            layer.bias.data *= mask


# ------------------------------------------------------------------
# Structured L2 neuron pruning  (per-layer, row-wise L2 norm)
# ------------------------------------------------------------------

def prune_neurons_l2_structured(model: nn.Module, amount: float) -> None:
    """Zero the weakest `amount` fraction of output neurons in each hidden
    layer independently, ranked by L2 row-norm.
    """
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


# ------------------------------------------------------------------
# Random neuron pruning
# ------------------------------------------------------------------

def prune_neurons_random(model: nn.Module, amount: float, seed: int = 0) -> None:
    """Remove a uniformly random `amount` fraction of hidden neurons."""
    rng     = torch.Generator()
    rng.manual_seed(seed)
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    hidden  = linears[:-1]

    total   = sum(m.out_features for m in hidden)
    n_prune = max(1, int(round(amount * total)))

    # Build a global random permutation of neuron indices and mark the first
    # n_prune as pruned.
    perm      = torch.randperm(total, generator=rng)
    prune_set = set(perm[:n_prune].tolist())

    offset = 0
    for layer in hidden:
        n_out = layer.out_features
        dev   = layer.weight.device
        mask  = torch.ones(n_out, device=dev)
        for local_i in range(n_out):
            if (offset + local_i) in prune_set:
                mask[local_i] = 0.0
        layer.weight.data *= mask.unsqueeze(1)
        if layer.bias is not None:
            layer.bias.data *= mask
        offset += n_out


# ------------------------------------------------------------------
# Activation-based neuron pruning
# (mean absolute activation over a calibration mini-batch)
# ------------------------------------------------------------------

def prune_neurons_activation(
    model: nn.Module,
    amount: float,
    X_calib,
    device,
    n_batches: int = 10,
    batch_sz: int  = 128,
) -> None:
    """Remove the `amount` fraction of hidden neurons with the smallest
    mean absolute activation over a calibration set.

    Uses forward hooks to capture activations after each hidden Linear layer.
    """
    if isinstance(X_calib, np.ndarray):
        X_calib = torch.from_numpy(X_calib).float()

    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    hidden  = linears[:-1]

    # Register forward hooks
    activation_sums = {id(m): torch.zeros(m.out_features) for m in hidden}
    activation_counts = {id(m): 0 for m in hidden}
    hooks = []

    def make_hook(layer_id):
        def hook(module, inp, out):
            # out: [batch, out_features]  (pre-nonlinearity, but sufficient for magnitude)
            activation_sums[layer_id]  += out.detach().abs().mean(dim=0).cpu()
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
            xb    = X_calib[idx_b].to(device)
            model(xb)

    for h in hooks:
        h.remove()
    if was_training:
        model.train()

    # Build global score tensor
    scores_per_layer = []
    for layer in hidden:
        cnt = max(1, activation_counts[id(layer)])
        scores_per_layer.append(activation_sums[id(layer)] / cnt)

    all_scores = torch.cat(scores_per_layer)
    n_prune    = max(1, int(round(amount * all_scores.numel())))
    prune_indices = torch.topk(all_scores, n_prune, largest=False).indices
    prune_set = set(prune_indices.tolist())

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


# ------------------------------------------------------------------
# Taylor-expansion neuron pruning
# (|mean gradient * weight| summed over each output row)
# ------------------------------------------------------------------

def prune_neurons_taylor(
    model: nn.Module,
    amount: float,
    X_calib,
    y_calib,
    device,
    n_batches: int = 10,
    batch_sz: int  = 128,
) -> None:
    """Remove the `amount` fraction of hidden neurons with the smallest
    first-order Taylor importance: Σ_j |w_ij · g_ij| summed over the row.
    """
    if isinstance(X_calib, np.ndarray):
        X_calib = torch.from_numpy(X_calib).float()
    if isinstance(y_calib, np.ndarray):
        y_calib = torch.from_numpy(y_calib).long()

    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    hidden  = linears[:-1]

    importance = {id(m): torch.zeros(m.out_features) for m in hidden}

    was_training = model.training
    model.train()
    n = X_calib.shape[0]
    criterion = nn.CrossEntropyLoss()

    for b in range(n_batches):
        idx_b = torch.randperm(n, generator=torch.Generator().manual_seed(b))[:batch_sz]
        xb    = X_calib[idx_b].to(device)
        yb    = y_calib[idx_b].to(device)
        model.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        with torch.no_grad():
            for layer in hidden:
                if layer.weight.grad is not None:
                    # Row-wise Taylor: sum |w_ij * grad_ij| across each output neuron
                    row_score = (layer.weight.data * layer.weight.grad).abs().sum(dim=1)
                    importance[id(layer)] += row_score.cpu()

    model.zero_grad()
    if not was_training:
        model.eval()

    for layer in hidden:
        importance[id(layer)] /= n_batches

    all_scores = torch.cat([importance[id(m)] for m in hidden])
    n_prune    = max(1, int(round(amount * all_scores.numel())))
    prune_indices = torch.topk(all_scores, n_prune, largest=False).indices
    prune_set = set(prune_indices.tolist())

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


# ------------------------------------------------------------------
# Helper: compute how many neurons to prune to reach a target count
# ------------------------------------------------------------------

def _compute_neuron_prune_amount(model: nn.Module, target_neurons: int) -> float:
    """Return the fraction of hidden neurons to remove so that at most
    `target_neurons` hidden neurons remain (clamped to [0, 0.99]).
    """
    current = _count_hidden_neurons(model)
    neurons_to_remove = current - target_neurons
    if neurons_to_remove <= 0:
        return 0.0
    amount = neurons_to_remove / current
    return min(amount, 0.99)


# ------------------------------------------------------------------
# Helper: trim to exact neuron count after threshold-tie overshoot
# ------------------------------------------------------------------

def _trim_to_exact_neuron_count(model: nn.Module, target: int) -> None:
    """If _count_hidden_neurons > target (due to threshold ties), zero out
    the excess lowest-scoring neurons by L1 row-norm and call
    physically_remove_zeroed_neurons again to make the removal permanent.
    Repeats until the count matches or no further reduction is possible.
    """
    for _ in range(20):   # safety cap on iterations
        current = _count_hidden_neurons(model)
        if current <= target:
            break
        excess = current - target

        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        hidden  = linears[:-1]

        # Collect (score, layer_index, neuron_index) for all live neurons
        candidates = []
        for li, layer in enumerate(hidden):
            row_norms = layer.weight.data.abs().sum(dim=1)
            for ni in range(layer.out_features):
                candidates.append((row_norms[ni].item(), li, ni))

        # Zero the `excess` lowest-scoring neurons
        candidates.sort(key=lambda x: x[0])
        to_zero = candidates[:excess]

        with torch.no_grad():
            for _, li, ni in to_zero:
                hidden[li].weight.data[ni] = 0.0
                if hidden[li].bias is not None:
                    hidden[li].bias.data[ni] = 0.0

        physically_remove_zeroed_neurons(model)


# ------------------------------------------------------------------
# Helper: physically remove zeroed-out neurons from Linear layers
# ------------------------------------------------------------------

def _replace_module(model: nn.Module, old: nn.Module, new: nn.Module) -> None:
    """Walk model and replace the first occurrence of `old` with `new`."""
    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if child is old:
                setattr(parent, child_name, new)
                return


def physically_remove_zeroed_neurons(model: nn.Module) -> None:
    """Rebuild hidden Linear layers so that all-zero output rows are
    permanently removed and the adjacent layer's input columns match.

    Processes one hidden layer at a time, re-fetching the live module
    list after each replacement so that stale references never cause
    dimension mismatches in subsequent iterations.
    """
    while True:
        # Re-fetch linears after every replacement so refs are always fresh
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        hidden  = linears[:-1]   # exclude the classifier head

        replaced_any = False
        for i, layer_out in enumerate(hidden):
            row_norms = layer_out.weight.data.abs().sum(dim=1)
            live_idx  = (row_norms > 0).nonzero(as_tuple=False).squeeze(1)

            if live_idx.numel() == layer_out.out_features:
                continue   # layer is already clean

            dev   = layer_out.weight.device
            dtype = layer_out.weight.dtype

            # Rebuild layer_out keeping only live rows
            new_out = nn.Linear(layer_out.in_features, live_idx.numel(),
                                bias=layer_out.bias is not None,
                                dtype=dtype, device=dev)
            with torch.no_grad():
                new_out.weight.copy_(layer_out.weight[live_idx])
                if layer_out.bias is not None:
                    new_out.bias.copy_(layer_out.bias[live_idx])
            _replace_module(model, layer_out, new_out)

            # Rebuild the next layer trimming its input columns to match
            # Re-fetch linears so layer_in_next is the current live object
            linears_now  = [m for m in model.modules() if isinstance(m, nn.Linear)]
            layer_in_next = linears_now[i + 1]   # index is still valid; only out was replaced

            new_in = nn.Linear(live_idx.numel(), layer_in_next.out_features,
                               bias=layer_in_next.bias is not None,
                               dtype=dtype, device=dev)
            with torch.no_grad():
                new_in.weight.copy_(layer_in_next.weight[:, live_idx])
                if layer_in_next.bias is not None:
                    new_in.bias.copy_(layer_in_next.bias)
            _replace_module(model, layer_in_next, new_in)

            replaced_any = True
            break   # restart the while-loop with fully fresh refs

        if not replaced_any:
            break   # all hidden layers are clean


# ==================================================================
# 1. Train / load
# ==================================================================

model, scaler, (X_train, X_test, y_train, y_test), timings, device = load_or_train(
    data_dir, model_path,
    seed=seed, test_size=test_size, batch_size=batch_size,
    lr=lr, epochs=epochs, weight_decay=weight_decay,
    threshold=threshold, force_train=force_train,
    patience=15,
    min_delta=1e-4,
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
record_stage("Baseline", net, noc, X_test, y_test, device, timings)
#noc.visualize_all(title="Baseline", metrics=make_metrics(stages))

# ==================================================================
# 3. Iterative unstructured L1 edge pruning + fine-tuning
#    (identical to the original edge file — provides the shared
#     pre-pruned checkpoint from which all comparison branches start)
# ==================================================================

original_edge_count = net.graph.number_of_edges()
target_removals     = round(original_edge_count * l1_target)

total_weights = sum(
    p.numel() for name, p in net.model.named_parameters()
    if "weight" in name and p.dim() == 2
)

for step in range(1, l1_steps + 1):
    print(f"L1 loop: {step}")
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
#noc.visualize_all(title="After L1 pruning", metrics=make_metrics(stages))

# ------------------------------------------------------------------
# Save L1 checkpoint — used later for all neuron-matched comparisons.
# We deepcopy the full model (not just state_dict) because
# remove_lonely_neurons has already structurally resized the layers.
				   
				 
# ------------------------------------------------------------------
l1_model_copy      = copy.deepcopy(net.model)
l1_neuron_count    = _count_hidden_neurons(net.model)
l1_edge_count      = net.graph.number_of_edges()
print(f"\n[Checkpoint] Saved L1 model — "
      f"{l1_neuron_count} hidden neurons, {l1_edge_count:,} edges remaining.")

# ==================================================================
# 4. Comm.Cost aware neuron pruning + fine-tune
# ==================================================================

net.prune_high_indegree_low_magnitude(
    low_mag_percentile=low_mag_percentile,
    degree_quantile=degree_quantile,
    min_in_degree=min_in_degree,
)
net.remove_lonely_neurons()

reapply_zero_masks(net)

fine_tune(net.model, X_train, y_train, device=device, lr=lr,
          epochs=zmft_epochs, seed=seed)

"""
#TBD
"""
net.commit_pruning()
net.resync_graph()  #added due to additional reapply_zero_masks(net) and fine_tune() step above
"""
#TBD
"""   

print("\n=== After Comm.Cost aware neuron pruning ===")
eval_nn(net.model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
noc = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
               title="After Comm.Cost aware pruning", visualize=False)
record_stage("After Comm.Cost pruning", net, noc, X_test, y_test, device, timings)
#noc.visualize_all(title="After Comm.Cost pruning", metrics=make_metrics(stages))

# Record neuron count achieved by Comm.Cost pruning — used to calibrate matched branches
commcost_neuron_count = _count_hidden_neurons(net.model)
commcost_edge_count   = net.graph.number_of_edges()
print(f"\n[Comparison] Comm.Cost pruning neuron count: {commcost_neuron_count} "
      f"(started from {l1_neuron_count} after L1 edge pruning)")

# ==================================================================
# 5. Heuristic bridge-link pruning  (applied on top of Comm.Cost result)
# ==================================================================

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
#noc_bridge.visualize_all(title="After Bridge-Link pruning", metrics=make_metrics(stages))

# ==================================================================
# Generic helper: run a neuron-matched comparison branch
# ==================================================================

def run_neuron_matched_branch(
    label: str,
    l1_model_copy: nn.Module,
    l1_neuron_count: int,
    commcost_neuron_count: int,
    prune_fn,
    prune_kwargs: dict,
    X_train, y_train, X_test, y_test,
    device, timings,
) -> None:
    """Reload the unstructured-L1 checkpoint, prune hidden neurons to reach
    `commcost_neuron_count` using `prune_fn`, fine-tune identically to
    the Comm.Cost branch, then record the stage.

    Matching is done on NEURON count so that every comparison branch
    removes the same number of neurons as Comm.Cost did.
					 
    """
    print("\n" + "=" * 70)
    print(f"  STEP: {label}")
    print(f"  Target: match Comm.Cost neuron count ({commcost_neuron_count} neurons)")
    print(f"  Starting from L1 checkpoint ({l1_neuron_count} neurons)")
    print("=" * 70)

    net_branch = PrunableNetwork(copy.deepcopy(l1_model_copy), eps=0.0)

    branch_neuron_count = _count_hidden_neurons(net_branch.model)
    assert branch_neuron_count == l1_neuron_count, (
        f"Checkpoint neuron count mismatch: "
        f"expected {l1_neuron_count}, got {branch_neuron_count}"
    )

    amount_needed = _compute_neuron_prune_amount(net_branch.model, commcost_neuron_count)

    print(f"\n[{label}] Hidden neurons in checkpoint : {branch_neuron_count}")
    print(f"[{label}] Target neuron count          : {commcost_neuron_count}")
    print(f"[{label}] Fraction to prune            : {amount_needed:.4f}")

    if amount_needed <= 0:
        print(f"[{label}] WARNING: Comm.Cost neuron count >= checkpoint count — skipping.")
        return
	
    # Apply the structured neuron pruning criterion (zeros rows).
    # We intentionally prune to a count ABOVE commcost_neuron_count here so that
    # commit_pruning()'s internal lonely-neuron cleanup (which removes neurons whose
    # all incoming edges were deleted) can bring the count down further without
    # overshooting the target.  _trim_to_exact_neuron_count after commit_pruning
    # then corrects any remaining overshoot to land exactly on the target.
    intermediate_target = commcost_neuron_count + 50   # generous buffer > max cascade loss
    amount_intermediate = _compute_neuron_prune_amount(net_branch.model, intermediate_target)
    prune_fn(net_branch.model, amount_intermediate, **prune_kwargs)
    # Physically shrink the Linear layers so that _count_hidden_neurons
    # reflects the true live neuron count and PrunableNetwork graph is correct.
    physically_remove_zeroed_neurons(net_branch.model)
    # commit_pruning rebuilds the graph and may remove additional lonely neurons
    # (e.g. neurons in layer N+1 whose every incoming connection from layer N was pruned).
    # This is intentional — it brings us closer to or past the exact target.
    net_branch.commit_pruning()
    # Trim any overshoot (current > target) down to the exact target.
    # If commit_pruning's cascade already landed us at or below target, this is a no-op.
    after_commit = _count_hidden_neurons(net_branch.model)
    if after_commit > commcost_neuron_count:
        _trim_to_exact_neuron_count(net_branch.model, commcost_neuron_count)
        net_branch.commit_pruning()   # re-commit after trim
    elif after_commit < commcost_neuron_count:
        print(f"[{label}] WARNING: after commit got {after_commit} neurons "
              f"(target {commcost_neuron_count}); cascade removed too many — cannot restore. "
              f"Consider increasing intermediate_target buffer.")

    matched_neurons = _count_hidden_neurons(net_branch.model)
    print(f"[{label}] Neurons after pruning : {matched_neurons} "
          f"(target was {commcost_neuron_count}, diff={matched_neurons - commcost_neuron_count:+d})")

    reapply_zero_masks(net_branch)
    fine_tune(net_branch.model, X_train, y_train, device=device, lr=lr,
              epochs=zmft_epochs, seed=seed)

    print(f"\n=== {label} ===")
    eval_nn(net_branch.model, X_test=X_test, y_test=y_test, device=device,
            timings=timings, evaluate_fn=evaluate_model)
    noc_branch = eval_noc(net_branch.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                          title=label, visualize=False)
    record_stage(label, net_branch, noc_branch, X_test, y_test, device, timings)
    #noc_branch.visualize_all(title=label, metrics=make_metrics(stages))

# ==================================================================
# 6. Struct-L1-matched comparison (neuron count matched to Comm.Cost)
# ==================================================================

print("\n" + "=" * 70)
print("  STEP 6: Struct-L1-matched comparison")
print(f"  Target: match Comm.Cost neuron count ({commcost_neuron_count} neurons)")
print(f"  Starting from L1 checkpoint ({l1_neuron_count} neurons)")
print("=" * 70)


net_struct_l1 = PrunableNetwork(copy.deepcopy(l1_model_copy), eps=0.0)

assert _count_hidden_neurons(net_struct_l1.model) == l1_neuron_count, (
    f"Checkpoint neuron count mismatch: "
    f"expected {l1_neuron_count}, got {_count_hidden_neurons(net_struct_l1.model)}"
)

amount_needed_l1 = _compute_neuron_prune_amount(net_struct_l1.model, commcost_neuron_count)

print(f"\n[Struct-L1-matched] Hidden neurons in checkpoint : {l1_neuron_count}")
print(f"[Struct-L1-matched] Target neuron count          : {commcost_neuron_count}")
print(f"[Struct-L1-matched] Fraction to prune            : {amount_needed_l1:.4f}")

if amount_needed_l1 <= 0:
    print("[Struct-L1-matched] WARNING: Comm.Cost neuron count >= checkpoint count — skipping.")
elif 0 < amount_needed_l1 < 1.0:
    # Prune to an intermediate target above commcost so that commit_pruning's
    # lonely-neuron cascade doesn't push us below the final target.
    intermediate_target_l1 = commcost_neuron_count + 50
    amount_intermediate_l1 = _compute_neuron_prune_amount(net_struct_l1.model, intermediate_target_l1)
    prune_neurons_l1_structured(net_struct_l1.model, amount_intermediate_l1)
    physically_remove_zeroed_neurons(net_struct_l1.model)
    net_struct_l1.commit_pruning()
    after_commit_l1 = _count_hidden_neurons(net_struct_l1.model)
    if after_commit_l1 > commcost_neuron_count:
        _trim_to_exact_neuron_count(net_struct_l1.model, commcost_neuron_count)
        net_struct_l1.commit_pruning()
    elif after_commit_l1 < commcost_neuron_count:
        print(f"[Struct-L1-matched] WARNING: after commit got {after_commit_l1} neurons "
              f"(target {commcost_neuron_count}); cascade removed too many — cannot restore.")

matched_neuron_count_l1 = _count_hidden_neurons(net_struct_l1.model)
print(f"[Struct-L1-matched] Neurons after pruning : {matched_neuron_count_l1} "
      f"(target was {commcost_neuron_count}, diff={matched_neuron_count_l1 - commcost_neuron_count:+d})")

reapply_zero_masks(net_struct_l1)
fine_tune(net_struct_l1.model, X_train, y_train, device=device, lr=lr,
          epochs=zmft_epochs, seed=seed)

print("\n=== After Struct-L1-matched neuron pruning (same fine-tune as Comm.Cost) ===")
eval_nn(net_struct_l1.model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
noc_struct_l1m = eval_noc(net_struct_l1.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                           title="After Struct-L1-matched", visualize=False)
record_stage("After Struct-L1-matched", net_struct_l1, noc_struct_l1m, X_test, y_test, device, timings)
#noc_struct_l1m.visualize_all(title="After Struct-L1-matched", metrics=make_metrics(stages))

# ==================================================================
# 7. Struct-L2-matched comparison
# ==================================================================

run_neuron_matched_branch(
    label="After Struct-L2-matched",
    l1_model_copy=l1_model_copy,
    l1_neuron_count=l1_neuron_count,
    commcost_neuron_count=commcost_neuron_count,
    prune_fn=prune_neurons_l2_structured,
    prune_kwargs={},
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
)

# ==================================================================
# 8. Random neuron-matched comparison
# ==================================================================

run_neuron_matched_branch(
    label="After Random-matched",
    l1_model_copy=l1_model_copy,
    l1_neuron_count=l1_neuron_count,
    commcost_neuron_count=commcost_neuron_count,
    prune_fn=prune_neurons_random,
    prune_kwargs={"seed": seed},
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
)

# ==================================================================
# 9. Activation neuron-matched comparison
# ==================================================================

run_neuron_matched_branch(
    label="After Activation-matched",
    l1_model_copy=l1_model_copy,
    l1_neuron_count=l1_neuron_count,
    commcost_neuron_count=commcost_neuron_count,
    prune_fn=prune_neurons_activation,
    prune_kwargs={
        "X_calib":   X_train,
		
        "device":    device,
        "n_batches": 10,
        "batch_sz":  128,
    },
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    device=device,   timings=timings,
)

# ==================================================================
# 10. Taylor neuron-matched comparison
# ==================================================================

run_neuron_matched_branch(
    label="After Taylor-matched",
    l1_model_copy=l1_model_copy,
    l1_neuron_count=l1_neuron_count,
    commcost_neuron_count=commcost_neuron_count,
    prune_fn=prune_neurons_taylor,
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

# ==================================================================
# 11. Final comparison summary
# ==================================================================

print_summary_table()
