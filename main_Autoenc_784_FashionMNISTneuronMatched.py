import math
import copy
import types
import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from DNN_Autoenc_784_FashionMNIST import load_or_train, fine_tune, evaluate_model
from prunable_network import PrunableNetwork
from eval_helper import *
from visualize import visualize_model
from mapping_helpers import build_wctg_structural, compute_structural_cost

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

data_dir: str       = "./data"
model_path: str     = "DNN_Autoenc_784_FashionMNIST.pth"
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

alpha: float     = 1.0   # reconstruction loss weight
beta: float      = 1.0   # classification loss weight

zmft_epochs: int = 5     # fine-tune epochs after neuron pruning

P: int         = 36
N: int         = 36
mesh_rows: int = int(math.sqrt(P))
mesh_cols: int = int(math.sqrt(P))

# ------------------------------------------------------------------
# Encoder+Classifier wrapper — what PrunableNetwork sees
# ------------------------------------------------------------------

class EncoderClassifier(nn.Module):
    def __init__(self, full_model: nn.Module):
        super().__init__()
        self.fc_enc1 = full_model.fc_enc1
        self.relu1   = full_model.relu_e1
        self.fc_enc2 = full_model.fc_enc2
        self.relu2   = full_model.relu_e2
        self.fc_cls  = full_model.fc_cls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc_enc1(x))
        x = self.relu2(self.fc_enc2(x))
        return self.fc_cls(x)


# ------------------------------------------------------------------
# Bottleneck-aware prune_neurons patch
# ------------------------------------------------------------------

def _autoenc_prune_neurons(self, nodes):
    from graph import construct_nn_graph
    node_set = set(nodes)
    if not node_set:
        return self
    BOTTLENECK_LAYER = 2
    bottleneck_nodes = {n for n in node_set if n[0] == BOTTLENECK_LAYER}
    other_nodes      = node_set - bottleneck_nodes
    if bottleneck_nodes:
        bottleneck_indices = sorted({idx for (_, idx) in bottleneck_nodes})
        bottleneck_size    = self._model.fc_enc2.out_features
        keep_mask = torch.ones(bottleneck_size, dtype=torch.bool,
                               device=self._model.fc_enc2.weight.device)
        for idx in bottleneck_indices:
            keep_mask[idx] = False
        kept = int(keep_mask.sum().item())
        if kept > 0:
            with torch.no_grad():
                dev = self._model.fc_enc2.weight.device
                new_enc2 = nn.Linear(self._model.fc_enc2.in_features, kept,
                                     bias=self._model.fc_enc2.bias is not None).to(dev)
                new_enc2.weight.copy_(self._model.fc_enc2.weight[keep_mask])
                if self._model.fc_enc2.bias is not None:
                    new_enc2.bias.copy_(self._model.fc_enc2.bias[keep_mask])
                self._model.fc_enc2 = new_enc2
                full_model.fc_enc2  = new_enc2
                new_cls = nn.Linear(kept, self._model.fc_cls.out_features,
                                    bias=self._model.fc_cls.bias is not None).to(dev)
                new_cls.weight.copy_(self._model.fc_cls.weight[:, keep_mask])
                if self._model.fc_cls.bias is not None:
                    new_cls.bias.copy_(self._model.fc_cls.bias)
                self._model.fc_cls = new_cls
                full_model.fc_cls  = new_cls
                new_dec1 = nn.Linear(kept, full_model.fc_dec1.out_features,
                                     bias=full_model.fc_dec1.bias is not None).to(dev)
                new_dec1.weight.copy_(full_model.fc_dec1.weight[:, keep_mask])
                if full_model.fc_dec1.bias is not None:
                    new_dec1.bias.copy_(full_model.fc_dec1.bias)
                full_model.fc_dec1 = new_dec1
    if other_nodes:
        from pruning import prune_neurons as _prune_neurons
        self._model = _prune_neurons(self._model, src=sorted(other_nodes),
                                     linear_names=self._graph.graph["linear_names"])
    full_model.fc_enc1 = self._model.fc_enc1
    full_model.fc_enc2 = self._model.fc_enc2
    full_model.fc_cls  = self._model.fc_cls
    for _, m in self._model.named_modules():
        if isinstance(m, nn.Linear) and hasattr(m, "weight_orig"):
            prune.remove(m, "weight")
    self._graph = construct_nn_graph(self._model, eps=self._eps)
    return self


def _autoenc_remove_lonely_neurons(self):
    from graph import construct_nn_graph, filter_lonely_neurons
    layers     = self._graph.graph["layers"]
    last_layer = len(layers) - 1
    lonely     = filter_lonely_neurons(self._graph)
    prunable   = {node for node in lonely if 0 < node[0] < last_layer}
    if prunable:
        self.prune_neurons(sorted(prunable))
    return lonely


def install_autoenc_patch(net: PrunableNetwork) -> None:
    net.prune_neurons         = types.MethodType(_autoenc_prune_neurons,        net)
    net.remove_lonely_neurons = types.MethodType(_autoenc_remove_lonely_neurons, net)


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
    """Total FLOPs = 2 x sum(in_features x out_features) across all Linear layers."""
    total_macs = 0
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            total_macs += mod.in_features * mod.out_features
    return 2 * total_macs


def record_stage(name, net, noc, X_test, y_test, device, timings):
    report  = evaluate_model(full_model, X_test, y_test, device=device, timings=timings)
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
    print("  Model: Autoencoder+Classifier [784->128->64->128->784 + 64->10] | Dataset: FashionMNIST")
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
# All helpers operate on the EncoderClassifier wrapper's Linear layers
# (fc_enc1, fc_enc2, fc_cls).  The bottleneck layer (fc_enc2) requires
# special handling: removing its output neurons also shrinks fc_dec1's
# input columns in full_model (the decoder is not in the wrapper).
#
# Call sequence after zeroing rows:
#   physically_remove_zeroed_neurons_autoenc(wrapper_model, cur_full_model)
#   _trim_to_exact_neuron_count_autoenc(wrapper_model, cur_full_model, target)
#   net.commit_pruning()
#   # remove_lonely_neurons is SKIPPED — layers already physically resized
# ==================================================================

def _count_hidden_neurons_wrapper(wrapper: nn.Module) -> int:
    """Count live output neurons in fc_enc1 and fc_enc2 (hidden layers of the wrapper).
    fc_cls is the output layer and is excluded.
    """
    linears = [m for m in wrapper.modules() if isinstance(m, nn.Linear)]
    # linears = [fc_enc1, fc_enc2, fc_cls]  — exclude last (fc_cls)
    return sum(m.out_features for m in linears[:-1])


# ------------------------------------------------------------------
# Structured L1 neuron pruning (per-layer, row-wise L1 norm)
# ------------------------------------------------------------------

def prune_neurons_l1_structured(wrapper: nn.Module, amount: float) -> None:
    """Zero the weakest `amount` fraction of output neurons in each hidden
    layer independently, ranked by L1 row-norm.

    Per-layer pruning (rather than a single global threshold) prevents the
    smaller bottleneck layer (fc_enc2, 64 neurons) from being disproportionately
    gutted relative to the wider fc_enc1 (128 neurons), which would collapse
    accuracy to chance level.
    """
    linears = [m for m in wrapper.modules() if isinstance(m, nn.Linear)]
    hidden  = linears[:-1]   # fc_enc1, fc_enc2

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
# Structured L2 neuron pruning (per-layer, row-wise L2 norm)
# ------------------------------------------------------------------

def prune_neurons_l2_structured(wrapper: nn.Module, amount: float) -> None:
    """Zero the weakest `amount` fraction of output neurons in each hidden
    layer independently, ranked by L2 row-norm.

    Per-layer pruning prevents the bottleneck layer from being collapsed
    by a global threshold skewed by fc_enc1's larger magnitude scores.
    """
    linears = [m for m in wrapper.modules() if isinstance(m, nn.Linear)]
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

def prune_neurons_random(wrapper: nn.Module, amount: float, seed: int = 0) -> None:
    """Uniformly randomly zero `amount` fraction of hidden neurons."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    linears = [m for m in wrapper.modules() if isinstance(m, nn.Linear)]
    hidden  = linears[:-1]

    total     = sum(m.out_features for m in hidden)
    n_prune   = max(1, int(round(amount * total)))
    perm      = torch.randperm(total, generator=rng)
    prune_set = set(perm[:n_prune].tolist())

    idx = 0
    for layer in hidden:
        n   = layer.out_features
        dev = layer.weight.device
        keep = torch.ones(n, device=dev)
        for local_i in range(n):
            if (idx + local_i) in prune_set:
                keep[local_i] = 0.0
        layer.weight.data *= keep.unsqueeze(1)
        if layer.bias is not None:
            layer.bias.data *= keep
        idx += n


# ------------------------------------------------------------------
# Activation-based neuron pruning
# ------------------------------------------------------------------

def prune_neurons_activation(
    wrapper: nn.Module,
    amount: float,
    X_calib,
    device,
    n_batches: int = 10,
    batch_sz: int  = 128,
) -> None:
    """Zero the `amount` fraction of hidden neurons with the smallest
    mean absolute activation over a calibration set.
    """
    if isinstance(X_calib, np.ndarray):
        X_calib = torch.from_numpy(X_calib).float()

    linears = [m for m in wrapper.modules() if isinstance(m, nn.Linear)]
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

    was_training = wrapper.training
    wrapper.eval()
    n = X_calib.shape[0]
    with torch.no_grad():
        for b in range(n_batches):
            idx_b = torch.randperm(n, generator=torch.Generator().manual_seed(b))[:batch_sz]
            wrapper(X_calib[idx_b].to(device))

    for h in hooks:
        h.remove()
    if was_training:
        wrapper.train()

    scores_per_layer = [
        activation_sums[id(m)] / max(1, activation_counts[id(m)]) for m in hidden
    ]
    all_scores = torch.cat(scores_per_layer)
    n_prune    = max(1, int(round(amount * all_scores.numel())))
    threshold  = torch.topk(all_scores, n_prune, largest=False).values.max()

    for layer, scores in zip(hidden, scores_per_layer):
        mask = (scores > threshold).float().to(layer.weight.device)
        layer.weight.data *= mask.unsqueeze(1)
        if layer.bias is not None:
            layer.bias.data *= mask


# ------------------------------------------------------------------
# Taylor-expansion neuron pruning
# ------------------------------------------------------------------

def prune_neurons_taylor(
    wrapper: nn.Module,
    amount: float,
    X_calib,
    y_calib,
    device,
    n_batches: int = 10,
    batch_sz: int  = 128,
) -> None:
    """Zero the `amount` fraction of hidden neurons with the smallest
    first-order Taylor importance (row-wise Σ|w·g|).
    """
    if isinstance(X_calib, np.ndarray):
        X_calib = torch.from_numpy(X_calib).float()
    if isinstance(y_calib, np.ndarray):
        y_calib = torch.from_numpy(y_calib).long()

    linears = [m for m in wrapper.modules() if isinstance(m, nn.Linear)]
    hidden  = linears[:-1]

    importance = {id(m): torch.zeros(m.out_features) for m in hidden}
    criterion  = nn.CrossEntropyLoss()
    was_training = wrapper.training
    wrapper.train()
    n = X_calib.shape[0]

    for b in range(n_batches):
        idx_b = torch.randperm(n, generator=torch.Generator().manual_seed(b))[:batch_sz]
        xb    = X_calib[idx_b].to(device)
        yb    = y_calib[idx_b].to(device)
        wrapper.zero_grad()
        loss = criterion(wrapper(xb), yb)
        loss.backward()
        with torch.no_grad():
            for layer in hidden:
                if layer.weight.grad is not None:
                    importance[id(layer)] += (layer.weight.data * layer.weight.grad).abs().sum(dim=1).cpu()

    wrapper.zero_grad()
    if not was_training:
        wrapper.eval()

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


# ==================================================================
# Physical neuron removal helpers (autoencoder-aware)
# ==================================================================
# The autoencoder has a decoder (fc_dec1, fc_dec2) in full_model that
# is NOT part of the EncoderClassifier wrapper.  When bottleneck neurons
# (fc_enc2 outputs) are pruned, fc_dec1's input columns must also be
# trimmed to stay consistent.  physically_remove_zeroed_neurons_autoenc
# handles this by checking which layer in the wrapper is the bottleneck
# and propagating the column trim to cur_full_model.fc_dec1.
# ==================================================================

def _replace_module(model: nn.Module, old: nn.Module, new: nn.Module) -> None:
    """Walk model and replace the first occurrence of `old` with `new`."""
    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if child is old:
                setattr(parent, child_name, new)
                return


def physically_remove_zeroed_neurons_autoenc(
    wrapper: nn.Module,
    cur_full_model: nn.Module,
) -> None:
    """Rebuild hidden Linear layers in `wrapper` so that all-zero output
    rows are permanently removed.

    For the bottleneck layer (fc_enc2), also trims fc_dec1's input
    columns in `cur_full_model` to keep the decoder consistent.

    Processes one layer at a time, re-fetching refs each iteration to
    avoid stale-pointer dimension mismatches.
    """
    while True:
        linears = [m for m in wrapper.modules() if isinstance(m, nn.Linear)]
        hidden  = linears[:-1]   # [fc_enc1, fc_enc2]

        replaced_any = False
        for i, layer_out in enumerate(hidden):
            row_norms = layer_out.weight.data.abs().sum(dim=1)
            live_idx  = (row_norms > 0).nonzero(as_tuple=False).squeeze(1)

            if live_idx.numel() == layer_out.out_features:
                continue

            dev   = layer_out.weight.device
            dtype = layer_out.weight.dtype
            is_bottleneck = (i == len(hidden) - 1)   # fc_enc2 is last hidden layer

            # Rebuild layer_out (keep live rows only)
            new_out = nn.Linear(layer_out.in_features, live_idx.numel(),
                                bias=layer_out.bias is not None, dtype=dtype, device=dev)
            with torch.no_grad():
                new_out.weight.copy_(layer_out.weight[live_idx])
                if layer_out.bias is not None:
                    new_out.bias.copy_(layer_out.bias[live_idx])
            _replace_module(wrapper, layer_out, new_out)

            # Rebuild the next layer in the wrapper (trim input columns)
            linears_now   = [m for m in wrapper.modules() if isinstance(m, nn.Linear)]
            layer_in_next = linears_now[i + 1]
            new_in = nn.Linear(live_idx.numel(), layer_in_next.out_features,
                               bias=layer_in_next.bias is not None, dtype=dtype, device=dev)
            with torch.no_grad():
                new_in.weight.copy_(layer_in_next.weight[:, live_idx])
                if layer_in_next.bias is not None:
                    new_in.bias.copy_(layer_in_next.bias)
            _replace_module(wrapper, layer_in_next, new_in)

            # If this was the bottleneck layer, also trim fc_dec1 in full_model
            if is_bottleneck and hasattr(cur_full_model, "fc_dec1"):
                old_dec1  = cur_full_model.fc_dec1
                new_dec1  = nn.Linear(live_idx.numel(), old_dec1.out_features,
                                      bias=old_dec1.bias is not None, dtype=dtype, device=dev)
                with torch.no_grad():
                    new_dec1.weight.copy_(old_dec1.weight[:, live_idx])
                    if old_dec1.bias is not None:
                        new_dec1.bias.copy_(old_dec1.bias)
                cur_full_model.fc_dec1 = new_dec1

            # Sync full_model's encoder layers to match the wrapper
            if hasattr(cur_full_model, "fc_enc1"):
                linears_w = [m for m in wrapper.modules() if isinstance(m, nn.Linear)]
                cur_full_model.fc_enc1 = linears_w[0]
                cur_full_model.fc_enc2 = linears_w[1]
                cur_full_model.fc_cls  = linears_w[2]

            replaced_any = True
            break   # restart with fresh refs

        if not replaced_any:
            break


def _trim_to_exact_neuron_count_autoenc(
    wrapper: nn.Module,
    cur_full_model: nn.Module,
    target: int,
) -> None:
    """Trim any overshoot (due to threshold ties) down to exactly `target`
    hidden neurons by zeroing the excess lowest-L1-scored neurons and
    calling physically_remove_zeroed_neurons_autoenc again.
    """
    for _ in range(20):
        current = _count_hidden_neurons_wrapper(wrapper)
        if current <= target:
            break
        excess = current - target

        linears = [m for m in wrapper.modules() if isinstance(m, nn.Linear)]
        hidden  = linears[:-1]

        candidates = []
        for li, layer in enumerate(hidden):
            row_norms = layer.weight.data.abs().sum(dim=1)
            for ni in range(layer.out_features):
                candidates.append((row_norms[ni].item(), li, ni))

        candidates.sort(key=lambda x: x[0])
        with torch.no_grad():
            for _, li, ni in candidates[:excess]:
                hidden[li].weight.data[ni] = 0.0
                if hidden[li].bias is not None:
                    hidden[li].bias.data[ni] = 0.0

        physically_remove_zeroed_neurons_autoenc(wrapper, cur_full_model)


def _compute_neuron_prune_amount(wrapper: nn.Module, target_neurons: int) -> float:
    """Fraction of hidden neurons to remove to reach `target_neurons`."""
    current = _count_hidden_neurons_wrapper(wrapper)
    neurons_to_remove = current - target_neurons
    if neurons_to_remove <= 0:
        return 0.0
    return min(neurons_to_remove / current, 0.99)


# ==================================================================
# Generic helper: run a neuron-matched comparison branch
# ==================================================================

def run_neuron_matched_branch(
    label: str,
    l1_full_model_copy_arg: nn.Module,
    l1_neuron_count: int,
    commcost_neuron_count: int,
    prune_fn,
    prune_kwargs: dict,
    X_train, y_train, X_test, y_test,
    device, timings,
) -> None:
    """Reload the unstructured-L1 checkpoint, apply a structured neuron
    pruning criterion to match `commcost_neuron_count`, fine-tune with
    the same settings as the Comm.Cost branch, and record the stage.
    """
    global full_model

    print("\n" + "=" * 70)
    print(f"  STEP: {label}")
    print(f"  Target: match Comm.Cost neuron count ({commcost_neuron_count} neurons)")
    print(f"  Starting from L1 checkpoint ({l1_neuron_count} neurons)")
    print("=" * 70)

    cur_full_model = copy.deepcopy(l1_full_model_copy_arg)
    enc_cls_branch = EncoderClassifier(copy.deepcopy(l1_full_model_copy_arg)).to(device)
    net_branch     = PrunableNetwork(enc_cls_branch, eps=0.0)
    install_autoenc_patch(net_branch)

    branch_neuron_count = _count_hidden_neurons_wrapper(net_branch.model)
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

    prune_fn(net_branch.model, amount_needed, **prune_kwargs)
    physically_remove_zeroed_neurons_autoenc(net_branch.model, cur_full_model)
    _trim_to_exact_neuron_count_autoenc(net_branch.model, cur_full_model, commcost_neuron_count)
    net_branch.commit_pruning()
    # remove_lonely_neurons skipped — layers already physically resized

    matched_neurons = _count_hidden_neurons_wrapper(net_branch.model)
    print(f"[{label}] Neurons after pruning : {matched_neurons} "
          f"(target was {commcost_neuron_count}, diff={matched_neurons - commcost_neuron_count:+d})")

    reapply_zero_masks(net_branch)

    # Use cur_full_model as the active full_model for fine-tune and eval
    full_model = cur_full_model
    fine_tune(full_model, X_train, y_train, device=device, lr=lr,
              epochs=zmft_epochs, alpha=alpha, beta=beta, seed=seed)

    print(f"\n=== {label} ===")
    eval_nn(full_model, X_test=X_test, y_test=y_test, device=device,
            timings=timings, evaluate_fn=evaluate_model)
    noc_branch = eval_noc(net_branch.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                          title=label, visualize=False)
    record_stage(label, net_branch, noc_branch, X_test, y_test, device, timings)
    noc_branch.visualize_all(title=label, metrics=make_metrics(stages))


# ==================================================================
# 1. Train / load
# ==================================================================

full_model, scaler, (X_train, X_test, y_train, y_test), timings, device = load_or_train(
    data_dir, model_path,
    seed=seed, test_size=test_size, batch_size=batch_size,
    lr=lr, epochs=epochs, weight_decay=weight_decay,
    threshold=threshold, force_train=force_train,
    alpha=alpha, beta=beta,
    patience=15,
    min_delta=1e-4,
)

enc_cls_wrapper = EncoderClassifier(full_model).to(device)
net = PrunableNetwork(enc_cls_wrapper, eps=0.0)
install_autoenc_patch(net)

# ==================================================================
# 2. Baseline evaluation
# ==================================================================

print("\n=== Initial model ===")
eval_nn(full_model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
noc = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
               title="Baseline", visualize=False)
record_stage("Baseline", net, noc, X_test, y_test, device, timings)
noc.visualize_all(title="Baseline", metrics=make_metrics(stages))

# ==================================================================
# 3. Iterative unstructured L1 edge pruning + fine-tuning
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
    fine_tune(full_model, X_train, y_train, device=device, lr=lr,
              epochs=ft_epochs, alpha=alpha, beta=beta, seed=seed + step)

    removed = original_edge_count - net.graph.number_of_edges()
    print(f"  [Step {step}/{l1_steps}] Edges removed: {removed}/{original_edge_count} "
          f"({removed / original_edge_count:.1%} of original)")

net.commit_pruning()
net.remove_lonely_neurons()

print(f"\n=== After iterative L1 pruning ({l1_target:.0%} of original links removed) ===")
eval_nn(full_model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
noc = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
               title="After L1 pruning", visualize=False)
record_stage("After L1 25% pre pruning", net, noc, X_test, y_test, device, timings)
noc.visualize_all(title="After L1 pruning", metrics=make_metrics(stages))

# ------------------------------------------------------------------
# Save L1 checkpoint — used later for all neuron-matched branches.
# Deepcopy both the wrapper model and the full model (including decoder)
# because remove_lonely_neurons may have already resized layers.
# ------------------------------------------------------------------
l1_model_copy      = copy.deepcopy(net.model)    # EncoderClassifier wrapper
l1_full_model_copy = copy.deepcopy(full_model)   # full autoencoder+classifier
l1_neuron_count    = _count_hidden_neurons_wrapper(net.model)
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

fine_tune(full_model, X_train, y_train, device=device, lr=lr,
          epochs=zmft_epochs, alpha=alpha, beta=beta, seed=seed)
"""
#TBD
"""
net.commit_pruning()
net.resync_graph()  #added due to additional reapply_zero_masks(net) and fine_tune() step above
"""
#TBD
"""
print("\n=== After Comm.Cost aware neuron pruning ===")
eval_nn(full_model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
noc = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
               title="After Comm.Cost aware pruning", visualize=False)
record_stage("After Comm.Cost pruning", net, noc, X_test, y_test, device, timings)
noc.visualize_all(title="After Comm.Cost pruning", metrics=make_metrics(stages))

# Record neuron count achieved by Comm.Cost pruning — target for all matched branches
commcost_neuron_count = _count_hidden_neurons_wrapper(net.model)
commcost_edge_count   = net.graph.number_of_edges()
print(f"\n[Comparison] Comm.Cost pruning neuron count: {commcost_neuron_count} "
      f"(started from {l1_neuron_count} after L1 edge pruning)")

# ==================================================================
# 5. Heuristic bridge-link pruning
# ==================================================================

net.prune_links_heuristic(
    groups=noc.groups,
    amount=bridge_amount,
    debug=False,
)
net.commit_pruning()

#Fine Tune Similar to Comm. Cost Fine Tuning
"""
reapply_zero_masks(net)
fine_tune(full_model, X_train, y_train, device=device, lr=lr,
          epochs=zmft_epochs, seed=seed)
net.commit_pruning()
net.resync_graph() 
"""

print("\n=== After heuristic bridge-link pruning ===")
eval_nn(full_model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
noc_bridge = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                      title="After Bridge-Link pruning", visualize=False)
record_stage("After Bridge-Link pruning", net, noc_bridge, X_test, y_test, device, timings)
noc_bridge.visualize_all(title="After Bridge-Link pruning", metrics=make_metrics(stages))

# ==================================================================
# 6. Struct-L1-matched comparison (neuron count matched to Comm.Cost)
# ==================================================================

print("\n" + "=" * 70)
print("  STEP 6: Struct-L1-matched comparison")
print(f"  Target: match Comm.Cost neuron count ({commcost_neuron_count} neurons)")
print(f"  Starting from L1 checkpoint ({l1_neuron_count} neurons)")
print("=" * 70)

full_model_sl1 = copy.deepcopy(l1_full_model_copy)
enc_cls_sl1    = EncoderClassifier(copy.deepcopy(l1_full_model_copy)).to(device)
net_struct_l1  = PrunableNetwork(enc_cls_sl1, eps=0.0)
install_autoenc_patch(net_struct_l1)

assert _count_hidden_neurons_wrapper(net_struct_l1.model) == l1_neuron_count, (
    f"Checkpoint neuron count mismatch: "
    f"expected {l1_neuron_count}, got {_count_hidden_neurons_wrapper(net_struct_l1.model)}"
)

amount_needed_l1 = _compute_neuron_prune_amount(net_struct_l1.model, commcost_neuron_count)

print(f"\n[Struct-L1-matched] Hidden neurons in checkpoint : {l1_neuron_count}")
print(f"[Struct-L1-matched] Target neuron count          : {commcost_neuron_count}")
print(f"[Struct-L1-matched] Fraction to prune            : {amount_needed_l1:.4f}")

if amount_needed_l1 <= 0:
    print("[Struct-L1-matched] WARNING: Comm.Cost neuron count >= checkpoint count — skipping.")
elif 0 < amount_needed_l1 < 1.0:
    prune_neurons_l1_structured(net_struct_l1.model, amount_needed_l1)
    physically_remove_zeroed_neurons_autoenc(net_struct_l1.model, full_model_sl1)
    _trim_to_exact_neuron_count_autoenc(net_struct_l1.model, full_model_sl1, commcost_neuron_count)
    net_struct_l1.commit_pruning()
    # remove_lonely_neurons skipped — layers already physically resized

matched_neuron_count_l1 = _count_hidden_neurons_wrapper(net_struct_l1.model)
print(f"[Struct-L1-matched] Neurons after pruning : {matched_neuron_count_l1} "
      f"(target was {commcost_neuron_count}, diff={matched_neuron_count_l1 - commcost_neuron_count:+d})")

reapply_zero_masks(net_struct_l1)
full_model = full_model_sl1
fine_tune(full_model, X_train, y_train, device=device, lr=lr,
          epochs=zmft_epochs, alpha=alpha, beta=beta, seed=seed)

print("\n=== After Struct-L1-matched neuron pruning (same fine-tune as Comm.Cost) ===")
eval_nn(full_model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
noc_sl1m = eval_noc(net_struct_l1.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                    title="After Struct-L1-matched", visualize=False)
record_stage("After Struct-L1-matched", net_struct_l1, noc_sl1m, X_test, y_test, device, timings)
noc_sl1m.visualize_all(title="After Struct-L1-matched", metrics=make_metrics(stages))

# ==================================================================
# 7. Struct-L2-matched comparison
# ==================================================================

run_neuron_matched_branch(
    label="After Struct-L2-matched",
    l1_full_model_copy_arg=l1_full_model_copy,
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
    l1_full_model_copy_arg=l1_full_model_copy,
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
    l1_full_model_copy_arg=l1_full_model_copy,
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
    l1_full_model_copy_arg=l1_full_model_copy,
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
