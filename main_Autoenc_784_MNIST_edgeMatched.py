import math
import copy

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from DNN_Autoenc_784 import load_or_train, fine_tune, evaluate_model
from prunable_network import PrunableNetwork
from eval_helper import *
from visualize import visualize_model
from mapping_helpers import build_wctg_structural, compute_structural_cost

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

data_dir: str       = "./data"
model_path: str     = "DNN_Autoenc_784.pth"
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

alpha: float         = 1.0   # reconstruction loss weight
beta: float          = 1.0   # classification loss weight

zmft_epochs: int = 10        # fine-tune epochs after neuron pruning

P: int         = 36
N: int         = 36
mesh_rows: int = int(math.sqrt(P))
mesh_cols: int = int(math.sqrt(P))

import types

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
    """Total FLOPs = 2 x sum(in_features x out_features) across all Linear layers.
    Uses dense dimensions — aligns with batch inference time.
    """
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
    print("  PIPELINE COMPARISON — Baseline vs L1(25%) vs Comm.Cost vs Bridge vs matched variants")
    print("  Model: Autoencoder+Classifier [784->128->64->128->784 + 64->10] | Dataset: MNIST")
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
# Pruning criterion helpers (L2, Taylor, SNIP)
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


def _accumulate_taylor_scores(wrapper_model, full_mdl, X_calib, y_calib, device,
                               n_batches=10, batch_sz=128):
    """Accumulate |w * grad| scores.

    We run the wrapper_model forward (encoder + classifier path) with
    CrossEntropyLoss only — this ensures gradients flow directly through
    the wrapper's Linear layers so mod.weight.grad is populated.
    The reconstruction loss cannot be used here because the decoder is
    not part of wrapper_model, but the classification gradient is
    sufficient to score the encoder+classifier weights.
    """
    import numpy as np
    was_training = wrapper_model.training
    wrapper_model.train()
    wrapper_model.zero_grad()
    if isinstance(X_calib, np.ndarray):
        X_calib = torch.from_numpy(X_calib).float()
    if isinstance(y_calib, np.ndarray):
        y_calib = torch.from_numpy(y_calib).long()
    n = X_calib.shape[0]
    importance = {}
    for mod in wrapper_model.modules():
        if isinstance(mod, nn.Linear):
            importance[mod] = torch.zeros_like(mod.weight.data)
    for b in range(n_batches):
        idx     = torch.randperm(n, generator=torch.Generator().manual_seed(b))[:batch_sz]
        xb      = X_calib[idx].to(device)
        yb      = y_calib[idx].to(device)
        cls_out = wrapper_model(xb)          # forward through EncoderClassifier only
        loss    = nn.CrossEntropyLoss()(cls_out, yb)
        wrapper_model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for mod in wrapper_model.modules():
                if isinstance(mod, nn.Linear) and mod.weight.grad is not None:
                    importance[mod] += (mod.weight.data * mod.weight.grad).abs()
    wrapper_model.zero_grad()
    if not was_training:
        wrapper_model.eval()
    for mod in importance:
        importance[mod] /= n_batches
    return importance


def _apply_scores_at_threshold(wrapper_model, importance, threshold):
    with torch.no_grad():
        for mod, scores in importance.items():
            mod.weight.data.mul_((scores > threshold).float())


def _count_nonzero_weights(wrapper_model):
    total = 0
    for mod in wrapper_model.modules():
        if isinstance(mod, nn.Linear):
            total += int((mod.weight.data != 0).sum().item())
    return total


def prune_taylor_global(wrapper_model, amount, full_mdl, X_calib, y_calib, device,
                        n_batches=10, batch_sz=128, target_edges=None):
    """Taylor pruning with optional exact edge-count targeting via phase-2 trim."""
    importance = _accumulate_taylor_scores(
        wrapper_model, full_mdl, X_calib, y_calib, device, n_batches, batch_sz)
    for mod, scores in importance.items():
        scores[mod.weight.data == 0] = float("-inf")
    all_scores = torch.cat([s.view(-1) for s in importance.values()])
    n_finite   = int((all_scores > float("-inf")).sum().item())
    if target_edges is None:
        n_prune = int(round(amount * all_scores.numel()))
    else:
        n_prune = max(0, n_finite - target_edges)
    if n_prune > 0:
        threshold = torch.topk(all_scores, n_prune, largest=False).values.max()
        _apply_scores_at_threshold(wrapper_model, importance, threshold)
    if target_edges is None:
        return
    current = _count_nonzero_weights(wrapper_model)
    excess  = current - target_edges
    if excess > 0:
        candidates = []
        for mod in wrapper_model.modules():
            if isinstance(mod, nn.Linear):
                w  = mod.weight.data
                nz = w.nonzero(as_tuple=False)
                for idx in nz:
                    candidates.append((w[idx[0], idx[1]].abs().item(),
                                       mod, idx[0].item(), idx[1].item()))
        candidates.sort(key=lambda x: x[0])
        with torch.no_grad():
            for _, mod, r, c in candidates[:excess]:
                mod.weight.data[r, c] = 0.0


def prune_snip_global(wrapper_model, amount, full_mdl, X_calib, y_calib, device,
                      n_batches=10, batch_sz=128, target_edges=None):
    """SNIP — post-training equivalent to Taylor (same |w*g| formula)."""
    prune_taylor_global(wrapper_model, amount, full_mdl, X_calib, y_calib, device,
                        n_batches, batch_sz, target_edges=target_edges)


# ------------------------------------------------------------------
# Generic helper: run a matched-pruning comparison branch
# ------------------------------------------------------------------

def run_matched_branch(
    label: str,
    l1_full_model_copy_arg: nn.Module,
    l1_edge_count: int,
    commcost_edge_count: int,
    prune_fn,
    prune_kwargs: dict,
    X_train, y_train, X_test, y_test,
    device, timings,
) -> None:
    """Reload the L1 checkpoint, prune to commcost_edge_count, fine-tune,
    then record. target_edges and full_mdl are injected automatically."""
    global full_model

    print("\n" + "=" * 70)
    print(f"  STEP: {label}")
    print(f"  Target: match Comm.Cost edge count ({commcost_edge_count:,} edges)")
    print(f"  Starting from L1 checkpoint ({l1_edge_count:,} edges)")
    print("=" * 70)

    full_model     = copy.deepcopy(l1_full_model_copy_arg)
    enc_cls_branch = EncoderClassifier(copy.deepcopy(l1_full_model_copy_arg)).to(device)
    net_branch     = PrunableNetwork(enc_cls_branch, eps=0.0)
    install_autoenc_patch(net_branch)

    assert net_branch.graph.number_of_edges() == l1_edge_count, (
        f"Checkpoint edge count mismatch: "
        f"expected {l1_edge_count}, got {net_branch.graph.number_of_edges()}"
    )

    total_weights_branch = sum(
        p.numel() for name, p in net_branch.model.named_parameters()
        if "weight" in name and p.dim() == 2
    )
    amount_needed = (total_weights_branch - commcost_edge_count) / total_weights_branch

    print(f"\n[{label}] Total weight slots : {total_weights_branch:,}")
    print(f"[{label}] Cumulative amount   : {amount_needed:.4f}")

    if amount_needed <= 0:
        print(f"[{label}] WARNING: Comm.Cost edge count >= total weights — skipping.")
        return
    if amount_needed >= 1.0:
        print(f"[{label}] WARNING: amount_needed >= 1.0, clamping to 0.99.")
        amount_needed = 0.99

    if 0 < amount_needed < 1.0:
        import inspect
        kwargs = dict(prune_kwargs)
        sig    = inspect.signature(prune_fn)
        if "target_edges" in sig.parameters:
            kwargs["target_edges"] = commcost_edge_count
        if "full_mdl" in sig.parameters:
            kwargs["full_mdl"] = full_model
        prune_fn(net_branch.model, amount_needed, **kwargs)
        net_branch.commit_pruning()

    matched_edges = net_branch.graph.number_of_edges()
    print(f"[{label}] Edges after pruning : {matched_edges:,} "
          f"(target was {commcost_edge_count:,}, diff={matched_edges - commcost_edge_count:+,})")

    reapply_zero_masks(net_branch)
    fine_tune(full_model, X_train, y_train, device=device, lr=lr,
              epochs=zmft_epochs, alpha=alpha, beta=beta, seed=seed)

    print(f"\n=== {label} ===")
    eval_nn(full_model, X_test=X_test, y_test=y_test, device=device,
            timings=timings, evaluate_fn=evaluate_model)
    noc_branch = eval_noc(net_branch.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                          title=label, visualize=False)
    record_stage(label, net_branch, noc_branch, X_test, y_test, device, timings)
    noc_branch.visualize_all(title=label, metrics=make_metrics(stages))


# ------------------------------------------------------------------
# 1. Train / load
# ------------------------------------------------------------------

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

# ------------------------------------------------------------------
# 2. Baseline evaluation
# ------------------------------------------------------------------

print("\n=== Initial model ===")
eval_nn(full_model, X_test=X_test, y_test=y_test, device=device,
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
# Save L1 checkpoint — used later for the L1-matched comparison
# We deepcopy the full model object (not just state_dict) because
# remove_lonely_neurons has already structurally resized the layers
# (e.g. fc1: 300→299), so the original MLP architecture no longer
# matches. Deepcopying the model preserves the resized architecture.
# ------------------------------------------------------------------
l1_model_copy       = copy.deepcopy(net.model)   # wrapper
l1_full_model_copy  = copy.deepcopy(full_model)   # full model
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

# Deepcopy so l1_full_model_copy stays pristine for L2/Taylor/SNIP branches.
full_model         = copy.deepcopy(l1_full_model_copy)
enc_cls_wrapper_l1 = EncoderClassifier(copy.deepcopy(l1_full_model_copy)).to(device)
net_l1             = PrunableNetwork(enc_cls_wrapper_l1, eps=0.0)
install_autoenc_patch(net_l1)

assert net_l1.graph.number_of_edges() == l1_edge_count, (
    f"Checkpoint edge count mismatch: "
    f"expected {l1_edge_count}, got {net_l1.graph.number_of_edges()}"
)

total_weights_l1 = sum(
    p.numel() for name, p in net_l1.model.named_parameters()
    if "weight" in name and p.dim() == 2
)
edges_to_remove = l1_edge_count - commcost_edge_count
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

matched_edge_count = net_l1.graph.number_of_edges()
print(f"[L1-matched] Edges after pruning: {matched_edge_count:,} "
      f"(target was {commcost_edge_count:,}, "
      f"diff={matched_edge_count - commcost_edge_count:+,})")

reapply_zero_masks(net_l1)
fine_tune(full_model, X_train, y_train, device=device, lr=lr,
          epochs=zmft_epochs, alpha=alpha, beta=beta, seed=seed)

print("\n=== After L1-matched pruning ===")
eval_nn(full_model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
noc_l1m = eval_noc(net_l1.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                   title="After L1-matched", visualize=False)
record_stage("After L1-matched", net_l1, noc_l1m, X_test, y_test, device, timings)
noc_l1m.visualize_all(title="After L1-matched", metrics=make_metrics(stages))

# ------------------------------------------------------------------
# 7. L2-matched comparison
# ------------------------------------------------------------------

run_matched_branch(
    label="After L2-matched",
    l1_full_model_copy_arg=l1_full_model_copy,
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
    l1_full_model_copy_arg=l1_full_model_copy,
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
    l1_full_model_copy_arg=l1_full_model_copy,
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
