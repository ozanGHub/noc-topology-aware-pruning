import math

from banknote_mlp import load_or_train, fine_tune
from prunable_network import PrunableNetwork
from eval_helper import *
from visualize import visualize_model
from banknote_mlp import evaluate_model

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

# Neural network and training
csv_path: str       = "BankNote_Authentication.csv"
model_path: str     = "banknote_mlp.pth"
seed: int           = 42
test_size: float    = 0.2
batch_size: int     = 64
lr: float           = 1e-3
epochs: int         = 50
weight_decay: float = 1e-4
threshold: float    = 0.5
force_train: bool   = True

# Iterative L1 pruning
l1_target: float  = 0.25   # fraction of original edges to remove
l1_steps: int     = 5      # number of prune → fine-tune iterations
ft_epochs: int    = 10     # fine-tuning epochs per step

# Neuron pruning
low_mag_percentile: float = 75.0  # raise to remove more neurons
degree_quantile: float    = 0.25  # lower to remove more neurons
min_in_degree: int        = 0

# Heuristic bridge-link pruning
bridge_amount: float = 0.25

# NoC and mesh
P: int         = 9
N: int         = 6
mesh_rows: int = int(math.sqrt(P))
mesh_cols: int = int(math.sqrt(P))

# ------------------------------------------------------------------
# 1. Train / load
# ------------------------------------------------------------------

model, scaler, (X_train, X_test, y_train, y_test), timings, device = load_or_train(
    csv_path, model_path,
    seed=seed, test_size=test_size, batch_size=batch_size,
    lr=lr, epochs=epochs, weight_decay=weight_decay,
    threshold=threshold, force_train=force_train,
)

net = PrunableNetwork(model, eps=0.0)

# ------------------------------------------------------------------
# 2. Baseline evaluation
# ------------------------------------------------------------------

print("\n=== Initial model ===")
net.visualize("Initial model")
eval_nn(net.model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
noc = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
               title="Initial model")
noc.visualize_mesh(title="Mesh")

# ------------------------------------------------------------------
# 3. Iterative L1 link pruning + fine-tuning
# ------------------------------------------------------------------
original_edge_count = net.graph.number_of_edges()
target_removals = round(original_edge_count * l1_target)

# Total weight count is fixed — includes already-zeroed weights
total_weights = sum(
    p.numel() for name, p in net.model.named_parameters()
    if "weight" in name and p.dim() == 2
)

for step in range(1, l1_steps + 1):
    already_removed = original_edge_count - net.graph.number_of_edges()
    remaining_to_remove = target_removals - already_removed
    steps_left = l1_steps - step + 1
    this_step_target = round(remaining_to_remove / steps_left)

    # Express as fraction of ALL weights (including already-zeroed ones)
    cumulative_target = already_removed + this_step_target
    amount = cumulative_target / total_weights

    if amount >= 1.0:
        print(f"  [Step {step}/{l1_steps}] Skipping — not enough edges remaining.")
        break

    net.prune_links_l1_global(amount=amount)
    fine_tune(net.model, X_train, y_train, device=device, lr=lr,
              epochs=ft_epochs, seed=seed + step)

    removed = original_edge_count - net.graph.number_of_edges()
    print(f"  [Step {step}/{l1_steps}] Edges removed: {removed}/{original_edge_count} "
          f"({removed / original_edge_count:.1%} of original)")

visualize_model(net.model, title="After iterative L1 pruning and fine-tuning before removing lonely neurons")
net.commit_pruning() # Necessary after fine tuning!
net.remove_lonely_neurons() # Necessary after link pruning!

visualize_model(net.model, title="After iterative L1 pruning and fine-tuning")
print(f"\n=== After iterative L1 pruning ({l1_target:.0%} of original links removed) ===")
net.visualize("After iterative L1 pruning and fine-tuning")
eval_nn(net.model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
noc = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
               title="After iterative L1 pruning and fine-tuning")

# ------------------------------------------------------------------
# 4. Neuron pruning (high in-degree, low magnitude)
# ------------------------------------------------------------------

net.prune_high_indegree_low_magnitude(
    low_mag_percentile=low_mag_percentile,
    degree_quantile=degree_quantile,
    min_in_degree=min_in_degree,
)

print("\n=== After high in-degree / low magnitude neuron pruning ===")
net.visualize("After neuron pruning")
eval_nn(net.model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
noc = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
               title="After neuron pruning")

# ------------------------------------------------------------------
# 5. Heuristic bridge-link pruning
# ------------------------------------------------------------------

net.prune_links_heuristic(
    groups=noc.groups,
    amount=bridge_amount,
    debug=True,
)
net.remove_lonely_neurons() # Necessary after link pruning!

print("\n=== After heuristic bridge-link pruning ===")
net.visualize("After heuristic bridge-link pruning")
eval_nn(net.model, X_test=X_test, y_test=y_test, device=device,
        timings=timings, evaluate_fn=evaluate_model)
noc = eval_noc(net.graph, P=P, N=N, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
               title="After heuristic bridge-link pruning")