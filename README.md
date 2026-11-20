# Topology-Aware Pruning and Mapping for NoC-Based DNN Accelerators

Code, checkpoints and logs for the paper *"Topology-Aware Pruning and Mapping Algorithms for Deep Neural Network Accelerators"* (Concurrency and Computation: Practice and Experience, under revision).

The pipeline prunes a trained fully-connected network so that the communication cost on a 2-D mesh Network-on-Chip is reduced, then groups the surviving neurons and maps the groups to processing elements (PEs):

1. **Communication-cost-aware neuron pruning**: removes neurons that sit at cross-group boundaries of the weighted communication task graph (WCTG) and carry low-magnitude weights.
2. **Bridge-link edge pruning**: removes the weakest cross-group edges, in batches, until the accuracy budget is used.
3. **Grouping and mapping**: in-degree BFD neuron grouping, then a simulated-annealing placement that minimises the WCTG node-to-node (N2N) cost.

Classical baselines (Struct-L1/L2, Taylor, Activation, Random for neurons; L1, L2, first-order Taylor, SNIP for edges) are in the same scripts.

> **Citation.** If you use this code, please cite the paper. Each tagged release of this repository is archived on Zenodo with a persistent DOI; use the "Cite this repository" button on GitHub (`CITATION.cff`) or the DOI shown on the Zenodo record.

## 1. Setup

```
pip install -r requirements.txt
```

- `gurobipy` is listed because of the optional exact-mapping module (`mapping_ilp.py`). It needs a Gurobi licence (free academic licences exist). The default pipeline uses the simulated-annealing mapper (`mapping_sa.py`), which does not.
- MNIST, Fashion-MNIST and CIFAR-10 are public datasets. They are downloaded automatically by `torchvision` the first time a script runs (see `data/ReadMe.txt`).
- Reference environment of the reported runs: Python 3.10.19, PyTorch 2.10.0, CUDA 12.6, one NVIDIA GeForce RTX 4090 (24 GB), Linux.
- `requirements.txt` pins the CUDA 12.6 build of PyTorch (`torch==2.10.0+cu126`) used for the reported runs. For another CUDA version, change the `--extra-index-url` and the `+cu` suffix accordingly, or install a matching PyTorch build first.

## 2. Repository layout

| Path | Content |
|---|---|
| `main_*.py` | Experiment scripts. Name pattern: `main_[Model]_[Dataset]_[PruningType]_[ComparisonType]`. Full list in `main__ReadMe.txt`. |
| `DNN_*.py`, `DNN_*.pth` | Model definitions / training and the trained baseline checkpoints. |
| `pruning.py`, `pruning_meta.py`, `comm_cost_neuron_selector.py`, `prunable_network*.py` | Neuron and edge pruning heuristics and the prunable-network wrappers. |
| `graph.py`, `graph_conv.py`, `grouping_heuristic.py`, `noc_mapper.py`, `mapping_*.py` | WCTG construction, in-degree BFD grouping, SA/ILP mapping onto the mesh. |
| `aggregate_seeds.py` | Aggregates the per-seed logs into mean and standard deviation. |
| `figure_*.py` | Scripts that draw the figures of the paper (`figure__ReadMe.txt`). |
| `_testResults*.txt` | Result summaries of the reported experiments (see Section 4). |
| `output_*.log` | Raw logs of the reported CIFAR-10 runs, one file per seed. |
| `trajectoryAndTimingLogs/` | Per-run pruning trajectories and timing logs. |
| `interimExperimentalResultLogs-notUsedEvalOnly/` | Intermediate and exploratory logs, **not used for the reported results**; kept for transparency (for example the retest of LeNet-300-100 on a retrained baseline mentioned in the Limitations). |
| `archiveFiles/` | Early tests and follow-up experiments. Not part of the reported results. |

`__README_FIRST.txt` explains the script naming, shows the flow of Branch A step by step and lists the result files.

## 3. How to run

Each script is run directly; there are no command-line arguments. The CIFAR-10 scripts read their settings from **environment variables**; the defaults at the top of each script are the configuration used for that model.

The reported runs were made on Linux with bash. As an example, the LeNet-5 operating point of Table 12 (accuracy budget 1 pp, neuron phase limited to 0.8 pp, three seeds):

```bash
for s in 42 10042 20042; do
  RUN_TAG=_tot1n08 ACCURACY_TOLERANCE=0.01 NEURON_SEQUENTIAL=1 NEURON_TOL=0.008 NEURONS_PER_STEP=3 \
  L2_LONELY=1 SEED=$s SNAPSHOT_INTERVAL=0.0025 EVAL_FAST=1 GRAPH_FAST=1 EDGE_FAST=1 BRIDGE_GROUPS_FAST=1 \
  python -u main_LeNet5_CIFAR10_e2e_AccuracyBudget_v8.py > output_LeNet5_CIFAR10_v8_tot1n08_seed$s.log 2>&1
done
```

On Windows PowerShell set the variables with `$env:SEED = "42"` before the `python` call.

Variables used by the CIFAR-10 scripts include `SEED` (random seed; the paper uses 42, 10042 and 20042), `MODEL_PATH` (baseline checkpoint), `FT_LR` (fine-tuning learning rate), `ACCURACY_TOLERANCE` (accuracy budget), `BRANCHES` (subset of branches, e.g. `AD`), `NEURON_MAX_FRAC` and `NEURON_SHARE` (matched-neuron-count and share settings) and `RUN_TAG`. Each script defines its own defaults at the top of the file.

> **Warning.** `FORCE_TRAIN=1` retrains the baseline and **overwrites** the checkpoint file. Do not set it unless you intend to replace the supplied baselines.

The MNIST and Fashion-MNIST scripts have the settings in the file and are single runs. The CIFAR-10 results are mean ± standard deviation over three seeds.

## 4. Mapping from paper to code

| Paper | Content | Script(s) | Result files and logs |
|---|---|---|---|
| Table 6 | LeNet-300-100, MNIST, 1 % budget | `main_LeNet_300_100_e2e_AccuracyBudget.py` | `output4_LeNet_300_100.log`, `output9_LeNet_300_100.log`; summary in `_testResults_e2e_AccuracyBudget.txt` |
| Table 7 | LeNet-500-300, MNIST, 1 % budget | `main_LeNet_500_300_e2e_AccuracyBudget.py` | `output5_LeNet_500_300.log`, `output10_LeNet_500_300.log`; same summary file |
| Table 8 | MLP-1024-1024, MNIST, 1 % budget | `main_MLP_784_1024_e2e_AccuracyBudget.py` | `output7_MLP_784_1024.log`; same summary file |
| Table 9 | Autoencoder, MNIST and Fashion-MNIST, 1 % budget | `main_Autoenc_784_MNIST_e2e_AccuracyBudget.py` (named `main_Autoenc_784_e2e_AccuracyBudget.py` in the summary file), `main_Autoenc_784_FashionMNIST_e2e_AccuracyBudget.py` | `output6_Autoenc784.log`, `output8_Autoenc784_FaMNIST.log`; same summary file |
| Table 10 | Accuracy-budget sweep, LeNet-500-300 | `main_LeNet_500_300_BudgetSweep.py` | `_testResults_e2e_AccuracySweepLeNet500-300.txt` |
| Table 11 | Cross-model summary | derived from Tables 6-9 and 12 | (see above) |
| Table 12 | CIFAR-10, final state, three seeds | one script per model, one run per seed (commands below) | `_testResults<Model>_CIFAR10_multipleSeed.txt` (commands and aggregated results), `output_<Model>_CIFAR10_*_seed<seed>.log`, `aggregate_seeds.py` |
| Table 13 | CIFAR-10, matched neuron count (LeNet-5, AlexNet) | `main_LeNet5_CIFAR10_e2e_AccuracyBudget_v8.py`, `main_AlexNet_CIFAR10_e2e_AccuracyBudget_Metaheuristic_v8.py` with the matched-count variables (commands below) | `output_LeNet5_CIFAR10_v8_n20/n30/n40*.log`, `output_AlexNet_v8_n20ne*.log`, `output_AlexNet_v8_n30ne*.log` |
| Table 5 | Offline cost of grouping and mapping, Branch A, seed 42 | same CIFAR-10 scripts | the `TIMING:` line of Branch A in the log named below; machine-readable copies in `trajectoryAndTimingLogs/timing_*.csv` |
| Figures | All figures | `figure_*.py` (`figure__ReadMe.txt`) | generated as `.svg`, `.png`, `.pdf` |

**LeNet-5 matched neuron count (Table 13, rows 20 %, 30 %, 40 %).** The LeNet-5 rows are produced by:

```bash
for f in 0.20 0.30 0.40; do tag=_n${f#0.}; for s in 42 10042 20042; do
  RUN_TAG=$tag SNAPSHOT_INTERVAL=0.0025 NEURON_SEQUENTIAL=1 NEURON_TOL=0.02 NEURON_CAP_FROM_L1=1 \
  NEURON_MAX_FRAC=$f NEURON_STEP_FRAC=0.015 L2_LONELY=1 SEED=$s \
  EVAL_FAST=1 GRAPH_FAST=1 EDGE_FAST=1 BRIDGE_GROUPS_FAST=1 \
  python -u main_LeNet5_CIFAR10_e2e_AccuracyBudget_v8.py > output_LeNet5_CIFAR10_v8${tag}_seed$s.log 2>&1
done; done
```

**Other CIFAR-10 models (Table 12).** The commands and the aggregated results of each model are in its `_testResults<Model>_CIFAR10_multipleSeed.txt`. The runs behind Table 12 are:

```bash
for s in 42 10042 20042; do
  # LeNet-300-100 and LeNet-500-300 (first-trained checkpoints; Branches A-G)
  SEED=$s EVAL_FAST=1 GRAPH_FAST=1 BRIDGE_FAST=1 BRIDGE_GROUPS_FAST=1 \
    python -u main_LeNet_300_100_CIFAR10_e2e_AccuracyBudget_Metaheuristic_v3.py > output_LeNet_300_100_CIFAR10_v3_A-G_seed$s.log 2>&1
  SEED=$s EVAL_FAST=1 GRAPH_FAST=1 BRIDGE_FAST=1 BRIDGE_GROUPS_FAST=1 \
    python -u main_LeNet_500_300_CIFAR10_e2e_AccuracyBudget_Metaheuristic_v3.py > output_LeNet_500_300_CIFAR10_v3_A-G_seed$s.log 2>&1
  # MLP-1024 (retrained baseline, Branches A-K)
  MODEL_PATH=DNN_MLP_3072_1024_CIFAR10_full300.pth FT_LR=1e-4 SEED=$s EVAL_FAST=1 GRAPH_FAST=1 BRIDGE_FAST=1 \
    python -u main_MLP_3072_1024_CIFAR10_e2e_AccuracyBudget_Metaheuristic_v3.py > output_MLP_3072_1024_CIFAR10_v3_full300_A-K_seed$s.log 2>&1
  # AutoEnc-64 (Branches A-I) and AutoEnc-128 (L2_LONELY=1)
  SEED=$s EVAL_FAST=1 GRAPH_FAST=1 BRIDGE_FAST=1 \
    python -u main_Autoenc_3072_64_CIFAR10_e2e_AccuracyBudget_Metaheuristic_v3.py > output_Autoenc_3072_64_CIFAR10_v3_A-I_seed$s.log 2>&1
  SEED=$s L2_LONELY=1 EVAL_FAST=1 GRAPH_FAST=1 BRIDGE_FAST=1 \
    python -u main_Autoenc_3072_128_CIFAR10_e2e_AccuracyBudget_Metaheuristic_v3.py > output_Autoenc_3072_128_CIFAR10_v3_L2lonely_seed$s.log 2>&1
done
```

**AlexNet (Table 12 row and Table 13 rows).** All AlexNet runs start from the fully annealed baseline `DNN_AlexNet_CIFAR10_full200.pth` and use Branches A and D. Table 12 (neuron phase limited to 80 % of the remaining accuracy budget):

```bash
for s in 42 10042 20042; do
  NEURON_MAX_FRAC= NEURON_SHARE=0.8 NEURON_SHARE_BASIS=remaining RUN_TAG=_share08 BRANCHES=A,D \
  L2_LONELY=1 NEURON_SEQUENTIAL=1 NEURON_STEP_FRAC=0.015 SEED=$s \
  EVAL_FAST=1 GRAPH_FAST=1 EDGE_FAST=1 BRIDGE_GROUPS_FAST=1 \
  python -u main_AlexNet_CIFAR10_e2e_AccuracyBudget_Metaheuristic_v8.py > output_AlexNet_v8_share08_seed$s.log 2>&1
done
```

Table 13 (matched neuron count: 30 % and 20 % of the 1,536 fully-connected neurons):

```bash
for s in 42 10042 20042; do for f in 0.30 0.20; do
  tag=n${f#0.}ne
  MODEL_PATH=DNN_AlexNet_CIFAR10_full200.pth FT_LR=1e-4 L1_TARGET=0.10 BRANCHES=A,D L2_LONELY=1 \
  NEURON_SEQUENTIAL=1 NEURON_STEP_FRAC=0.015 SNAPSHOT_INTERVAL=0.005 SNAPSHOT_NEURON_END=1 \
  RUN_TAG=_$tag NEURON_MAX_FRAC=$f NEURON_TOL=0.05 SEED=$s \
  EVAL_FAST=1 GRAPH_FAST=1 EDGE_FAST=1 BRIDGE_GROUPS_FAST=1 \
  python -u main_AlexNet_CIFAR10_e2e_AccuracyBudget_Metaheuristic_v8.py > output_AlexNet_v8_${tag}_seed$s.log 2>&1
done; done
```

The five-seed repeat of the 30 % case (Section 5.11.2) adds seeds 30042 and 40042 to the second loop. Every log records the effective settings of its run in its `[Checkpoint]` lines. Per-seed results are aggregated with `python aggregate_seeds.py "output_AlexNet_v8_n30ne_seed*.log" aggregate_AlexNet_n30ne_v8.csv`.

**Table 5 sources** (Branch A, seed 42, the `TIMING: total=...` line): LeNet-5 `output_LeNet5_CIFAR10_v8_tot1n08_seed42.log`; AlexNet `output_AlexNet_v8_share08_seed42.log`; AutoEnc-64 `output_Autoenc_3072_64_CIFAR10_v3_A-I_seed42.log`; LeNet-300-100 `output_LeNet_300_100_CIFAR10_v3_A-G_seed42.log`; AutoEnc-128 `output_Autoenc_3072_128_CIFAR10_v3_L2lonely_seed42.log`; LeNet-500-300 `output_LeNet_500_300_CIFAR10_v3_A-G_seed42.log`; MLP-1024 `output_MLP_3072_1024_CIFAR10_v3_A-K_seed42.log`.

The commands above are the normalised form of those recorded in the `_testResults*_CIFAR10_multipleSeed.txt` files; every variable not shown keeps its default from the top of the script. The mean and standard deviation of each Table 12 row can be recomputed from the three per-seed logs of its run family with `aggregate_seeds.py`.

**Not reported.** Matched-neuron-count runs for the fully-connected and bottleneck CIFAR-10 models are not part of the results. The forced-count variant currently draws all removals from the last hidden layer, so those results are withheld until the procedure is corrected.

## 5. Checkpoints and logs

- `DNN_*.pth` are the trained baseline checkpoints (first-trained for the smaller models; retrained last-epoch checkpoints for AlexNet and MLP-1024, see Limitation (iii) of the paper). Files ending `_original` or `_full200` / `_full300` are the earlier snapshot and the full-schedule retrain of the same model.
- The `output_*.log` files and the `_testResults*.txt` summaries are the raw outputs from which the tables were built.

## 6. Reproducibility notes

- Fixed seeds are set for fine-tuning and for the simulated-annealing mapping. Mapping cost is itself stochastic (for AlexNet the baseline WCTG cost varies slightly between seeds), so percentage reductions are computed against the baseline of the same run.
- Absolute run times depend on the GPU and CPU. The offline-cost table reports wall-clock times from the reference machine.

## 7. License

Code and logs are released under the MIT licence (see `LICENSE`).

## 8. Funding

Supported by TÜBİTAK under grant number 123R075.
