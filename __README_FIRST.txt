Scripts to be executed are explained in: 
    "main__ReadMe.txt". 
    'Which script does what?' is given in this read me file.
    
    ### ------------------------------------------------ ###
    ### PLEASE READ "main__ReadMe.txt" AS THE FIRST STEP ###
    ### ------------------------------------------------ ###
    
    General naming Structure for the scripts is as follows:
        main_[DNNModel]_[Dataset-Optional]_[PruningType]and/or[ComparisonType]
              DNNModel    : MNIST		AutoEnc-64		784-128-64-128-784+64-10     Bottleneck
                            MNIST		LeNet-300-100 	784-300-100-10            	Small Feedforward
                            MNIST		LeNet-500-300 	784-500-300-10            	Medium Feedforward
                            MNIST		MLP-1024-1024 	784-1024-1024-10          	Large Feedforward
							
							FaMNIST		AutoEnc-64		784-128-64-128-784+64-10     Bottleneck	
							
							CIFAR-10	LeNet-300-100 	3072-300-100-10 			Small Feedforward
							CIFAR-10	LeNet-500-300 	3072-500-300-10 			Medium Feedforward
							CIFAR-10	MLP-1024-1024 	3072-1024-1024-10 			Large Feedforward
							CIFAR-10	AutoEnc-64 		3072-128-64-128-3072+64-10 	Bottleneck
							CIFAR-10	AutoEnc-128		3072-512-128-512-3072+128-10 Medium Bottleneck
							CIFAR-10	LeNet-5 		conv1-conv3-c5-f6-out 		Convolutional
							CIFAR-10	AlexNet 		conv1:conv5-c6-c7-out 		Convolutional

              Dataset     : MNIST (for all DNNs) and FaMNIST (for Autoencoder only) and CIFAR-10
              PruningType : Neuron Pruning only
                            Edge Pruning only
                            e2e: end to End test (neuron pruning + edge pruning)
           ComparisonType : accuracy: fixed accuracy or accuracy sweep 
                                      (accuracy budget is fixed, all methods freely 
                                       perform prunig with their core strength until 
                                       budget (i.e.: max 1% drop from baseline) is exhausted)
                            matched : (neuron or edge pruning amount 
                                       are matched accross all methods)

    ### ------------------------------------------------ ###
    ###                 RESULTS SUMMARY                  ###
    ### ------------------------------------------------ ###

RESULTS ARE SUMMARIZED IN:
1- testResults_e2e_AccuracyBudget.txt:  Main end to end comparison file.
                                        sets a fixed accuracy drop threshold (i.e. 1%)
                                        and all models run freely until they reach this drop.
                                        Then e2e algorithms are compared for each model.

2- testResults_e2e_AccuracySweepLeNet500-300.txt:  Executed for LeNet500-300 only.
                                        Performs heuristic method e2e comparison with Taylor + L2 baseline
                                        for accuracies ranging from 0.5% to 5%.

3&4- testResultsMatchedEdgePruning.txt:  
   testResultsMatchedNeuronPruning.txt: These are the individual comparisons of each model
                                        with the constraint of same edge or neuron pruning count.
                                        These results are used as basis for selecting which end 
                                        to end combination to be used in the main comparison tests.

5&6&7&8&9&10&11 _testResults[ModelName]_CIFAR10_multipleSeed.txt:
										These are the individual comparisons of each model for
										CIFAR-10 Dataset

    ### ------------------------------------------------ ###
    ###          FIGURE plottings for the paper          ###
    ### ------------------------------------------------ ###
FIGURE plottings for the paper are generated with figure_[Name].py files. 
                                Which *.py do what is explained in "figure__ReadMe.txt"

Example low for main_LeNet_300_100_e2e_AccuracyBudget.py:
####################################################################
#                                                                  #
#   BRANCH A (Heuristic Comm. Cost + Heuristic Bridge Link) FLOW   #
#                                                                  #
####################################################################

LeNet 784-300-100-10 weights (.pth)
            │
            ▼
    load_or_train()
    MNIST, seed=42, batch=64
            │
            ▼
  ┌─────────────────────────────┐
  │        BASELINE             │
  │  construct_nn_graph()       │  nodes=(layer,neuron), edges=non-zero weights
  │  NoCMapper.run()            │  indegree_grouping → WCTG → SA → cost
  │  record_stage("Baseline")   │  acc=0.9822, cost=66,976
  └─────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────┐
  │   SHARED L1 PRE-PRUNING     │
  │   5 × 5% iterative L1       │  remove 25% weakest-magnitude edges
  │   + fine-tune 10 epochs     │  globally, in equal steps
  │   commit + remove lonely    │
  │   record("After L1 25%")    │  acc=0.9758, cost=52,293
  └─────────────────────────────┘
            │
            ▼
   deepcopy → l1_model_copy          ← checkpoint all branches start from
   anchor_accuracy = 0.9822          ← BASELINE accuracy (not L1)
   floor = 0.9822 − 0.01 = 0.9722
   snapshots at drops: [0.5%, 1.0%]
            │
            ▼
╔═══════════════════════════════════════════════════════════╗
║         BRANCH A: Comm. Cost Neuron + Bridge-Link         ║
║  neuron_tolerance=0.5%  |  full budget=1.0%               ║
║  neurons_per_step=3     |  neuron:edge ratio = 1:1        ║
║  micro_ft=2 epochs      |  bridge_step=2%  post=5%        ║
╚═══════════════════════════════════════════════════════════╝
            │
            ▼
  net = PrunableNetwork(deepcopy(l1_model_copy))
  best_model = current model
            │
            ▼
  ┌─────────────────────────────────────────────────────┐
  │          INTERLEAVED LOOP  (max 1000 steps)         │
  │                                                     │
  │  decide: neuron step or edge step? (1:1 ratio)      │
  │  save pre_step_model (for revert)                   │
  │                                                     │
  │  ── NEURON STEP (while drop < 0.5%) ──              │
  │  iterative_commcost_neuron_pruning(max_neurons=3)   │
  │    loop per neuron:                                 │
  │      G = net.graph  (always fresh)                  │
  │      indegree_grouping(G, N=36)  → groups           │
  │      select_neurons_high_indegree_low_magnitude()   │
  │        Stage-1: bottom 90% magnitude AND            │
  │                 bottom 25% in-degree quantile       │
  │      score each candidate:                          │
  │        Σ |w| on cross-group edges in/out            │
  │      remove highest-scoring neuron                  │
  │      net.prune_neurons() → rebuilds graph           │
  │    repeat × 3 neurons max                           │
  │  net.remove_lonely_neurons()                        │
  │                                                     │
  │  ── EDGE STEP ──                                    │
  │  _bridge_edge_step(amount = 2% of active edges)     │
  │    eval_noc() → fresh groups                        │
  │    collect all cross-group (bridge) edges           │
  │    sort by |weight| ascending                       │
  │    zero cheapest N bridge edges                     │
  │    fallback to L2 if no bridge edges remain         │
  │    net.remove_lonely_neurons()                      │
  │    net.commit_pruning() + resync_graph()            │
  │                                                     │
  │  reapply_zero_masks()                               │
  │  fine_tune(2 epochs)                                │
  │  commit_pruning() + resync_graph()                  │
  │  evaluate → acc, drop = anchor − acc                │
  │                                                     │
  │  if drop ≤ 1.0%:                                    │
  │    best_model = current                             │
  │    if drop ≥ 0.5%: end neuron phase →               │
  │         switch to edge-only at 5%/step              │
  │    if drop crosses 0.5% threshold → SNAPSHOT        │
  │    if drop crosses 1.0% threshold → SNAPSHOT        │
  │  else:                                              │
  │    revert to best_model                             │
  │    STOP                                             │
  └─────────────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────┐
  │  SNAPSHOT @ drop ≤ 0.5%     │  first crossing only
  │  NoCMapper.run(net.graph)   │  full SA mapping
  │  record_stage()             │
  └─────────────────────────────┘
            │  (loop continues, edge-only phase)
            ▼
  ┌─────────────────────────────┐
  │  SNAPSHOT @ drop ≤ 1.0%     │
  │  NoCMapper.run(net.graph)   │
  │  record_stage()             │
  └─────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────────────┐
  │  FINAL RECORD                                       │
  │  eval_nn()  →  accuracy, F1, precision, recall      │
  │  NoCMapper.run(net.graph)                           │
  │    indegree_grouping → WCTG → mesh → SA             │
  │    cost = Σ WCTG[i,j] × hops(core_i, core_j)        │
  │  build_wctg_structural() → sp_cost                  │
  │  record_stage("A: Comm. Cost Neuron + Bridge-Link   │
  │               [final]")                             │
  │  acc=0.9728 (−0.96%), cost=1,530 (−97.72%)          │
  │  neurons=921 (−22.86%), edges=1,605 (−99.40%)       │
  │  FLOPs=123,428 (−76.82%), inf=0.10ms (−59.13%)      │
  └─────────────────────────────────────────────────────┘
            │
            ▼
   trajectory saved per micro-step
   (acc, drop, neurons, edges,
    cross_group_edges, cross_group_w_sum, flops)



###################################################################
#                                                                 #
#              FULL FLOW INCLUDING ALL BRANCHES                   #
#                                                                 #
###################################################################
    main_LeNet_300_100_e2e_AccuracyBudget.py — Full Branch Flow
============================================================
Model  : LeNet 784-300-100-10  (MNIST)
Branches: 
  BRANCH A: Comm. Cost Neuron + Bridge-Link
  BRANCH B: Comm. Cost Neuron + L2 Edge 
  BRANCH C: Comm. Cost Neuron + L1 Edge
  BRANCH D: Taylor Neuron + L2 Edge  
  BRANCH E: Taylor Neuron + L1 Edge
  BRANCH F: L1 Neuron + L2 edge - Not Tested due to not being best or 2nd best in individual comparisons
  BRANCH G: Taylor Neuron + Bridge-Link 
Key globals
  ACCURACY_TOLERANCE      = 0.01   (1.0%)
  SNAPSHOT_INTERVAL       = 0.005  (0.5%)
  MICRO_FT_EPOCHS         = 5      (overridden per branch where noted)
  NEURON_STEPS_PER_EDGE   = 3      (overridden per branch where noted)
  NEURONS_PER_STEP        = 3      (default; overridden per branch)
  EDGE_STEP_FRACTION      = 0.02   (2%  — L1 / L2 methods)
  BRIDGE_STEP_FRACTION    = 0.02   (2%  — granular cross-group pruner)
  MAX_STEPS               = 1000
  P = N = 36 cores, 6×6 mesh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SHARED PIPELINE  (steps 1–4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LeNet 784-300-100-10 weights (.pth)
            │
            ▼
    load_or_train()
    MNIST, seed=42, batch=64
            │
            ▼
  ┌─────────────────────────────┐
  │        BASELINE             │
  │  construct_nn_graph()       │  nodes=(layer,neuron), edges=non-zero weights
  │  NoCMapper.run()            │  indegree_grouping → WCTG → SA → cost
  │  record_stage("Baseline")   │  acc=0.9822, cost=66,976
  └─────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────┐
  │   SHARED L1 PRE-PRUNING     │
  │   5 × 5% iterative L1       │  remove 25% weakest-magnitude edges
  │   + fine-tune 10 epochs     │  globally, in equal steps
  │   commit + remove lonely    │
  │   record("After L1 25%")    │  acc=0.9758, cost=52,293
  └─────────────────────────────┘
            │
            ▼
   deepcopy → l1_model_copy          ← checkpoint all branches start from
   anchor_accuracy = 0.9822          ← BASELINE accuracy (not L1)
   floor = 0.9822 − 0.01 = 0.9722
   snapshots at drops: [0.5%, 1.0%]
            │
            ├──────────────────────────────────────────────────────────────────┐
            │                         FORK TO BRANCHES                         │
            │   A            B            C            D            E       G  │
            └──────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BRANCH A  ·  Comm. Cost Neuron + Bridge-Link        [PROPOSED]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

╔═══════════════════════════════════════════════════════════╗
║         BRANCH A: Comm. Cost Neuron + Bridge-Link         ║
║  neuron_tolerance=0.5%  |  full budget=1.0%               ║
║  neurons_per_step=3     |  neuron:edge ratio = 1:1        ║
║  micro_ft=2 epochs      |  bridge_step=2%  post=5%        ║
╚═══════════════════════════════════════════════════════════╝
            │
            ▼
  net = PrunableNetwork(deepcopy(l1_model_copy))
  best_model = current model
            │
            ▼
  ┌─────────────────────────────────────────────────────┐
  │          INTERLEAVED LOOP  (max 1000 steps)         │
  │                                                     │
  │  decide: neuron step or edge step? (1:1 ratio)      │
  │  save pre_step_model (for revert)                   │
  │                                                     │
  │  ── NEURON STEP (while drop < 0.5%) ──              │
  │  iterative_commcost_neuron_pruning(max_neurons=3)   │
  │    loop per neuron:                                 │
  │      G = net.graph  (always fresh)                  │
  │      indegree_grouping(G, N=36)  → groups           │
  │      select_neurons_high_indegree_low_magnitude()   │
  │        Stage-1: bottom 90% magnitude AND            │
  │                 bottom 25% in-degree quantile       │
  │      score each candidate:                          │
  │        Σ |w| on cross-group edges in/out            │
  │      remove highest-scoring neuron                  │
  │      net.prune_neurons() → rebuilds graph           │
  │    repeat × 3 neurons max                           │
  │  net.remove_lonely_neurons()                        │
  │                                                     │
  │  ── EDGE STEP ──                                    │
  │  _bridge_edge_step(amount = 2% of active edges)     │
  │    eval_noc() → fresh groups                        │
  │    collect all cross-group (bridge) edges           │
  │    sort by |weight| ascending                       │
  │    zero cheapest N bridge edges                     │
  │    fallback to L2 if no bridge edges remain         │
  │    net.remove_lonely_neurons()                      │
  │    net.commit_pruning() + resync_graph()            │
  │                                                     │
  │  reapply_zero_masks()                               │
  │  fine_tune(2 epochs)                                │
  │  commit_pruning() + resync_graph()                  │
  │  evaluate → acc, drop = anchor − acc                │
  │                                                     │
  │  if drop ≤ 1.0%:                                    │
  │    best_model = current                             │
  │    if drop ≥ 0.5%: end neuron phase →               │
  │         switch to edge-only at 5%/step              │
  │    if drop crosses 0.5% threshold → SNAPSHOT        │
  │    if drop crosses 1.0% threshold → SNAPSHOT        │
  │  else:                                              │
  │    revert to best_model                             │
  │    STOP                                             │
  └─────────────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────┐
  │  SNAPSHOT @ drop ≤ 0.5%     │  first crossing only
  │  NoCMapper.run(net.graph)   │  full SA mapping
  │  record_stage()             │
  └─────────────────────────────┘
            │  (loop continues, edge-only phase)
            ▼
  ┌─────────────────────────────┐
  │  SNAPSHOT @ drop ≤ 1.0%     │
  │  NoCMapper.run(net.graph)   │
  │  record_stage()             │
  └─────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────────────┐
  │  FINAL RECORD                                       │
  │  eval_nn()  →  accuracy, F1, precision, recall      │
  │  NoCMapper.run(net.graph)                           │
  │    indegree_grouping → WCTG → mesh → SA             │
  │    cost = Σ WCTG[i,j] × hops(core_i, core_j)        │
  │  build_wctg_structural() → sp_cost                  │
  │  record_stage("A: Comm. Cost Neuron + Bridge-Link   │
  │               [final]")                             │
  │  acc=0.9728 (−0.96%), cost=1,530 (−97.72%)          │
  │  neurons=921 (−22.86%), edges=1,605 (−99.40%)       │
  │  FLOPs=123,428 (−76.82%), inf=0.10ms (−59.13%)      │
  └─────────────────────────────────────────────────────┘
            │
            ▼
   trajectory saved per micro-step
   (acc, drop, neurons, edges,
    cross_group_edges, cross_group_w_sum, flops)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BRANCH B  ·  Comm. Cost Neuron + L2 Edge            [PROPOSED]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

╔═══════════════════════════════════════════════════════════╗
║         BRANCH B: Comm. Cost Neuron + L2 Edge             ║
║  neuron_tolerance=0.5%  |  full budget=1.0%               ║
║  neurons_per_step=3     |  neuron:edge ratio = 1:1        ║
║  micro_ft=2 epochs      |  L2_step=2%      post=5%        ║
╚═══════════════════════════════════════════════════════════╝
            │
            ▼
  net = PrunableNetwork(deepcopy(l1_model_copy))
  best_model = current model
            │
            ▼
  ┌─────────────────────────────────────────────────────┐
  │          INTERLEAVED LOOP  (max 1000 steps)         │
  │                                                     │
  │  decide: neuron step or edge step? (3:1 ratio)      │
  │    every 4th step is an edge step;                  │
  │    steps 1,2,3 = neuron; step 4 = edge; repeat      │
  │  save pre_step_model (for revert)                   │
  │                                                     │
  │  ── NEURON STEP (while drop < 0.5%) ──              │
  │  iterative_commcost_neuron_pruning(max_neurons=3)   │
  │    (identical selection logic to Branch A)          │
  │    G = net.graph  (always fresh)                    │
  │    indegree_grouping(G, N=36)  → groups             │
  │    Stage-1 filter: bottom 90% magnitude AND         │
  │                    bottom 25% in-degree quantile    │
  │    score: Σ |w| on cross-group edges in/out         │
  │    remove highest-scoring neuron × 3 max            │
  │    net.prune_neurons() → rebuilds graph             │
  │  net.remove_lonely_neurons()                        │
  │                                                     │
  │  ── EDGE STEP ──                                    │
  │  _l2_edge_step(amount = cumulative fraction)        │
  │    target_edges = current_edges − 2% of active      │
  │    amount = (already_zero + to_prune) / total_w     │
  │    prune_l2_global(model, amount):                  │
  │      score each weight by w²                        │
  │      zero the n_prune_active smallest-score edges   │
  │      (tie-breaking: second pass for exact count)    │
  │    net.commit_pruning() + resync_graph()            │
  │                                                     │
  │  reapply_zero_masks()                               │
  │  fine_tune(2 epochs)                                │
  │  commit_pruning() + resync_graph()                  │
  │  evaluate → acc, drop = anchor − acc                │
  │                                                     │
  │  if drop ≤ 1.0%:                                    │
  │    best_model = current                             │
  │    if drop ≥ 0.5%: end neuron phase →               │
  │         switch to edge-only at 5%/step              │
  │    if drop crosses 0.5% threshold → SNAPSHOT        │
  │    if drop crosses 1.0% threshold → SNAPSHOT        │
  │  else:                                              │
  │    revert to best_model                             │
  │    STOP                                             │
  └─────────────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────┐
  │  SNAPSHOT @ drop ≤ 0.5%     │
  │  NoCMapper.run(net.graph)   │
  │  record_stage()             │
  └─────────────────────────────┘
            │  (loop continues, edge-only phase)
            ▼
  ┌─────────────────────────────┐
  │  SNAPSHOT @ drop ≤ 1.0%     │
  │  NoCMapper.run(net.graph)   │
  │  record_stage()             │
  └─────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────────────┐
  │  FINAL RECORD                                       │
  │  eval_nn()  →  accuracy, F1, precision, recall      │
  │  NoCMapper.run(net.graph)                           │
  │ record_stage("B: CommCost Neuron + L2 edge [final]")│
  └─────────────────────────────────────────────────────┘
            │
            ▼
   trajectory saved per micro-step

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BRANCH C  ·  Comm. Cost Neuron + L1 Edge            [PROPOSED]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

╔═══════════════════════════════════════════════════════════╗
║         BRANCH C: Comm. Cost Neuron + L1 Edge             ║
║  neuron_tolerance=0.5%  |  full budget=1.0%               ║
║  neurons_per_step=3     |  neuron:edge ratio = 1:1        ║
║  micro_ft=2 epochs      |  L1_step=2%      post=5%        ║
╚═══════════════════════════════════════════════════════════╝
            │
            ▼
  net = PrunableNetwork(deepcopy(l1_model_copy))
  best_model = current model
            │
            ▼
  ┌─────────────────────────────────────────────────────┐
  │          INTERLEAVED LOOP  (max 1000 steps)         │
  │                                                     │
  │  decide: neuron step or edge step? (3:1 ratio)      │
  │    every 4th step is an edge step;                  │
  │    steps 1,2,3 = neuron; step 4 = edge; repeat      │
  │  save pre_step_model (for revert)                   │
  │                                                     │
  │  ── NEURON STEP (while drop < 0.5%) ──              │
  │  iterative_commcost_neuron_pruning(max_neurons=3)   │
  │    (identical selection logic to Branches A & B)    │
  │    G = net.graph  (always fresh)                    │
  │    indegree_grouping(G, N=36)  → groups             │
  │    Stage-1 filter: bottom 90% magnitude AND         │
  │                    bottom 25% in-degree quantile    │
  │    score: Σ |w| on cross-group edges in/out         │
  │    remove highest-scoring neuron × 3 max            │
  │    net.prune_neurons() → rebuilds graph             │
  │  net.remove_lonely_neurons()                        │
  │                                                     │
  │  ── EDGE STEP ──                                    │
  │  _l1_edge_step(amount = cumulative fraction)        │
  │    target_edges = current_edges − 2% of active      │
  │    amount = (already_zero + to_prune) / total_w     │
  │    PrunableNetwork(model).prune_links_l1_global():  │
  │      score each weight by |w|                       │
  │      zero the n_prune_active smallest-score edges   │
  │    net.commit_pruning() + resync_graph()            │
  │                                                     │
  │  reapply_zero_masks()                               │
  │  fine_tune(2 epochs)                                │
  │  commit_pruning() + resync_graph()                  │
  │  evaluate → acc, drop = anchor − acc                │
  │                                                     │
  │  if drop ≤ 1.0%:                                    │
  │    best_model = current                             │
  │    if drop ≥ 0.5%: end neuron phase →               │
  │         switch to edge-only at 5%/step              │
  │    if drop crosses 0.5% threshold → SNAPSHOT        │
  │    if drop crosses 1.0% threshold → SNAPSHOT        │
  │  else:                                              │
  │    revert to best_model                             │
  │    STOP                                             │
  └─────────────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────┐
  │  SNAPSHOT @ drop ≤ 0.5%     │
  │  NoCMapper.run(net.graph)   │
  │  record_stage()             │
  └─────────────────────────────┘
            │  (loop continues, edge-only phase)
            ▼
  ┌─────────────────────────────┐
  │  SNAPSHOT @ drop ≤ 1.0%     │
  │  NoCMapper.run(net.graph)   │
  │  record_stage()             │
  └─────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────────────┐
  │  FINAL RECORD                                       │
  │  eval_nn()  →  accuracy, F1, precision, recall      │
  │  NoCMapper.run(net.graph)                           │
  │ record_stage("C: CommCost Neuron + L1 edge [final]")│
  └─────────────────────────────────────────────────────┘
            │
            ▼
   trajectory saved per micro-step

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BRANCH D  ·  Taylor Neuron + L2 Edge             [CLASSICAL]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

╔═══════════════════════════════════════════════════════════╗
║         BRANCH D: Taylor Neuron + L2 Edge                 ║
║  neuron_tolerance=none  |  full budget=1.0% shared        ║
║  neurons_per_step=1     |  neuron:edge ratio = 1:1        ║
║  micro_ft=5 epochs      |  L2_step=2%     post=5%         ║
╚═══════════════════════════════════════════════════════════╝
            │
            ▼
  net = PrunableNetwork(deepcopy(l1_model_copy))
  best_model = current model
            │
            ▼
  ┌─────────────────────────────────────────────────────┐
  │          INTERLEAVED LOOP  (max 1000 steps)         │
  │                                                     │
  │  decide: neuron step or edge step? (1:1 ratio)      │
  │  save pre_step_model (for revert)                   │
  │                                                     │
  │  ── NEURON STEP (while drop < 1.0%, no split) ──    │
  │  _taylor_neuron_step(max_neurons=1)                 │
  │    fraction = 1 / total_hidden_neurons              │
  │    prune_neurons_taylor(model, fraction):           │
  │      for 10 mini-batches of 128 samples:            │
  │        forward → loss → backward                    │
  │        importance[layer] += |w * grad|.sum(dim=1)   │
  │      importance /= n_batches  (normalise)           │
  │      all_scores = concat all layer importances      │
  │      n_prune = max(1, round(fraction × total))      │
  │      threshold = topk(scores, n_prune, smallest)    │
  │      zero rows (neurons) below threshold            │
  │    physically_remove_zeroed_neurons(model)          │
  │      shrink Linear layers, fix successor in-dim     │
  │    net.commit_pruning() + resync_graph()            │
  │                                                     │
  │  ── EDGE STEP ──                                    │
  │  _l2_edge_step(amount = cumulative fraction)        │
  │    target_edges = current_edges − 2% of active      │
  │    amount = (already_zero + to_prune) / total_w     │
  │    prune_l2_global(model, amount):                  │
  │      score each weight by w²                        │
  │      zero the n_prune_active smallest-score edges   │
  │    net.commit_pruning() + resync_graph()            │
  │                                                     │
  │  reapply_zero_masks()                               │
  │  fine_tune(5 epochs)                                │
  │  commit_pruning() + resync_graph()                  │
  │  evaluate → acc, drop = anchor − acc                │
  │                                                     │
  │  if drop ≤ 1.0%:                                    │
  │    best_model = current                             │
  │    [no neuron split — both neuron & edge continue   │
  │     until the full 1.0% budget is consumed]         │
  │    if drop crosses 0.5% threshold → SNAPSHOT        │
  │    if drop crosses 1.0% threshold → SNAPSHOT        │
  │  else:                                              │
  │    revert to best_model                             │
  │    STOP                                             │
  └─────────────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────┐
  │  SNAPSHOT @ drop ≤ 0.5%     │
  │  NoCMapper.run(net.graph)   │
  │  record_stage()             │
  └─────────────────────────────┘
            │  (loop continues; post-neuron edge phase at 5%/step
            │   once neurons are exhausted or budget fully spent)
            ▼
  ┌─────────────────────────────┐
  │  SNAPSHOT @ drop ≤ 1.0%     │
  │  NoCMapper.run(net.graph)   │
  │  record_stage()             │
  └─────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────────────┐
  │  FINAL RECORD                                       │
  │  eval_nn()  →  accuracy, F1, precision, recall      │
  │  NoCMapper.run(net.graph)                           │
  │  record_stage("D: Taylor Neuron + L2 edge [final]") │
  └─────────────────────────────────────────────────────┘
            │
            ▼
   trajectory saved per micro-step

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BRANCH E  ·  Taylor Neuron + L1 Edge             [CLASSICAL]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

╔═══════════════════════════════════════════════════════════╗
║         BRANCH E: Taylor Neuron + L1 Edge                 ║
║  neuron_tolerance=none  |  full budget=1.0% shared        ║
║  neurons_per_step=1     |  neuron:edge ratio = 1:1        ║
║  micro_ft=5 epochs      |  L1_step=2%     post=5%         ║
╚═══════════════════════════════════════════════════════════╝
            │
            ▼
  net = PrunableNetwork(deepcopy(l1_model_copy))
  best_model = current model
            │
            ▼
  ┌─────────────────────────────────────────────────────┐
  │          INTERLEAVED LOOP  (max 1000 steps)         │
  │                                                     │
  │  decide: neuron step or edge step? (1:1 ratio)      │
  │  save pre_step_model (for revert)                   │
  │                                                     │
  │  ── NEURON STEP (while drop < 1.0%, no split) ──    │
  │  _taylor_neuron_step(max_neurons=1)                 │
  │    (identical Taylor criterion to Branch D)         │
  │    fraction = 1 / total_hidden_neurons              │
  │    for 10 mini-batches of 128:                      │
  │      importance += |w * grad|.sum(dim=1)            │
  │    importance /= n_batches                          │
  │    zero lowest-importance fraction of neurons       │
  │    physically_remove_zeroed_neurons(model)          │
  │    net.commit_pruning() + resync_graph()            │
  │                                                     │
  │  ── EDGE STEP ──                                    │
  │  _l1_edge_step(amount = cumulative fraction)        │
  │    target_edges = current_edges − 2% of active      │
  │    amount = (already_zero + to_prune) / total_w     │
  │    PrunableNetwork(model).prune_links_l1_global():  │
  │      score each weight by |w|                       │
  │      zero n_prune_active smallest-|w| edges         │
  │    net.commit_pruning() + resync_graph()            │
  │                                                     │
  │  reapply_zero_masks()                               │
  │  fine_tune(5 epochs)                                │
  │  commit_pruning() + resync_graph()                  │
  │  evaluate → acc, drop = anchor − acc                │
  │                                                     │
  │  if drop ≤ 1.0%:                                    │
  │    best_model = current                             │
  │    if drop crosses 0.5% threshold → SNAPSHOT        │
  │    if drop crosses 1.0% threshold → SNAPSHOT        │
  │  else:                                              │
  │    revert to best_model                             │
  │    STOP                                             │
  └─────────────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────┐
  │  SNAPSHOT @ drop ≤ 0.5%     │
  │  NoCMapper.run(net.graph)   │
  │  record_stage()             │
  └─────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────┐
  │  SNAPSHOT @ drop ≤ 1.0%     │
  │  NoCMapper.run(net.graph)   │
  │  record_stage()             │
  └─────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────────────┐
  │  FINAL RECORD                                       │
  │  eval_nn()  →  accuracy, F1, precision, recall      │
  │  NoCMapper.run(net.graph)                           │
  │  record_stage("E: Taylor Neuron + L1 edge [final]") │
  └─────────────────────────────────────────────────────┘
            │
            ▼
   trajectory saved per micro-step

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BRANCH G  ·  Taylor Neuron + Bridge-Link         [CLASSICAL]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

╔═══════════════════════════════════════════════════════════╗
║         BRANCH G: Taylor Neuron + Bridge-Link             ║
║  neuron_tolerance=none  |  full budget=1.0% shared        ║
║  neurons_per_step=1     |  neuron:edge ratio = 1:1        ║
║  micro_ft=5 epochs      |  bridge_step=2%  post=5%        ║
╚═══════════════════════════════════════════════════════════╝
            │
            ▼
  net = PrunableNetwork(deepcopy(l1_model_copy))
  best_model = current model
            │
            ▼
  ┌─────────────────────────────────────────────────────┐
  │          INTERLEAVED LOOP  (max 1000 steps)         │
  │                                                     │
  │  decide: neuron step or edge step? (1:1 ratio)      │
  │  save pre_step_model (for revert)                   │
  │                                                     │
  │  ── NEURON STEP (while drop < 1.0%, no split) ──    │
  │  _taylor_neuron_step(max_neurons=1)                 │
  │    (identical Taylor criterion to Branches D & E)   │
  │    fraction = 1 / total_hidden_neurons              │
  │    for 10 mini-batches of 128:                      │
  │      importance += |w * grad|.sum(dim=1)            │
  │    importance /= n_batches                          │
  │    zero lowest-importance fraction of neurons       │
  │    physically_remove_zeroed_neurons(model)          │
  │    net.commit_pruning() + resync_graph()            │
  │                                                     │
  │  ── EDGE STEP ──                                    │
  │  _bridge_edge_step(amount = 2% of active edges)     │
  │    eval_noc() → fresh NoC grouping                  │
  │    build node → group-id map                        │
  │    target: remove 2% of active edges this step      │
  │    collect all cross-group (bridge) edges           │
  │      (src_grp ≠ dst_grp, weight ≠ 0)                │
  │    sort by |weight| ascending                       │
  │    zero cheapest N bridge edges                     │
  │    fallback to L2 if no bridge edges remain         │
  │    net.remove_lonely_neurons()                      │
  │    net.commit_pruning() + resync_graph()            │
  │                                                     │
  │  reapply_zero_masks()                               │
  │  fine_tune(5 epochs)                                │
  │  commit_pruning() + resync_graph()                  │
  │  evaluate → acc, drop = anchor − acc                │
  │                                                     │
  │  if drop ≤ 1.0%:                                    │
  │    best_model = current                             │
  │    [no neuron split — Taylor and bridge steps both  │
  │     continue until the 1.0% budget is consumed]     │
  │    if drop crosses 0.5% threshold → SNAPSHOT        │
  │    if drop crosses 1.0% threshold → SNAPSHOT        │
  │  else:                                              │
  │    revert to best_model                             │
  │    STOP                                             │
  └─────────────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────┐
  │  SNAPSHOT @ drop ≤ 0.5%     │
  │  NoCMapper.run(net.graph)   │
  │  record_stage()             │
  └─────────────────────────────┘
            │  (loop continues; post-neuron edge phase at 5%/step
            │   once Taylor neurons are exhausted)
            ▼
  ┌─────────────────────────────┐
  │  SNAPSHOT @ drop ≤ 1.0%     │
  │  NoCMapper.run(net.graph)   │
  │  record_stage()             │
  └─────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────────────┐
  │  FINAL RECORD                                       │
  │  eval_nn()  →  accuracy, F1, precision, recall      │
  │  NoCMapper.run(net.graph)                           │
  │  record_stage("G: Taylor Neuron + Bridge-Link       │
  │               [final]")                             │
  └─────────────────────────────────────────────────────┘
            │
            ▼
   trajectory saved per micro-step

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY  (step 12)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌──────────────────────────────────────────────────────┐
  │  print_summary_tables()                              │
  │                                                      │
  │  TABLE 1 — ISO-ACCURACY                              │
  │    rows: all "@drop≤X%" snapshot records             │
  │    one row per branch × threshold                    │
  │    columns: Accuracy, WCTG n2n cost, Cross-Group     │
  │             cost, Edges, Neurons, Inf.(ms),          │
  │             FLOPs, Size(KB), F1, Precision, Recall   │
  │    deltas relative to Baseline                       │
  │                                                      │
  │  TABLE 2 — FINAL                                     │
  │    rows: Baseline, After-L1-25%, A[final]…G[final]   │
  │    same columns as Table 1                           │
  │    deltas relative to Baseline                       │
  └──────────────────────────────────────────────────────┘
            │
            ▼
  save_trajectories(
    path_csv  = "trajectory_LeNet300_100.csv",
    path_json = "trajectory_LeNet300_100.json"
  )
  columns: branch, step, phase, accuracy, accuracy_drop,
           neurons, edges, cross_group_edges,
           cross_group_w_sum, flops

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BRANCH PARAMETER SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Branch │ Neuron fn          │ Edge fn          │ neuron_tol │ ratio │ n/step │ FT ep │ edge% │ post%
 ───────┼────────────────────┼──────────────────┼────────────┼───────┼────────┼───────┼───────┼──────
   A    │ CommCost           │ Bridge-Link      │   0.5%     │  1:1  │   3    │   2   │  2%   │  5%
   B    │ CommCost           │ L2               │   0.5%     │  1:1  │   3    │   2   │  2%   │  5%
   C    │ CommCost           │ L1               │   0.5%     │  1:1  │   3    │   2   │  2%   │  5%
   D    │ Taylor             │ L2               │   none     │  1:1  │   1    │   5   │  2%   │  5%
   E    │ Taylor             │ L1               │   none     │  1:1  │   1    │   5   │  2%   │  5%
   G    │ Taylor             │ Bridge-Link      │   none     │  1:1  │   1    │   5   │  2%   │  5%

 CommCost = iterative_commcost_neuron_pruning
 Taylor = _taylor_neuron_step  (10 mini-batches, 128 samples, |w·grad| importance)
 Bridge-Link = _bridge_edge_step  (cross-group edges sorted by |w|; L2 fallback)
 L2 = _l2_edge_step  (global w² ranking)
 L1 = _l1_edge_step  (global |w| ranking)
 neuron_tol: accuracy drop cap for neuron phase; remaining headroom goes to edge-only
 ratio: neuron steps per edge step in interleaved loop (3:1 → 3 neuron, 1 edge per cycle)
 edge%: fraction of active edges removed per edge micro-step (interleaved phase)
 post%: fraction of active edges removed per edge micro-step (post-neuron edge-only phase)
 Branch F (L1 Neuron + L2 Edge) is commented out and not executed.
