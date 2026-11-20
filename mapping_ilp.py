import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB



def _communication_cost(mapped, spg_matrix, wctg_matrix, N, P):
    """
    Quadratic communication cost objective.

    Sum over all (task i, task j, core r, core s) of:
        spg[r][s] * wctg[i][j] * mapped[i][r] * mapped[j][s]
    """
    cost = gp.LinExpr()
    for i in range(N):
        for j in range(N):
            if wctg_matrix[i][j] == 0:      # skip zero-weight task pairs early
                continue
            for r in range(P):
                for s in range(P):
                    if spg_matrix[r][s] == 0:
                        continue
                    cost += spg_matrix[r][s] * wctg_matrix[i][j] * mapped[i, r] * mapped[j, s]
    return cost


def map_nodes(
    wctg_matrix: np.ndarray,
    spg_matrix: np.ndarray,
    P: int,
) -> dict:
    """
    Map WCTG tasks onto P processor cores minimising communication cost.

    Parameters
    ----------
    wctg_matrix : np.ndarray, shape (N, N)
        Weighted communication task graph adjacency matrix.
        wctg_matrix[i][j] = communication weight from task i to task j.
    spg_matrix : np.ndarray, shape (P, P)
        Shortest-path distance matrix of the mesh (from Floyd-Warshall).
        spg_matrix[r][s] = hop distance between core r and core s.
    P : int
        Number of available processor cores. Must be >= N.

    Returns
    -------
    dict with keys:
        'mapping'       : list of (task_id, core_id) pairs for the optimal solution.
        'cost'          : optimal objective value (communication cost).
        'mapped_matrix' : np.ndarray (N x P) binary assignment matrix.
        'solve_time'    : wall-clock seconds spent in Gurobi optimisation.
    """
    wctg_matrix = np.array(wctg_matrix, dtype=float)
    spg_matrix  = np.array(spg_matrix,  dtype=float)

    N = wctg_matrix.shape[0]    # number of tasks derived from WCTG

    if wctg_matrix.shape != (N, N):
        raise ValueError(f"wctg_matrix must be square, got shape {wctg_matrix.shape}")
    if spg_matrix.shape != (P, P):
        raise ValueError(f"spg_matrix must be ({P},{P}), got shape {spg_matrix.shape}")
    if P < N:
        raise ValueError(f"Need at least as many cores as tasks (P={P} < N={N})")

    print(f"Building model  [N={N} tasks, P={P} cores]")
    m = gp.Model("task-mapping")
    m.Params.OutputFlag = 1     # set to 0 to suppress Gurobi console output

    # --- Decision variables: mapped[i, r] = 1 iff task i is on core r ----------
    mapped = m.addMVar((N, P), vtype=GRB.BINARY, name="mapped")

    # --- Constraints -----------------------------------------------------------
    # Each task is assigned to exactly one core
    m.addConstrs(
        (mapped[i, :].sum() == 1 for i in range(N)),
        name="one_core_per_task",
    )
    # Each core hosts at most one task
    m.addConstrs(
        (mapped[:, r].sum() <= 1 for r in range(P)),
        name="one_task_per_core",
    )

    # --- Objective: minimise total communication cost --------------------------
    m.setObjective(_communication_cost(mapped, spg_matrix, wctg_matrix, N, P), GRB.MINIMIZE)

    print("Model built. Running optimisation...")
    t0 = time.time()
    m.optimize()
    solve_time = time.time() - t0

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not find an optimal solution (status={m.Status})")

    # --- Extract solution ------------------------------------------------------
    mapped_matrix = np.array(mapped.X, dtype=float).round().astype(int)
    mapping = [
        (task, core)
        for task in range(N)
        for core in range(P)
        if mapped_matrix[task, core] == 1
    ]

    print(f"\nOptimisation complete in {solve_time:.2f}s")
    print(f"Optimal communication cost: {m.ObjVal:.4f}")
    print("Mapping (task → core):")
    for task, core in mapping:
        print(f"  Task {task + 1:>3}  →  Core {core + 1:>3}")

    return {
        "mapping":       mapping,
        "cost":          m.ObjVal,
        "mapped_matrix": mapped_matrix,
        "solve_time":    solve_time,
    }