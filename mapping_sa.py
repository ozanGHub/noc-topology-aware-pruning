import time
import numpy as np


def _compute_cost(assignment: np.ndarray, wctg: np.ndarray, spg: np.ndarray, N: int) -> float:
    """Total communication cost for a given task→core assignment."""
    cores = assignment[:N]
    # Vectorised: sum of wctg[i,j] * spg[cores[i], cores[j]] for all i,j
    return float(np.einsum("ij,ij->", wctg, spg[np.ix_(cores, cores)]))


def map_nodes(
    wctg_matrix: np.ndarray,
    spg_matrix: np.ndarray,
    P: int,
    *,
    max_iter: int = 200_000,
    t_start: float = 100.0,
    t_end: float = 0.01,
    seed: int | None = None,
) -> dict:
    """
    Map WCTG tasks onto P processor cores minimising communication cost
    using simulated annealing.

    Parameters
    ----------
    wctg_matrix : np.ndarray, shape (N, N)
        Weighted communication task graph adjacency matrix.
    spg_matrix : np.ndarray, shape (P, P)
        Shortest-path distance matrix of the mesh.
    P : int
        Number of available processor cores. Must be >= N.
    max_iter : int
        Number of SA iterations.
    t_start, t_end : float
        Start / end temperatures (geometric cooling).
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys: 'mapping', 'cost', 'mapped_matrix', 'solve_time'.
    """
    wctg = np.asarray(wctg_matrix, dtype=float)
    spg = np.asarray(spg_matrix, dtype=float)
    N = wctg.shape[0]

    if wctg.shape != (N, N):
        raise ValueError(f"wctg_matrix must be square, got shape {wctg.shape}")
    if spg.shape != (P, P):
        raise ValueError(f"spg_matrix must be ({P},{P}), got shape {spg.shape}")
    if P < N:
        raise ValueError(f"Need at least as many cores as tasks (P={P} < N={N})")

    rng = np.random.default_rng(seed)
    cooling = (t_end / t_start) ** (1.0 / max_iter)

    # ── Initial solution: random assignment (one task per core) ───────
    perm = rng.permutation(P)
    assignment = perm.copy()          # assignment[i] = core for task i (first N entries matter)

    cur_cost = _compute_cost(assignment, wctg, spg, N)
    best_assignment = assignment.copy()
    best_cost = cur_cost

    # ── Simulated annealing loop ─────────────────────────────────────
    print(f"Simulated annealing  [N={N}, P={P}, iters={max_iter:,}]")
    t0 = time.time()
    temp = t_start

    for it in range(max_iter):
        # Neighbour: swap two positions in the permutation.
        # At least one of the two must be an assigned task (index < N)
        # so we always change the mapping.
        a = rng.integers(N)           # always pick a task
        b = rng.integers(P)           # swap with any position
        if a == b:
            continue

        # Swap and evaluate
        assignment[a], assignment[b] = assignment[b], assignment[a]
        new_cost = _compute_cost(assignment, wctg, spg, N)
        delta = new_cost - cur_cost

        if delta < 0 or rng.random() < np.exp(-delta / temp):
            cur_cost = new_cost
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_assignment[:] = assignment
        else:
            assignment[a], assignment[b] = assignment[b], assignment[a]  # revert

        temp *= cooling

    solve_time = time.time() - t0

    # ── Build output ─────────────────────────────────────────────────
    mapped_matrix = np.zeros((N, P), dtype=int)
    mapping = []
    for task in range(N):
        core = int(best_assignment[task])
        mapped_matrix[task, core] = 1
        mapping.append((task, core))

    print(f"\nDone in {solve_time:.2f}s")
    print(f"Best communication cost: {best_cost:.4f}")
    print("Mapping (task → core):")
    for task, core in mapping:
        print(f"  Task {task + 1:>3}  →  Core {core + 1:>3}")

    return {
        "mapping":       mapping,
        "cost":          best_cost,
        "mapped_matrix": mapped_matrix,
        "solve_time":    solve_time,
    }