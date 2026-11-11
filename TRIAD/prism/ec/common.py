"""
Shared utilities for CSDR, TS-NSGA-II and RPS-NSGA-II: ideal/nadir normalisation, pairwise angles,
and non-dominated sorting from an arbitrary domination matrix. The sort guards against cycles:
the CSDR paper (Property 3) notes that the SDR/CSDR relation is not transitive, whereas the standard
fast non-dominated sort assumes a proper partial order and can loop forever on a cycle.
"""
import numpy as np
from pymoo.util.dominator import Dominator


def normalize_ideal_nadir(F, eps=1e-12):
    ideal = F.min(axis=0)
    nadir = F.max(axis=0)
    span = np.maximum(nadir - ideal, eps)
    return (F - ideal) / span


def pairwise_angles(Fn, eps=1e-12):
    """(n,n) angle matrix in radians; the diagonal is +inf so a point is never its own nearest neighbour."""
    norms = np.maximum(np.linalg.norm(Fn, axis=1), eps)
    cos = (Fn @ Fn.T) / np.outer(norms, norms)
    cos = np.clip(cos, -1.0, 1.0)
    theta = np.arccos(cos)
    np.fill_diagonal(theta, np.inf)
    return theta


def niche_threshold_angle(theta, a):
    """
    θ̄: the ⌊a·N/100⌋-th smallest (1-indexed) of the nearest-neighbour angles.
    a=50 gives ⌊N/2⌋, as in the original SDR definition.
    """
    n = theta.shape[0]
    nearest = theta.min(axis=1)
    idx = int(np.floor(a * n / 100))
    idx = min(max(idx, 1), n)
    theta_bar = float(np.sort(nearest)[idx - 1])
    # Duplicate or near-duplicate individuals drive the nearest angle to 0; theta_bar=0 would divide
    # by zero in the θ>θ̄ branch of CSDR, so a small lower bound is used.
    return max(theta_bar, 1e-9)


def pareto_relation_matrix(F):
    """1 = row dominates column, -1 = column dominates row, 0 = non-dominated (no constraints, n_constr=0)."""
    return Dominator.calc_domination_matrix(F)


def fast_non_dominated_sort_from_matrix(M):
    """
    Build non-dominated fronts from any (n,n) domination matrix (same convention as
    Dominator.calc_domination_matrix: M[i,j]=1 means i dominates j). Unlike pymoo's
    fast_non_dominated_sort, this handles domination cycles in which some individuals never reach a
    zero domination count: when a round finds no new zero-count individual, all remaining unranked
    individuals are merged into a final front.
    """
    n = M.shape[0]
    if n == 0:
        return []

    n_dominated = np.zeros(n, dtype=int)
    dominates = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] == 1:
                dominates[i].append(j)
                n_dominated[j] += 1
            elif M[i, j] == -1:
                dominates[j].append(i)
                n_dominated[i] += 1

    ranked = np.zeros(n, dtype=bool)
    current = [i for i in range(n) if n_dominated[i] == 0]
    for i in current:
        ranked[i] = True
    fronts = [np.array(current, dtype=int)]
    n_ranked = len(current)

    while n_ranked < n:
        nxt = []
        for i in current:
            for j in dominates[i]:
                if ranked[j]:
                    continue
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    nxt.append(j)
        if not nxt:
            leftover = np.array([i for i in range(n) if not ranked[i]], dtype=int)
            fronts.append(leftover)
            break
        for i in nxt:
            ranked[i] = True
        n_ranked += len(nxt)
        fronts.append(np.array(nxt, dtype=int))
        current = nxt

    return fronts


def rank_from_fronts(fronts, n):
    rank = np.full(n, len(fronts), dtype=int)
    for r, front in enumerate(fronts):
        rank[front] = r
    return rank


def associate_to_reference_points(Fn, ref_dirs, eps=1e-12):
    """Assign each (normalised) individual to the reference direction with the smallest angle (same argmin as 1-cos)."""
    fn_norms = np.maximum(np.linalg.norm(Fn, axis=1), eps)
    ref_norms = np.maximum(np.linalg.norm(ref_dirs, axis=1), eps)
    cos = (Fn @ ref_dirs.T) / np.outer(fn_norms, ref_norms)
    cos = np.clip(cos, -1.0, 1.0)
    angles = np.arccos(cos)
    return np.argmin(angles, axis=1)


def truncate_by_score(front, score, n_remove, prefer="largest", random_state=None):
    """
    Truncate a front by score, removing the n_remove worst individuals; returns the kept indices.
    prefer="largest": larger is better (e.g. crowding distance), the n_remove smallest are removed;
    prefer="smallest": smaller is better (e.g. RPSet_Density or d2), the n_remove largest are removed.
    """
    from pymoo.util.randomized_argsort import randomized_argsort

    order_dir = "descending" if prefer == "largest" else "ascending"
    order = randomized_argsort(score, order=order_dir, method="numpy", random_state=random_state)
    keep = order[: len(front) - n_remove]
    return front[keep]
