"""
TS-NSGA-II -- Ming, Gong & Wang, "A Two-Stage Evolutionary Algorithm With
Balanced Convergence and Diversity for Many-Objective Optimization",
IEEE Trans. Systems, Man, and Cybernetics: Systems, 52(10):6222-6234, 2022.
DOI: 10.1109/TSMC.2022.3143657.

Two dominance relations, two-stage environmental selection and a dedicated mating selection
(Algorithms 1-4 of the paper):
  Stage 1 (convergence): SPD (sub-region PBI dominance) -- solutions in the same reference-direction
    sub-region are ordered by PBI, solutions in different sub-regions are incomparable; the last front is
    truncated by d2 (PBI perpendicular distance), smaller kept first.
  Stage 2 (diversity): LPD (level-based Pareto dominance) -- Pareto dominance first, ties between
    Pareto-equivalent solutions broken by grid level; selection finishes by picking one solution per
    reference direction.
  Stage switch: I_stage=1 if g < (1-3/(η·m))×G_max, otherwise 2 (η=5, the paper's best setting).

Note: line 23 of Algorithm 2 reads x_closest = argmin_x D(x,w) with D defined as cosine similarity;
taken literally, argmin picks the solution with the largest angle, contradicting "closest". This
implementation follows the intended meaning (smallest angle, i.e. largest cosine similarity) and treats
the pseudo-code as a typo.
"""
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.survival import Survival
from pymoo.operators.selection.tournament import TournamentSelection

from prism.ec.common import (
    associate_to_reference_points,
    fast_non_dominated_sort_from_matrix,
    normalize_ideal_nadir,
    pareto_relation_matrix,
    rank_from_fronts,
    truncate_by_score,
)

THETA_PBI = 5.0
ETA_STAGE = 5
N_P_LEVELS = None  # None means 2*n_obj, see TS_NSGA2.__init__


def _compute_pbi(Fn, assoc, ref_dirs, theta=THETA_PBI, eps=1e-12):
    w = ref_dirs[assoc]
    w_norm = np.maximum(np.linalg.norm(w, axis=1, keepdims=True), eps)
    w_unit = w / w_norm
    d1 = np.sum(Fn * w_unit, axis=1)
    d2 = np.linalg.norm(Fn - d1[:, None] * w_unit, axis=1)
    pbi = d1 + theta * d2
    return d1, d2, pbi


def _spd_fronts(assoc, pbi, n_ref):
    """Total order by ascending PBI within each sub-region, incomparable across sub-regions;
    global front k is the union of the k-th ranked members of all sub-regions."""
    groups = []
    for r in range(n_ref):
        idx = np.where(assoc == r)[0]
        if len(idx) == 0:
            continue
        groups.append(idx[np.argsort(pbi[idx])])
    if not groups:
        return []
    max_len = max(len(g) for g in groups)
    fronts = []
    for k in range(max_len):
        front = np.array([g[k] for g in groups if len(g) > k], dtype=int)
        fronts.append(front)
    return fronts


def _compute_levels(Fn, n_p):
    """Bin each objective independently (0..n_p-1); the level is the largest bin over objectives (larger is worse)."""
    bins = np.floor(np.clip(Fn, 0.0, 1.0 - 1e-9) * n_p).astype(int)
    return bins.max(axis=1)


def _lpd_fronts(F, level):
    n = F.shape[0]
    M = pareto_relation_matrix(F)
    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] != 0:
                continue
            if level[i] < level[j]:
                M[i, j], M[j, i] = 1, -1
            elif level[j] < level[i]:
                M[i, j], M[j, i] = -1, 1
    return fast_non_dominated_sort_from_matrix(M)


def compute_stage_state(F, ref_dirs, i_stage, n_p, theta=THETA_PBI):
    """
    Returns (rank, diversity_score, assoc).
    Stage 1: diversity_score = d2 (smaller is better); stage 2: diversity_score = sub-region density SD (smaller is better).
    """
    Fn = normalize_ideal_nadir(F)
    assoc = associate_to_reference_points(Fn, ref_dirs)

    if i_stage == 1:
        _, d2, pbi = _compute_pbi(Fn, assoc, ref_dirs, theta)
        fronts = _spd_fronts(assoc, pbi, len(ref_dirs))
        rank = rank_from_fronts(fronts, len(F)) if fronts else np.zeros(len(F), dtype=int)
        return rank, d2, assoc, fronts
    else:
        level = _compute_levels(Fn, n_p)
        fronts = _lpd_fronts(F, level)
        rank = rank_from_fronts(fronts, len(F))
        counts = np.bincount(assoc, minlength=len(ref_dirs))
        sd = counts[assoc].astype(float)
        return rank, sd, assoc, fronts


def current_stage(n_gen, max_gen, n_obj, eta=ETA_STAGE):
    threshold = (1 - 3 / (eta * n_obj)) * max_gen
    return 1 if n_gen < threshold else 2


class TSTournament:
    """Algorithm 3: compare front index (CM), then the diversity measure (DM, smaller is better), then pick at random."""

    def __init__(self, ref_dirs, max_gen, n_p, theta=THETA_PBI, eta=ETA_STAGE):
        self.ref_dirs = ref_dirs
        self.max_gen = max(int(max_gen), 1)
        self.n_p = n_p
        self.theta = theta
        self.eta = eta

    def __call__(self, pop, P, algorithm, **kwargs):
        F = pop.get("F").astype(float, copy=False)
        n_gen = getattr(algorithm, "n_gen", None) or 0
        n_obj = F.shape[1]
        i_stage = current_stage(n_gen, self.max_gen, n_obj, self.eta)
        rank, dm, _assoc, _fronts = compute_stage_state(F, self.ref_dirs, i_stage, self.n_p, self.theta)

        n_tournaments, n_parents = P.shape
        if n_parents != 2:
            raise ValueError("TSTournament only supports binary tournaments")
        S = np.full(n_tournaments, -1, dtype=int)
        for i in range(n_tournaments):
            a, b = P[i, 0], P[i, 1]
            if rank[a] < rank[b]:
                S[i] = a
            elif rank[b] < rank[a]:
                S[i] = b
            elif dm[a] < dm[b]:
                S[i] = a
            elif dm[b] < dm[a]:
                S[i] = b
            else:
                S[i] = a if np.random.rand() < 0.5 else b
        return S[:, None]


class TSSurvival(Survival):
    def __init__(self, ref_dirs, max_gen, n_p, theta=THETA_PBI, eta=ETA_STAGE):
        super().__init__(filter_infeasible=True)
        self.ref_dirs = ref_dirs
        self.max_gen = max(int(max_gen), 1)
        self.n_p = n_p
        self.theta = theta
        self.eta = eta

    def _do(self, problem, pop, *args, n_survive=None, algorithm=None, random_state=None, **kwargs):
        F = pop.get("F").astype(float, copy=False)
        n_gen = getattr(algorithm, "n_gen", None) or 0
        n_obj = F.shape[1]
        i_stage = current_stage(n_gen, self.max_gen, n_obj, self.eta)
        rank, diversity, assoc, fronts = compute_stage_state(F, self.ref_dirs, i_stage, self.n_p, self.theta)

        if i_stage == 1:
            # fill exactly n_survive; the last front is truncated by ascending d2 (Algorithm 2, lines 15-19)
            survivors = []
            for front in fronts:
                if len(survivors) + len(front) > n_survive:
                    n_remove = len(survivors) + len(front) - n_survive
                    front = truncate_by_score(front, diversity[front], n_remove, prefer="smallest", random_state=random_state)
                survivors.extend(front.tolist())
                if len(survivors) >= n_survive:
                    break
            for r, front in enumerate(fronts):
                for gi in front:
                    if int(gi) in survivors:
                        pop[int(gi)].set("rank", r)
                        pop[int(gi)].set("crowding", -diversity[int(gi)])
            return pop[survivors]

        # stage 2: accumulate whole fronts until size >= n_survive (may exceed it; Algorithm 2, lines 11-14)
        pool = []
        for front in fronts:
            pool.extend(front.tolist())
            if len(pool) >= n_survive:
                break
        pool = np.array(pool, dtype=int)

        # then pick the best-aligned solution from the pool for each reference direction until n_survive are chosen (Algorithm 2, lines 20-27)
        Fn_pool = normalize_ideal_nadir(F[pool])
        remaining_mask = np.ones(len(pool), dtype=bool)
        selected = []
        ref_norms = np.maximum(np.linalg.norm(self.ref_dirs, axis=1), 1e-12)
        for w, w_norm in zip(self.ref_dirs, ref_norms):
            if not remaining_mask.any() or len(selected) >= n_survive:
                break
            idx_remaining = np.where(remaining_mask)[0]
            fn_remaining = Fn_pool[idx_remaining]
            fn_norms = np.maximum(np.linalg.norm(fn_remaining, axis=1), 1e-12)
            cos = (fn_remaining @ w) / (fn_norms * w_norm)
            best_local = idx_remaining[np.argmax(cos)]  # argmax = best aligned, see module docstring
            selected.append(pool[best_local])
            remaining_mask[best_local] = False

        # fewer reference directions than remaining slots (cannot happen when |W| = n_survive); fill up
        if len(selected) < n_survive:
            leftover = pool[remaining_mask][: n_survive - len(selected)]
            selected.extend(leftover.tolist())

        for gi in selected:
            pop[int(gi)].set("rank", 0)
            pop[int(gi)].set("crowding", 0.0)
        return pop[selected]


class TS_NSGA2(NSGA2):
    def __init__(self, ref_dirs, max_gen=100, n_p=None, theta=THETA_PBI, eta=ETA_STAGE, **kwargs):
        n_obj_hint = ref_dirs.shape[1]
        n_p = n_p or (2 * n_obj_hint)
        pop_size = len(ref_dirs)
        selection = TournamentSelection(func_comp=TSTournament(ref_dirs, max_gen, n_p, theta, eta))
        survival = TSSurvival(ref_dirs, max_gen, n_p, theta, eta)
        super().__init__(pop_size=pop_size, selection=selection, survival=survival, **kwargs)
