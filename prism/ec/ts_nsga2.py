"""
TS-NSGA-II: F. Ming, W. Gong and L. Wang, "A Two-Stage Evolutionary Algorithm With Balanced Convergence and
Diversity for Many-Objective Optimization", IEEE Trans. Systems, Man, and Cybernetics: Systems, 52(10):6222-6234, 2022.

Line-by-line port of the reference implementation written by the first author for PlatEMO
(BIMK/PlatEMO, Algorithms/Multi-objective optimization/TS-NSGA-II: TSNSGAII.m, EnvironmentalSelection.m,
EnvironmentalSelection1.m, SPDSort.m, LevelSort.m, DensityEstimate.m).

Stage 1 (FE < (0.8 + 0.2 (1 - 3/M)) maxFE): min-max normalization of the merged population, association with the
reference vector of smallest perpendicular distance d2, d1 = d2 = 0 for the extreme non-dominated solutions, SPD
sorting (Pareto dominance; for non-dominated pairs in the same sub-region, the smaller PBI = d1 + 5 d2 wins), last
front truncated by ascending d2. Mating: binary tournament on front number, then d2.
Stage 2: Pareto fronts merged until >= N, ordered by LevelSort, then for each reference vector the remaining
solution with the largest cosine (after min-max normalization of the remaining set) is selected. Mating: front
number from non-dominated sorting, then sub-region density.
Two guards stop loops that never terminate in the original code: the level is capped at n (relevant when the
range of an objective underflows to zero), and a sorting pass that assigns no solution places the rest in one front.
"""
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

EPS = 1e-12


def _minmax(F):
    zmin, zmax = F.min(axis=0), F.max(axis=0)
    return (F - zmin) / np.maximum(zmax - zmin, EPS)


def _cosine(A, W):
    """1 - pdist2(A, W, 'cosine')."""
    na = np.maximum(np.linalg.norm(A, axis=1, keepdims=True), EPS)
    nw = np.maximum(np.linalg.norm(W, axis=1, keepdims=True), EPS)
    return (A / na) @ (W / nw).T


def _pareto_matrix(F):
    """R[i, j] = 1: i Pareto-dominates j; -1: j dominates i; 0: non-dominated or equal."""
    less = (F[:, None, :] < F[None, :, :])
    more = (F[:, None, :] > F[None, :, :])
    any_less, any_more = less.any(axis=2), more.any(axis=2)
    R = np.zeros((len(F), len(F)), dtype=np.int8)
    R[any_less & ~any_more] = 1
    R[any_more & ~any_less] = -1
    return R


def spd_sort(PopObj, d1, d2, RP, n_sort):
    """SPDSort.m: returns (FrontNo, MaxFNo); unsorted solutions keep FrontNo = inf. Same comparison order and early exit as the original."""
    N = len(PopObj)
    R = _pareto_matrix(PopObj)
    pbi = d1 + 5.0 * d2
    same = RP[:, None] == RP[None, :]
    tie = (R == 0) & same
    R = R.copy()
    R[tie & (pbi[:, None] < pbi[None, :])] = 1
    R[tie & (pbi[:, None] > pbi[None, :])] = -1
    FrontNo = np.full(N, np.inf)
    MaxFNo = 0
    while np.sum(FrontNo < np.inf) < min(n_sort, N):
        MaxFNo += 1
        D = FrontNo != np.inf
        for i in range(N):
            if D[i]:
                continue
            cand = np.flatnonzero(~D[i + 1:]) + i + 1
            r = R[i, cand]
            neg = np.flatnonzero(r == -1)
            if len(neg):
                stop = neg[0]
                D[cand[:stop][r[:stop] == 1]] = True
                D[i] = True
            else:
                D[cand[r == 1]] = True
            if not D[i]:
                FrontNo[i] = MaxFNo
        if not np.any(FrontNo == MaxFNo):  # guard: a pass that assigns nobody (dominance cycle) would loop forever in the original
            FrontNo[FrontNo == np.inf] = MaxFNo
    return FrontNo, MaxFNo


def environmental_selection(F, W, N):
    """EnvironmentalSelection.m: returns (selected indices, their FrontNo, their d2)."""
    P = _minmax(F)
    normP = np.linalg.norm(P, axis=1)
    cos = np.clip(_cosine(P, W), -1.0, 1.0)
    d1_all = normP[:, None] * cos
    d2_all = normP[:, None] * np.sqrt(np.maximum(1.0 - cos ** 2, 0.0))
    RP = np.argmin(d2_all, axis=1)
    d2 = d2_all[np.arange(len(P)), RP]
    d1 = d1_all[np.arange(len(P)), RP]
    ND = NonDominatedSorting().do(P, only_non_dominated_front=True)
    ext = ND[np.argmax(P[ND], axis=0)]
    d1[ext] = 0.0
    d2[ext] = 0.0
    FrontNo, MaxFNo = spd_sort(P, d1, d2, RP, N)
    nxt = FrontNo < MaxFNo
    last = np.flatnonzero(FrontNo == MaxFNo)
    order = np.argsort(d2[last], kind="stable")
    nxt[last[order[: N - int(nxt.sum())]]] = True
    idx = np.flatnonzero(nxt)
    return idx, FrontNo[idx], d2[idx]


def level_sort(F, n):
    """LevelSort.m: level t = smallest t >= 1 with every objective <= Zmin + (t + 1) * interval."""
    zmax, zmin = F.max(axis=0), F.min(axis=0)
    interval = (zmax - zmin) / n
    level = np.zeros(len(F), dtype=int)
    for i, obj in enumerate(F):
        t = 0
        while True:
            t += 1
            # guard: t never exceeds n in exact arithmetic; the original loops forever if the interval underflows to 0
            if np.all(obj <= zmin + (t + 1) * interval) or t >= n:
                break
        level[i] = t
    return level


def density_estimate(F, W):
    """DensityEstimate.m: number of solutions in the sub-region (largest cosine) of each solution."""
    region = np.argmax(_cosine(_minmax(F), W), axis=1)
    return np.bincount(region, minlength=len(W))[region].astype(float)


def environmental_selection1(F, W, N, M):
    """EnvironmentalSelection1.m: returns the selected indices in the order of the reference vectors."""
    fronts = NonDominatedSorting().do(F)
    pool = []
    for fr in fronts:
        pool.extend(fr.tolist())
        if len(pool) >= N:
            break
    pool = np.array(pool, dtype=int)
    n = 2 * M
    lev = level_sort(F[pool], n)
    # levels 1..n are concatenated; in the original code the early exit never triggers
    pool = np.concatenate([pool[lev == i] for i in range(1, n + 1)])
    selected = []
    remaining = list(pool)
    for w in W:
        if not remaining:
            break
        S = _minmax(F[remaining])
        k = int(np.argmax(_cosine(S, w[None, :])[:, 0]))
        selected.append(remaining.pop(k))
    return np.array(selected, dtype=int)


class TSSurvival(Survival):
    def __init__(self, ref_dirs, max_eval):
        super().__init__(filter_infeasible=True)
        self.W = ref_dirs
        self.max_eval = max(int(max_eval), 1)

    def _do(self, problem, pop, *args, n_survive=None, algorithm=None, **kwargs):
        F = pop.get("F").astype(float, copy=False)
        M = F.shape[1]
        fe = algorithm.evaluator.n_eval if algorithm is not None and algorithm.evaluator is not None else 0
        if fe < (0.8 + 0.2 * (1 - 3 / M)) * self.max_eval:
            idx, front_no, d2 = environmental_selection(F, self.W, n_survive)
        else:
            idx = environmental_selection1(F, self.W, n_survive, M)
            sub = F[idx]
            front_no = np.empty(len(idx))
            for r, fr in enumerate(NonDominatedSorting().do(sub)):
                front_no[fr] = r + 1
            d2 = density_estimate(sub, self.W)
        out = pop[idx]
        # binary tournament (comp_by_rank_and_crowding): lower rank wins, then larger crowding, so -d2 is stored;
        # pymoo takes rank == 0 as the final set, PlatEMO's FrontNo starts at 1, hence the shift
        for k, ind in enumerate(out):
            ind.set("rank", front_no[k] - 1)
            ind.set("crowding", -d2[k])
        return out


class TS_NSGA2(NSGA2):
    """Population size = number of reference vectors (PlatEMO: [W, Problem.N] = UniformPoint(N, M)); max_eval = maxFE."""

    def __init__(self, ref_dirs, max_eval, **kwargs):
        super().__init__(pop_size=len(ref_dirs), survival=TSSurvival(ref_dirs, max_eval), **kwargs)
        self.tournament_type = "comp_by_rank_and_crowding"
