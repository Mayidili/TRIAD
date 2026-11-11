"""
NSGA-II/CSDR -- Shen, Wang & Wang, "A Controlled Strengthened Dominance
Relation for Evolutionary Many-Objective Optimization", IEEE Trans.
Cybernetics, 52(5):3645-3657, 2022. DOI: 10.1109/TCYB.2020.3015998.

Only the comparison used by non-dominated sorting is replaced (CSDR instead of plain Pareto
dominance); crowding distance, truncation, variation and tournament selection are standard NSGA-II.

CSDR (Eq. 6/7 of the paper; the convergence exponent k and the niche percentage a are annealed
linearly over generations, Strategy-1 settings k_max=1.6, a_max=60, Δk=1.1, Δa=45, Tables II/III):
  x dominates y iff x Pareto-dominates y, or:
    Con(x) < Con(y)                      if θxy ≤ θ̄
    Con(x)·(θxy/θ̄) < Con(y)              if θxy > θ̄
  Con(x) = Σ f̃_i(x)^k  (f̃: objectives normalised by the current population's ideal/nadir)
  θ̄ = the ⌊a·N/100⌋-th smallest nearest-neighbour angle in the population
"""
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.survival import Survival
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function

from prism.ec.common import (
    fast_non_dominated_sort_from_matrix,
    niche_threshold_angle,
    normalize_ideal_nadir,
    pairwise_angles,
    pareto_relation_matrix,
    truncate_by_score,
)

K_MAX, DELTA_K = 1.6, 1.1
A_MAX, DELTA_A = 60.0, 45.0


def csdr_domination_matrix(F, k, a):
    n = F.shape[0]
    Fn = normalize_ideal_nadir(F)
    Con = (Fn ** k).sum(axis=1)
    theta = pairwise_angles(Fn)
    theta_bar = niche_threshold_angle(theta, a)

    M = pareto_relation_matrix(F)
    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] != 0:
                continue
            th = theta[i, j]
            if th <= theta_bar:
                i_dom_j = Con[i] < Con[j]
                j_dom_i = Con[j] < Con[i]
            else:
                i_dom_j = Con[i] * (th / theta_bar) < Con[j]
                j_dom_i = Con[j] * (th / theta_bar) < Con[i]
            if i_dom_j and not j_dom_i:
                M[i, j], M[j, i] = 1, -1
            elif j_dom_i and not i_dom_j:
                M[i, j], M[j, i] = -1, 1
    return M


class CSDRSurvival(Survival):
    def __init__(self, max_gen, crowding_func="cd"):
        super().__init__(filter_infeasible=True)
        self.max_gen = max(int(max_gen), 1)
        self.crowding_func = get_crowding_function(crowding_func)

    def _do(self, problem, pop, *args, n_survive=None, algorithm=None, random_state=None, **kwargs):
        F = pop.get("F").astype(float, copy=False)
        gen = getattr(algorithm, "n_gen", None) or 0
        progress = min(gen / self.max_gen, 1.0)
        k = K_MAX - DELTA_K * progress
        a = A_MAX - DELTA_A * progress

        M = csdr_domination_matrix(F, k, a)
        fronts = fast_non_dominated_sort_from_matrix(M)

        survivors = []
        for rank, front in enumerate(fronts):
            if len(survivors) + len(front) > n_survive:
                n_remove = len(survivors) + len(front) - n_survive
                crowding = self.crowding_func.do(F[front, :], n_remove=n_remove)
                front = truncate_by_score(front, crowding, n_remove, prefer="largest", random_state=random_state)
                crowding = self.crowding_func.do(F[front, :], n_remove=0)
            else:
                crowding = self.crowding_func.do(F[front, :], n_remove=0)

            for local_i, global_i in enumerate(front):
                pop[int(global_i)].set("rank", rank)
                pop[int(global_i)].set("crowding", crowding[local_i])
            survivors.extend(front.tolist())

        return pop[survivors]


class NSGA2_CSDR(NSGA2):
    """NSGA-II with CSDR survival; tournaments compare rank, then crowding (Algorithm 1 of the paper)."""

    def __init__(self, pop_size=50, max_gen=100, **kwargs):
        super().__init__(pop_size=pop_size, survival=CSDRSurvival(max_gen=max_gen), **kwargs)
        self.tournament_type = "comp_by_rank_and_crowding"
