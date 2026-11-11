"""
RPS-NSGA-II -- Gu, Chen, Chen, Li & Xiong, "A many-objective evolutionary
algorithm with reference points-based strengthened dominance relation",
Information Sciences, 554:236-255, 2021. DOI: 10.1016/j.ins.2020.12.025.

Reference points split the population into sub-regions; Pareto-equivalent solutions are ranked by
convergence (Cov) and sub-region density (RPSet_Density) (Definition 1). The last front in environmental
selection is truncated by density instead of crowding (lower density is kept first), and the density of
extreme solutions in the first front is set to zero to protect boundary solutions (Section 3.3.2).
"""
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.survival import Survival

from prism.ec.common import (
    associate_to_reference_points,
    fast_non_dominated_sort_from_matrix,
    normalize_ideal_nadir,
    pareto_relation_matrix,
    truncate_by_score,
)


def rps_domination_matrix(F, ref_dirs):
    n = F.shape[0]
    Fn = normalize_ideal_nadir(F)
    assoc = associate_to_reference_points(Fn, ref_dirs)
    Cov = Fn.sum(axis=1)
    counts = np.bincount(assoc, minlength=len(ref_dirs))
    density = counts[assoc].astype(float)

    M = pareto_relation_matrix(F)
    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] != 0:
                continue
            if assoc[i] == assoc[j]:
                i_dom_j = Cov[i] < Cov[j]
                j_dom_i = Cov[j] < Cov[i]
            else:
                i_dom_j = (Cov[i] < Cov[j]) and (density[i] < density[j])
                j_dom_i = (Cov[j] < Cov[i]) and (density[j] < density[i])
            if i_dom_j and not j_dom_i:
                M[i, j], M[j, i] = 1, -1
            elif j_dom_i and not i_dom_j:
                M[i, j], M[j, i] = -1, 1
    return M, density


class RPSSurvival(Survival):
    def __init__(self, ref_dirs):
        super().__init__(filter_infeasible=True)
        self.ref_dirs = ref_dirs

    def _do(self, problem, pop, *args, n_survive=None, random_state=None, **kwargs):
        F = pop.get("F").astype(float, copy=False)
        M, density = rps_domination_matrix(F, self.ref_dirs)
        fronts = fast_non_dominated_sort_from_matrix(M)

        density = density.copy()
        if len(fronts) > 0:
            front0 = fronts[0]
            for obj in range(F.shape[1]):
                extreme_local = front0[np.argmin(F[front0, obj])]
                density[extreme_local] = 0.0

        survivors = []
        for rank, front in enumerate(fronts):
            if len(survivors) + len(front) > n_survive:
                n_remove = len(survivors) + len(front) - n_survive
                front = truncate_by_score(front, density[front], n_remove, prefer="smallest", random_state=random_state)
            for global_i in front:
                pop[int(global_i)].set("rank", rank)
                # the tournament treats larger "crowding" as better; density is better when smaller, so negate it
                pop[int(global_i)].set("crowding", -density[int(global_i)])
            survivors.extend(front.tolist())
        return pop[survivors]


class RPS_NSGA2(NSGA2):
    """Population size equals the number of reference directions; pop_size cannot be set independently."""

    def __init__(self, ref_dirs, **kwargs):
        super().__init__(pop_size=len(ref_dirs), survival=RPSSurvival(ref_dirs=ref_dirs), **kwargs)
        self.tournament_type = "comp_by_rank_and_crowding"
