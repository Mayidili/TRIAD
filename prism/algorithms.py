"""
Search algorithms: NSGA-II and three NSGA-II variants (CSDR, TS-NSGA-II, RPS-NSGA-II),
implemented from the formulas in the original papers (see prism/ec/).

NSGA-II and CSDR take pop_size directly. For TS-NSGA-II and RPS-NSGA-II the population
size equals the number of reference directions and cannot be set independently;
n_partitions=13 (105 directions for three objectives) is the value closest to pop_size=100.
"""
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.ref_dirs import get_reference_directions

from prism.ec.csdr import NSGA2_CSDR
from prism.ec.rps_nsga2 import RPS_NSGA2
from prism.ec.ts_nsga2 import TS_NSGA2

POP_SIZE = 100
N_OBJ = 3
N_EVAL_DEFAULT = 10000

# The number of Das-Dennis directions is C(n_partitions+n_obj-1, n_obj-1); with n_obj=3,
# n_partitions=13 gives 105, the closest value to POP_SIZE=100.
_REF_DIRS_N_PARTITIONS = 13

ALL_ALGO_NAMES = ["NSGA2", "CSDR", "TSNSGA2", "RPS"]


# Two objectives: n_partitions=99 gives exactly 100 directions, equal to the NSGA2/CSDR population.
_REF_DIRS_N_PARTITIONS_BY_NOBJ = {2: 99, 3: _REF_DIRS_N_PARTITIONS}


def _build_ref_dirs(n_obj=N_OBJ):
    return get_reference_directions("das-dennis", n_obj, n_partitions=_REF_DIRS_N_PARTITIONS_BY_NOBJ[n_obj])


def build_algorithm(name, pop_size=POP_SIZE, n_obj=N_OBJ, n_eval=N_EVAL_DEFAULT, **kwargs):
    """
    n_eval is used to estimate max_gen, which drives the parameter annealing of CSDR and the
    stage switch of TS-NSGA-II; pass the same n_eval as the termination criterion.
    max_gen is computed from the effective population size: for TSNSGA2/RPS this is the
    number of reference directions, not pop_size.
    """
    if name == "NSGA2":
        return NSGA2(pop_size=pop_size, **kwargs)
    if name == "CSDR":
        max_gen = max(n_eval // pop_size, 1)
        return NSGA2_CSDR(pop_size=pop_size, max_gen=max_gen, **kwargs)
    if name == "TSNSGA2":
        ref_dirs = _build_ref_dirs(n_obj)
        max_gen = max(n_eval // len(ref_dirs), 1)
        return TS_NSGA2(ref_dirs=ref_dirs, max_gen=max_gen, **kwargs)
    if name == "RPS":
        ref_dirs = _build_ref_dirs(n_obj)
        return RPS_NSGA2(ref_dirs=ref_dirs, **kwargs)
    raise ValueError(f"Unknown algorithm: {name}; choose from {ALL_ALGO_NAMES}")


def build_all_algorithms(pop_size=POP_SIZE, n_obj=N_OBJ, n_eval=N_EVAL_DEFAULT):
    return {name: build_algorithm(name, pop_size=pop_size, n_obj=n_obj, n_eval=n_eval) for name in ALL_ALGO_NAMES}
