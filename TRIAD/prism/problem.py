"""
TRIAD problem: three-segment encoding (filters + features + classifier), two objectives,
L2 logistic regression.

Chromosome: [filter_seg (n_filters) | feature_mask_seg (n_features) | C (1)], all real-valued in [0, 1].
Objectives (minimised):
  f1 = 1 - mean AUC over inner stratified 5-fold CV
  f2 = number of selected features / number of candidate features

Classifier: median imputation -> QuantileTransformer (normal output) -> LogisticRegression (L2, class_weight=balanced).
Imputation and the quantile transform are fitted per feature, so fitting them once per inner fold on all
candidate features and subsetting the selected columns at evaluation time is equivalent to refitting on the
selected features only, and much faster.

Parallelism: instead of ElementwiseProblem (which pickles the full data for every individual), _evaluate is
vectorised; an initializer stores the transformed inner-fold matrices in worker globals so that only
chromosomes are sent to the workers.
"""
import warnings

import numpy as np
from pymoo.core.problem import Problem
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

from prism.decode import decode_feature_mask, decode_filter_segment

C_LOW, C_HIGH = 1e-4, 1e1
N_HP = 1


def decode_c(g):
    return float(10.0 ** (np.log10(C_LOW) + float(g) * (np.log10(C_HIGH) - np.log10(C_LOW))))


def make_preprocessor(n_samples):
    return [SimpleImputer(strategy="median"),
            QuantileTransformer(n_quantiles=min(200, n_samples), output_distribution="normal")]


def fit_transform_pre(X_tr, X_te=None):
    steps = make_preprocessor(len(X_tr))
    A, B = X_tr, X_te
    for s in steps:
        A = s.fit_transform(A)
        if B is not None:
            B = s.transform(B)
    return A, B, steps


def make_lr(C):
    return LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=1000, class_weight="balanced")


def decode_solution(chrom, n_filters, n_features, feature_to_filter_index, threshold=0.5):
    active = decode_filter_segment(chrom[:n_filters], threshold)
    selected = decode_feature_mask(chrom[n_filters:n_filters + n_features], active, feature_to_filter_index, threshold)
    return selected, decode_c(chrom[n_filters + n_features])


# ---------------------------------------------------------------- worker side

_W = {}


def init_worker(folds, n_filters, feature_to_filter_index):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    _W["folds"] = folds
    _W["n_filters"] = n_filters
    _W["f2f"] = feature_to_filter_index


def evaluate_one(chrom):
    f2f = _W["f2f"]
    selected, C = decode_solution(chrom, _W["n_filters"], len(f2f), f2f)
    n_sel = int(selected.sum())
    if n_sel == 0:
        return 1.0, 1.0
    aucs = []
    for A, ya, B, yb in _W["folds"]:
        clf = make_lr(C).fit(A[:, selected], ya)
        aucs.append(roc_auc_score(yb, clf.decision_function(B[:, selected])))
    return 1.0 - float(np.mean(aucs)), n_sel / len(f2f)


def build_inner_folds(X, y, n_splits=5, random_state=0):
    folds = []
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=random_state).split(X, y):
        A, B, _ = fit_transform_pre(X[tr], X[te])
        folds.append((A, y[tr], B, y[te]))
    return folds


class TriadLRProblem(Problem):
    def __init__(self, n_filters, n_features, pool):
        self.pool = pool
        super().__init__(n_var=n_filters + n_features + N_HP, n_obj=2, n_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.array(self.pool.map(evaluate_one, list(X), chunksize=1))
