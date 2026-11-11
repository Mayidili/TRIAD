"""
Outer cross-validation folds for the Duke cohort: StratifiedGroupKFold on the labels, grouped by patient_id.
Output: <results>/nested_cv/splits.npz (outer_fold, patient_ids, y, n_outer, outer_seed, data_npz)
"""
import argparse
import os

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from prism.paths import DATA_ROOT, RESULTS_ROOT

CORPUS_KEYS = ["X_pool", "y", "patient_ids", "feature_names", "feature_to_filter_index", "n_filters"]


def load_corpus_npz(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in CORPUS_KEYS}


def make_splits(data_npz, split_path, n_outer, outer_seed):
    corpus = load_corpus_npz(data_npz)
    y, pids = corpus["y"], corpus["patient_ids"]
    outer_fold = np.full(len(y), -1, dtype=int)
    skf = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=outer_seed)
    for k, (_, test_idx) in enumerate(skf.split(corpus["X_pool"], y, groups=pids)):
        outer_fold[test_idx] = k
    assert (outer_fold >= 0).all()
    for k in range(n_outer):
        # leakage check: no patient in both the outer training and test folds
        assert not set(pids[outer_fold == k]) & set(pids[outer_fold != k])

    if os.path.exists(split_path):
        old = np.load(split_path, allow_pickle=True)
        if not np.array_equal(old["outer_fold"], outer_fold) or not np.array_equal(old["patient_ids"], pids):
            raise RuntimeError(f"{split_path} exists and differs from this split; use another --out or remove it first")
        print(f"{split_path} exists and is identical, skipping")
    else:
        os.makedirs(os.path.dirname(split_path) or ".", exist_ok=True)
        np.savez(split_path, outer_fold=outer_fold, patient_ids=pids, y=y,
                 n_outer=n_outer, outer_seed=outer_seed, data_npz=os.path.abspath(data_npz))

    for k in range(n_outer):
        train = outer_fold != k
        print(f"fold{k}: train={train.sum()} (pos={y[train].sum()}), test={(~train).sum()} (pos={y[~train].sum()})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(DATA_ROOT, "duke_feature_pool.npz"))
    parser.add_argument("--out", default=os.path.join(RESULTS_ROOT, "nested_cv/splits.npz"))
    parser.add_argument("--n-outer", type=int, default=5)
    parser.add_argument("--outer-seed", type=int, default=2026)
    args = parser.parse_args()
    make_splits(args.data, args.out, args.n_outer, args.outer_seed)
