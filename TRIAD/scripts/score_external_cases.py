#!/usr/bin/env python3
"""Per-case predicted probabilities of the frozen TRIAD model on the external cohorts (MedSAM2 masks).
Output: <run>/evaluation/external_case_scores.csv"""
import os
import pickle
import sys

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.abspath(__file__))]
from prism.paths import RUN_ROOT
V3 = RUN_ROOT
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import run_triad
import __main__
for k in dir(run_triad):
    if isinstance(getattr(run_triad, k), type):
        setattr(__main__, k, getattr(run_triad, k))
with open(os.path.join(V3, "final_models.pkl"), "rb") as f:
    fm = pickle.load(f)
rows = []
for cohort in ["ISPY1", "ISPY2"]:
    X, y, pids = run_triad.load_external(cohort, "medsam2", fm["feature_names"])
    p = fm["models"]["thr50"].predict(X)
    print(cohort, len(y), int(y.sum()), "AUC", round(roc_auc_score(y, p), 4))
    rows += [dict(cohort=cohort, pid=str(a), hr=int(b), prob=float(c)) for a, b, c in zip(pids, y, p)]
df = pd.DataFrame(rows)
df.to_csv(os.path.join(V3, "evaluation/external_case_scores.csv"), index=False)
print(df.groupby(["cohort", "hr"]).prob.describe(percentiles=[.25, .5, .75]).round(3).to_string())
