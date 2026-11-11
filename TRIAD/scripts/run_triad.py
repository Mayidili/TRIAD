"""
TRIAD pipeline.
  1. Candidate features: 1688 first post-contrast radiomic features + 44 DCE kinetic features (prism/kinetic.py);
     the kinetic features form a 19th filter configuration, "kinetic".
  2. Two objectives: 1 - AUC and the fraction of selected features.
  3. Classifier: L2 logistic regression (median imputation -> quantile normal transform -> LR);
     the third chromosome segment encodes C only (log scale, 1e-4 to 1e1). Fitness = mean inner stratified 5-fold CV AUC.
  4. Representative solution of each run (pre-specified): among solutions with inner AUC >= front maximum - 0.01,
     the one with the fewest features ("tol"). The knee point ("knee") and the maximum inner AUC point ("maxauc")
     are sensitivity analyses.
  5. Outer folds are taken from <results>/nested_cv/splits.npz.

Stages:
  pool      merge radiomic + kinetic features -> <run>/data/duke_pool_v3.npz
  split     copy the outer-fold assignment -> <run>/nested/splits.npz
  search    --scope full | fold, one algorithm, several seeds
  compare   HV / IGD of 4 algorithms x 10 seeds on the full cohort, Kruskal-Wallis + Dunn
  baseline  5 staged baselines (selection + LR grid over C), --scope nested (outer folds) or full (whole cohort, for external validation)
  evaluate  nested CV summary
  final     final model on the full cohort (main algorithm = highest mean HV in compare, or --algorithm)
  external  frozen models scored on I-SPY1/2 (MedSAM2 masks primary, official masks as sensitivity analysis)
"""
import argparse
import os
import pickle
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from prism.algorithms import ALL_ALGO_NAMES, build_algorithm
from prism.feature_extraction import FILTER_NAMES
from prism.paths import DATA_ROOT, KIN_NPZ, RAD_POOL, RESULTS_ROOT, RUN_ROOT
from prism.problem import (TriadLRProblem, build_inner_folds, decode_solution, fit_transform_pre,
                           init_worker, make_lr)

try:
    from scripts.baseline_selectors import mrmr_select
    from scripts.baseline_selectors import elasticnet_select
except ModuleNotFoundError:
    from baseline_selectors import mrmr_select
    from baseline_selectors import elasticnet_select

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# output root and Duke feature files can be overridden with environment variables (prism/paths.py)
V3 = RUN_ROOT
POOL_V3 = os.path.join(V3, "data/duke_pool_v3.npz")
V2_SPLITS = os.path.join(RESULTS_ROOT, "nested_cv/splits.npz")
ISPY_META = {"ISPY1": os.path.join(DATA_ROOT, "ISPY/ISPY1/BreastDCEDL_spy1_metadata.csv"),
             "ISPY2": os.path.join(DATA_ROOT, "ISPY/ISPY2/BreastDCEDL_ISPY2_metadata.csv")}

FILTER_NAMES_V3 = list(FILTER_NAMES) + ["kinetic"]
N_EVAL = 10000
TOL = 0.01
RULES = ["tol", "knee", "maxauc"]
MAIN_RULE = "tol"
THRESHOLDS = (0.30, 0.40, 0.50)
MAIN_THRESHOLD = 0.50
BASELINES = ["lasso", "mrmr", "elasticnet", "rfe", "full"]
C_GRID = np.logspace(-4, 1, 11)
HV_REF = np.array([0.5, 1.0])
# iteration cap for all iterative solvers (LassoCV / ElasticNetCV / logistic regression)
MAX_ITER = 1000


def parse_seeds(spec):
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def load_pool():
    d = np.load(POOL_V3, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_splits():
    return np.load(os.path.join(V3, "nested/splits.npz"), allow_pickle=True)


def run_dir(scope, fold=None):
    return os.path.join(V3, "full") if scope == "full" else os.path.join(V3, "nested", f"fold{fold}")


# ---------------------------------------------------------------- pool / split

def stage_pool():
    r = np.load(RAD_POOL, allow_pickle=True)
    k = np.load(KIN_NPZ, allow_pickle=True)
    assert np.array_equal(r["patient_ids"], k["patient_ids"]), "patient order differs between kinetic and radiomic features"
    ok = k["status"] == "ok"
    print(f"kinetic features: ok {ok.sum()}/{len(ok)}; other cases are median-imputed within folds")
    X = np.hstack([r["X_pool"], k["X"]])
    names = np.concatenate([r["feature_names"], k["feature_names"]])
    f2f = np.concatenate([r["feature_to_filter_index"], np.full(len(k["feature_names"]), len(FILTER_NAMES))])
    np.savez(POOL_V3, X_pool=X, y=r["y"], patient_ids=r["patient_ids"], feature_names=names,
             feature_to_filter_index=f2f, n_filters=len(FILTER_NAMES_V3), filter_names=np.array(FILTER_NAMES_V3),
             kinetic_status=k["status"])
    print(f"{POOL_V3}: X {X.shape}, n_filters {len(FILTER_NAMES_V3)}, HR+ {r['y'].mean():.3f}")


def stage_split():
    s = np.load(V2_SPLITS, allow_pickle=True)
    pool = load_pool()
    assert np.array_equal(s["patient_ids"], pool["patient_ids"])
    os.makedirs(os.path.join(V3, "nested"), exist_ok=True)
    np.savez(os.path.join(V3, "nested/splits.npz"), outer_fold=s["outer_fold"], patient_ids=s["patient_ids"],
             y=s["y"], n_outer=s["n_outer"], outer_seed=s["outer_seed"])
    for k in range(int(s["n_outer"])):
        te = s["outer_fold"] == k
        print(f"fold{k}: train={(~te).sum()} test={te.sum()} (pos={s['y'][te].sum()})")


# ---------------------------------------------------------------- search

def stage_search(scope, fold, algorithm, seeds, n_eval, n_proc):
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    pool_d = load_pool()
    X, y = pool_d["X_pool"], pool_d["y"]
    if scope == "fold":
        tr = load_splits()["outer_fold"] != fold
        X, y = X[tr], y[tr]
    n_filters, f2f = int(pool_d["n_filters"]), pool_d["feature_to_filter_index"]
    out = run_dir(scope, fold)
    os.makedirs(out, exist_ok=True)
    for seed in seeds:
        path = os.path.join(out, f"{algorithm}_seed{seed}.npz")
        if os.path.exists(path):
            print(f"{path} exists, skipping", flush=True)
            continue
        t0 = time.time()
        folds = build_inner_folds(X, y, n_splits=5, random_state=seed)
        with Pool(n_proc, initializer=init_worker, initargs=(folds, n_filters, f2f)) as pool:
            problem = TriadLRProblem(n_filters, X.shape[1], pool)
            algo = build_algorithm(algorithm, n_obj=2, n_eval=n_eval)
            res = minimize(problem, algo, get_termination("n_eval", n_eval), seed=seed, verbose=False)
        F = np.atleast_2d(res.F)
        Xs = np.atleast_2d(res.X)
        np.savez(path, X=Xs, F=F, algorithm=algorithm, seed=seed, n_eval=n_eval, seconds=time.time() - t0)
        print(f"{scope} fold={fold} {algorithm} seed{seed}: front {len(F)}, best inner AUC {1 - F[:, 0].min():.4f}, "
              f"{time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- representative solutions

def representative(F, rule):
    """Row index in F of the representative solution, chosen among non-dominated solutions (TS-NSGA-II returns the whole population, including dominated ones)."""
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return int(nd[_representative(F[nd], rule)])


def _representative(F, rule):
    auc, nf = 1 - F[:, 0], F[:, 1]
    if rule == "tol":
        cand = np.flatnonzero(auc >= auc.max() - TOL)
        best = cand[nf[cand] == nf[cand].min()]
        return int(best[np.argmax(auc[best])])
    if rule == "maxauc":
        cand = np.flatnonzero(auc == auc.max())
        return int(cand[np.argmin(nf[cand])])
    if rule == "knee":
        lo, hi = F.min(0), F.max(0)
        Fn = (F - lo) / np.maximum(hi - lo, 1e-12)
        return int(np.argmin(np.linalg.norm(Fn, axis=1)))
    raise ValueError(rule)


def load_representatives(path_dir, algorithm, seeds, rule, n_filters, n_features, f2f):
    masks, Cs = [], []
    for seed in seeds:
        r = np.load(os.path.join(path_dir, f"{algorithm}_seed{seed}.npz"))
        i = representative(r["F"], rule)
        m, C = decode_solution(r["X"][i], n_filters, n_features, f2f)
        masks.append(m)
        Cs.append(C)
    return np.array(masks), np.array(Cs)


def consensus(masks, t):
    freq = masks.mean(0)
    mask = freq >= t
    fallback = not mask.any()
    if fallback:
        mask = freq == freq.max()
    return mask, fallback, freq


class FittedLR:
    """Imputation + quantile transform + LR fitted on the given feature columns only; picklable."""

    def __init__(self, mask, C):
        self.mask, self.C = mask, C

    def fit(self, X, y):
        A, _, self.steps = fit_transform_pre(X[:, self.mask])
        self.clf = make_lr(self.C).fit(A, y)
        return self

    def predict(self, X):
        A = X[:, self.mask]
        for s in self.steps:
            A = s.transform(A)
        return self.clf.predict_proba(A)[:, 1]


# ---------------------------------------------------------------- compare

def stage_compare(seeds):
    import scikit_posthocs as sp
    from pymoo.indicators.hv import HV
    from pymoo.indicators.igd import IGD
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    from scipy.stats import kruskal

    runs = {a: [np.load(os.path.join(run_dir("full"), f"{a}_seed{s}.npz"))["F"] for s in seeds] for a in ALL_ALGO_NAMES}
    allF = np.vstack([F for v in runs.values() for F in v])
    if (allF > HV_REF).any():
        print(f"Warning: {(allF > HV_REF).any(1).sum()} solutions are worse than the HV reference point {HV_REF} and do not count towards HV")
    ref_front = allF[NonDominatedSorting().do(allF, only_non_dominated_front=True)]
    rows = []
    for a, Fs in runs.items():
        for s, F in zip(seeds, Fs):
            rows.append({"algorithm": a, "seed": s, "hv": HV(ref_point=HV_REF)(F), "igd": IGD(ref_front)(F),
                         "best_inner_auc": 1 - F[:, 0].min(), "front_size": len(F)})
    df = pd.DataFrame(rows)
    out = os.path.join(V3, "evaluation")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "algo_compare_hv_igd.csv"), index=False)
    summ = df.groupby("algorithm")[["hv", "igd", "best_inner_auc"]].agg(["mean", "std"])
    print(summ)
    H, p = kruskal(*[df[df.algorithm == a].hv for a in ALL_ALGO_NAMES])
    print(f"Kruskal-Wallis HV: H={H:.3f}, p={p:.4g}")
    dunn = sp.posthoc_dunn(df, val_col="hv", group_col="algorithm", p_adjust="bonferroni")
    dunn.to_csv(os.path.join(out, "algo_compare_dunn.csv"))
    print(dunn)
    best = df.groupby("algorithm").hv.mean().idxmax()
    with open(os.path.join(out, "main_algorithm.txt"), "w") as f:
        f.write(best + "\n")
    print(f"main algorithm (highest mean HV): {best}")


# ---------------------------------------------------------------- baselines

def select_baseline(method, X, y):
    if method == "lasso":
        las = LassoCV(cv=5, random_state=0, max_iter=MAX_ITER).fit(StandardScaler().fit_transform(X), y.astype(float))
        sel = np.abs(las.coef_) > 1e-6
        return sel if sel.any() else np.ones(X.shape[1], bool)
    if method == "mrmr":
        return mrmr_select(X, y, k=30, random_state=0)
    if method == "elasticnet":
        return elasticnet_select(X, y, max_iter=MAX_ITER)
    if method == "rfe":
        A, _, _ = fit_transform_pre(X)
        est = LogisticRegression(C=1.0, max_iter=MAX_ITER, class_weight="balanced")
        return RFECV(est, step=0.05, cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="roc_auc",
                     min_features_to_select=5, n_jobs=-1).fit(A, y).support_
    if method == "full":
        return np.ones(X.shape[1], bool)
    raise ValueError(method)


def fit_baseline(method, X, y):
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    sel = select_baseline(method, Xi, y)
    pipe = make_pipeline(SimpleImputer(strategy="median"),
                         QuantileTransformer(n_quantiles=min(200, len(y)), output_distribution="normal"),
                         LogisticRegression(max_iter=MAX_ITER, class_weight="balanced"))
    gs = GridSearchCV(pipe, {"logisticregression__C": C_GRID}, scoring="roc_auc",
                      cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=-1).fit(X[:, sel], y)
    return sel, gs.best_estimator_, float(gs.best_params_["logisticregression__C"])


def stage_baseline(method, scope):
    pool_d = load_pool()
    X, y = pool_d["X_pool"], pool_d["y"]
    out = os.path.join(V3, "baselines")
    os.makedirs(out, exist_ok=True)
    if scope == "full":
        sel, model, C = fit_baseline(method, X, y)
        with open(os.path.join(out, f"{method}_full.pkl"), "wb") as f:
            pickle.dump({"mask": sel, "model": model, "C": C, "feature_names": pool_d["feature_names"]}, f)
        print(f"{method} full: n={sel.sum()}, C={C:.4g}")
        return
    s = load_splits()
    of = s["outer_fold"]
    proba = np.full(len(y), np.nan)
    nf, Cs = [], []
    for k in range(int(s["n_outer"])):
        tr, te = of != k, of == k
        sel, model, C = fit_baseline(method, X[tr], y[tr])
        proba[te] = model.predict_proba(X[te][:, sel])[:, 1]
        nf.append(int(sel.sum()))
        Cs.append(C)
        print(f"{method} fold{k}: n={sel.sum()}, C={C:.4g}, outer AUC={roc_auc_score(y[te], proba[te]):.4f}", flush=True)
    np.savez(os.path.join(out, f"{method}.npz"), proba=proba, n_features=np.array(nf), C=np.array(Cs))


# ---------------------------------------------------------------- evaluate

def _strat_boot(y, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return np.percentile([roc_auc_score(y[i], p[i]) for i in (_strat_boot(y, rng) for _ in range(n))], [2.5, 97.5])


def paired_boot(y, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for i in range(n):
        idx = _strat_boot(y, rng)
        d[i] = roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])
    return np.percentile(d, [2.5, 97.5]), min(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0)


def stage_evaluate(algorithms, seeds):
    from pymoo.indicators.hv import HV
    from scipy.stats import wilcoxon

    pool_d = load_pool()
    X, y, names = pool_d["X_pool"], pool_d["y"], pool_d["feature_names"]
    n_filters, f2f = int(pool_d["n_filters"]), pool_d["feature_to_filter_index"]
    s = load_splits()
    of, n_outer = s["outer_fold"], int(s["n_outer"])
    oof, nfeat, fold_rows, feat_rows, hv_rows = {}, {}, [], [], []

    for algo in algorithms:
        for k in range(n_outer):
            for seed in seeds:
                F = np.load(os.path.join(run_dir("fold", k), f"{algo}_seed{seed}.npz"))["F"]
                hv_rows.append({"algorithm": algo, "fold": k, "seed": seed, "hv": HV(ref_point=HV_REF)(F),
                                "best_inner_auc": 1 - F[:, 0].min()})
        for rule in RULES:
            keys = [f"TRIAD-{algo}-{rule}-thr{int(t * 100)}" for t in THRESHOLDS] + [f"TRIAD-{algo}-{rule}-ensemble"]
            for key in keys:
                oof[key], nfeat[key] = np.full(len(y), np.nan), []
            for k in range(n_outer):
                tr, te = of != k, of == k
                masks, Cs = load_representatives(run_dir("fold", k), algo, seeds, rule, n_filters, X.shape[1], f2f)
                for t, key in zip(THRESHOLDS, keys):
                    mask, fb, freq = consensus(masks, t)
                    oof[key][te] = FittedLR(mask, float(np.median(Cs))).fit(X[tr], y[tr]).predict(X[te])
                    nfeat[key].append(int(mask.sum()))
                    fold_rows.append({"method": key, "fold": k, "n_features": int(mask.sum()), "fallback": fb,
                                      "n_kinetic": int((mask & (f2f == len(FILTER_NAMES))).sum()),
                                      "outer_auc": roc_auc_score(y[te], oof[key][te])})
                    if rule == MAIN_RULE and t == MAIN_THRESHOLD:
                        for j in np.flatnonzero(mask):
                            feat_rows.append({"algorithm": algo, "fold": k, "feature": names[j], "freq": freq[j]})
                preds = [FittedLR(m, C).fit(X[tr], y[tr]).predict(X[te]) for m, C in zip(masks, Cs) if m.any()]
                oof[keys[-1]][te] = np.mean(preds, axis=0)
                nfeat[keys[-1]].append(int(np.median(masks.sum(1))))
                fold_rows.append({"method": keys[-1], "fold": k, "n_features": int(np.median(masks.sum(1))),
                                  "fallback": False, "n_kinetic": np.nan,
                                  "outer_auc": roc_auc_score(y[te], oof[keys[-1]][te])})
                print(f"{algo} {rule} fold{k}: rep n={masks.sum(1).tolist()} C={np.round(np.log10(Cs), 2).tolist()} "
                      f"thr50 n={nfeat[keys[2]][-1]}", flush=True)

    for m in BASELINES:
        p = os.path.join(V3, "baselines", f"{m}.npz")
        if not os.path.exists(p):
            print(f"baseline {m} missing, skipping")
            continue
        b = np.load(p)
        oof[m], nfeat[m] = b["proba"], b["n_features"].tolist()
        for k in range(n_outer):
            te = of == k
            fold_rows.append({"method": m, "fold": k, "n_features": int(b["n_features"][k]), "fallback": False,
                              "n_kinetic": np.nan, "outer_auc": roc_auc_score(y[te], b["proba"][te])})

    summary = []
    for key, p in oof.items():
        assert not np.isnan(p).any(), key
        pf = [roc_auc_score(y[of == k], p[of == k]) for k in range(n_outer)]
        lo, hi = boot_ci(y, p)
        summary.append({"method": key, "oof_auc": roc_auc_score(y, p), "ci_low": lo, "ci_high": hi,
                        "fold_auc_mean": np.mean(pf), "fold_auc_sd": np.std(pf),
                        "n_features_per_fold": "/".join(map(str, nfeat[key]))})
    comps = []
    for algo in algorithms:
        main = f"TRIAD-{algo}-{MAIN_RULE}-thr{int(MAIN_THRESHOLD * 100)}"
        for m in [b for b in BASELINES if b in oof]:
            (lo, hi), p = paired_boot(y, oof[main], oof[m])
            comps.append({"triad": main, "baseline": m, "delta": roc_auc_score(y, oof[main]) - roc_auc_score(y, oof[m]),
                          "ci_low": lo, "ci_high": hi, "p": p, "p_bonferroni": min(p * len(BASELINES), 1.0)})
    if len(algorithms) >= 2:
        a, b = [f"TRIAD-{x}-{MAIN_RULE}-thr{int(MAIN_THRESHOLD * 100)}" for x in algorithms[:2]]
        (lo, hi), p = paired_boot(y, oof[a], oof[b])
        comps.append({"triad": a, "baseline": b, "delta": roc_auc_score(y, oof[a]) - roc_auc_score(y, oof[b]),
                      "ci_low": lo, "ci_high": hi, "p": p, "p_bonferroni": p})

    out = os.path.join(V3, "evaluation")
    os.makedirs(out, exist_ok=True)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(out, "nested_summary.csv"), index=False)
    pd.DataFrame(fold_rows).to_csv(os.path.join(out, "nested_per_fold.csv"), index=False)
    cdf = pd.DataFrame(comps)
    cdf.to_csv(os.path.join(out, "nested_vs_baselines.csv"), index=False)
    pd.DataFrame(feat_rows).to_csv(os.path.join(out, "nested_selected_features_main.csv"), index=False)
    hdf = pd.DataFrame(hv_rows)
    hdf.to_csv(os.path.join(out, "nested_hv.csv"), index=False)
    np.savez(os.path.join(out, "nested_oof.npz"), y=y, outer_fold=of, **{k.replace("-", "_"): v for k, v in oof.items()})
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.max_columns", 20):
        print(sdf.sort_values("oof_auc", ascending=False))
        print(cdf)
        print(hdf.groupby("algorithm")[["hv", "best_inner_auc"]].agg(["mean", "std"]))
    if {"CSDR", "NSGA2"} <= set(algorithms):
        a = hdf[hdf.algorithm == "CSDR"].sort_values(["fold", "seed"]).hv.values
        b = hdf[hdf.algorithm == "NSGA2"].sort_values(["fold", "seed"]).hv.values
        print(f"nested HV CSDR vs NSGA2 paired Wilcoxon p={wilcoxon(a, b).pvalue:.4g}, CSDR higher in {(a > b).sum()}/{len(a)}")


# ---------------------------------------------------------------- final / external

def main_algorithm(arg):
    if arg:
        return arg
    with open(os.path.join(V3, "evaluation/main_algorithm.txt")) as f:
        return f.read().strip()


def stage_final(algorithm, seeds):
    pool_d = load_pool()
    X, y, names = pool_d["X_pool"], pool_d["y"], pool_d["feature_names"]
    n_filters, f2f = int(pool_d["n_filters"]), pool_d["feature_to_filter_index"]
    masks, Cs = load_representatives(run_dir("full"), algorithm, seeds, MAIN_RULE, n_filters, X.shape[1], f2f)
    C = float(np.median(Cs))
    models = {}
    for t in THRESHOLDS:
        mask, fb, freq = consensus(masks, t)
        models[f"thr{int(t * 100)}"] = FittedLR(mask, C).fit(X, y)
        print(f"thr{int(t * 100)}: n={mask.sum()} (kinetic {int((mask & (f2f == len(FILTER_NAMES))).sum())}), fallback={fb}")
    models["ensemble"] = [FittedLR(m, c).fit(X, y) for m, c in zip(masks, Cs) if m.any()]
    freq = masks.mean(0)
    order = np.argsort(-freq)
    pd.DataFrame({"feature": names[order], "freq": freq[order]}).query("freq > 0").to_csv(
        os.path.join(V3, "evaluation/final_feature_frequency.csv"), index=False)
    with open(os.path.join(V3, "final_models.pkl"), "wb") as f:
        pickle.dump({"algorithm": algorithm, "rule": MAIN_RULE, "C": C, "rep_n": masks.sum(1), "rep_C": Cs,
                     "models": models, "feature_names": names}, f)
    print(f"final ({algorithm}): representative n={masks.sum(1).tolist()}, median C={C:.4g}")


def load_external(cohort, mask_kind, names):
    d = os.path.join(V3, "data")
    r = np.load(os.path.join(d, f"{cohort.lower()}_radiomics_{mask_kind}.npz"), allow_pickle=True)
    k = np.load(os.path.join(d, f"{cohort.lower()}_kinetic_{mask_kind}.npz"), allow_pickle=True)
    rdf = pd.DataFrame(r["X"], index=r["patient_ids"], columns=r["feature_names"])
    kdf = pd.DataFrame(k["X"], index=k["patient_ids"], columns=k["feature_names"])
    df = rdf.join(kdf, how="left")[list(names)]
    meta = pd.read_csv(ISPY_META[cohort]).set_index("pid")
    lab = meta.loc[meta.index.intersection(df.index), "HR"].dropna()
    df = df.loc[lab.index]
    ok_rad = r["status"][pd.Index(r["patient_ids"]).get_indexer(df.index)] == "ok"
    df, lab = df[ok_rad], lab[ok_rad]
    return df.values, lab.values.astype(int), df.index.values


def stage_external():
    with open(os.path.join(V3, "final_models.pkl"), "rb") as f:
        fm = pickle.load(f)
    names = fm["feature_names"]
    base = {}
    for m in BASELINES:
        p = os.path.join(V3, "baselines", f"{m}_full.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                base[m] = pickle.load(f)
    rows = []
    for cohort in ["ISPY1", "ISPY2"]:
        for mk in ["medsam2", "official"]:
            X, y, pids = load_external(cohort, mk, names)
            preds = {f"TRIAD-{k}": (m.predict(X) if k != "ensemble" else np.mean([e.predict(X) for e in m], 0))
                     for k, m in fm["models"].items()}
            for m, b in base.items():
                preds[m] = b["model"].predict_proba(X[:, b["mask"]])[:, 1]
            for key, p in preds.items():
                lo, hi = boot_ci(y, p)
                rows.append({"cohort": cohort, "mask": mk, "model": key, "n": len(y), "pos": int(y.sum()),
                             "auc": roc_auc_score(y, p), "ci_low": lo, "ci_high": hi})
            print(f"{cohort} {mk}: n={len(y)} pos={y.sum()}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(V3, "evaluation/external_summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["pool", "split", "search", "compare", "baseline", "evaluate", "final", "external"])
    ap.add_argument("--scope", choices=["full", "fold", "nested"], default="full")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--algorithm")
    ap.add_argument("--algorithms", default="CSDR,NSGA2,TSNSGA2,RPS")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    ap.add_argument("--n-proc", type=int, default=8)
    ap.add_argument("--method", choices=BASELINES)
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    if a.stage == "pool":
        stage_pool()
    elif a.stage == "split":
        stage_split()
    elif a.stage == "search":
        stage_search(a.scope, a.fold, a.algorithm, seeds, a.n_eval, a.n_proc)
    elif a.stage == "compare":
        stage_compare(seeds)
    elif a.stage == "baseline":
        stage_baseline(a.method, "full" if a.scope == "full" else "nested")
    elif a.stage == "evaluate":
        stage_evaluate(a.algorithms.split(","), seeds)
    elif a.stage == "final":
        stage_final(main_algorithm(a.algorithm), seeds)
    else:
        stage_external()
