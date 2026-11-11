"""
Additional experiments on the same outer folds, seeds 0-9 and evaluation budget (10,000 evaluations).

Variants (run within the nested-CV outer folds only and compared pairwise with the main TRIAD-CSDR result):
  random    random search: 10,000 chromosomes sampled uniformly over the full TRIAD encoding (filters + features + C),
            non-dominated solutions kept
  nofilter  no filter genes: chromosome = feature mask + C, all filter configurations always on, CSDR
  seq       staged: feature-mask MOEA (C fixed at 1.0, CSDR, 9,900 evaluations), then C tuned on a grid (11 values)
            for the representative solution on the same inner folds; total <= 10,000 evaluations
All variants store chromosomes at full length [filters | features | C] and use the representative-solution rule (tol),
the 50% consensus threshold and the median-C rule of run_triad.py.

Other stages:
  evaluate  nested OOF AUC, feature counts and inner HV of each variant; paired bootstrap against TRIAD-CSDR-tol-thr50
  final     final-configuration checks: objective values of the final mask + median C on the inner folds of each
            full-cohort seed and whether that seed's front dominates them; sensitivity to re-tuning C for the
            consensus mask within the outer folds
  ablation  leave-out ablation of the final 15-feature model (drop single features / the kinetic group / the radiomic
            group, refit on Duke, evaluate on I-SPY1/2) + Spearman correlation and VIF on Duke
  audit     external-validation checks: label coding, feature column order, per-case reproduction of the external
            predictions after reloading the model
"""
import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_triad as R  # noqa: E402
from prism.algorithms import build_algorithm  # noqa: E402
from prism.paths import ABLATION_DIR, MAMAMIA_DIR  # noqa: E402
from prism.problem import (C_HIGH, C_LOW, build_inner_folds, decode_solution, evaluate_one,  # noqa: E402
                           init_worker, make_lr)

REV = ABLATION_DIR
VARIANTS = ["random", "nofilter", "seq"]
MAIN_KEY = "TRIAD-CSDR-tol-thr50"
SEQ_C = 1.0
SEQ_EVAL = 9900


def load_final():
    """final_models.pkl is written by run_triad.py run as a script, so FittedLR is pickled under __main__."""
    import __main__
    __main__.FittedLR = R.FittedLR
    with open(os.path.join(R.V3, "final_models.pkl"), "rb") as f:
        return pickle.load(f)


def g_of_c(C):
    return (np.log10(C) - np.log10(C_LOW)) / (np.log10(C_HIGH) - np.log10(C_LOW))


def expand(Xr, variant, n_filters):
    """Expand a variant's short chromosome to full length [filters | features | C], with filter genes set to 1 (always on)."""
    Xr = np.atleast_2d(Xr)
    ones = np.ones((len(Xr), n_filters))
    if variant == "nofilter":
        return np.hstack([ones, Xr])
    if variant == "seq":
        return np.hstack([ones, Xr, np.full((len(Xr), 1), g_of_c(SEQ_C))])
    return Xr


class VariantProblem(Problem):
    def __init__(self, variant, n_filters, n_features, pool):
        self.variant, self.n_filters, self.pool = variant, n_filters, pool
        n_var = n_features + (1 if variant == "nofilter" else 0)
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.array(self.pool.map(evaluate_one, list(expand(X, self.variant, self.n_filters)), chunksize=1))


def tune_c(mask, folds):
    """Grid-tune C on the inner folds used by the search; returns (C, mean inner AUC)."""
    best = (None, -1.0)
    for C in R.C_GRID:
        aucs = [roc_auc_score(yb, make_lr(C).fit(A[:, mask], ya).decision_function(B[:, mask]))
                for A, ya, B, yb in folds]
        if np.mean(aucs) > best[1]:
            best = (float(C), float(np.mean(aucs)))
    return best


def run_dir(variant, fold):
    return os.path.join(REV, variant, f"fold{fold}")


def stage_search(variant, fold, seeds, n_proc):
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    pool_d = R.load_pool()
    X, y = pool_d["X_pool"], pool_d["y"]
    tr = R.load_splits()["outer_fold"] != fold
    X, y = X[tr], y[tr]
    n_filters, f2f = int(pool_d["n_filters"]), pool_d["feature_to_filter_index"]
    out = run_dir(variant, fold)
    os.makedirs(out, exist_ok=True)
    for seed in seeds:
        path = os.path.join(out, f"{variant}_seed{seed}.npz")
        if os.path.exists(path):
            print(f"{path} exists, skipping", flush=True)
            continue
        t0 = time.time()
        folds = build_inner_folds(X, y, n_splits=5, random_state=seed)
        extra = {}
        with Pool(n_proc, initializer=init_worker, initargs=(folds, n_filters, f2f)) as pool:
            if variant == "random":
                rng = np.random.default_rng(seed)
                Xall = rng.random((R.N_EVAL, n_filters + X.shape[1] + 1))
                Fall = np.array(pool.map(evaluate_one, list(Xall), chunksize=20))
                nd = NonDominatedSorting().do(Fall, only_non_dominated_front=True)
                Xs, F, n_eval = Xall[nd], Fall[nd], R.N_EVAL
            else:
                n_eval = R.N_EVAL if variant == "nofilter" else SEQ_EVAL
                problem = VariantProblem(variant, n_filters, X.shape[1], pool)
                res = minimize(problem, build_algorithm("CSDR", n_obj=2, n_eval=n_eval),
                               get_termination("n_eval", n_eval), seed=seed, verbose=False)
                F, Xs = np.atleast_2d(res.F), expand(res.X, variant, n_filters)
        if variant == "seq":
            # representative (tol rule) chosen with C fixed, then C tuned on the grid; the tuned C is written to the C gene of all front solutions
            i = R.representative(F, "tol")
            mask, _ = decode_solution(Xs[i], n_filters, X.shape[1], f2f)
            C, auc = tune_c(mask, folds)
            Xs[:, -1] = g_of_c(C)
            extra = {"tuned_C": C, "tuned_inner_auc": auc, "c_grid_evals": len(R.C_GRID)}
            n_eval = n_eval + len(R.C_GRID)
        np.savez(path, X=Xs, F=F, variant=variant, seed=seed, n_eval=n_eval, seconds=time.time() - t0, **extra)
        print(f"{variant} fold{fold} seed{seed}: front {len(F)}, best inner AUC {1 - F[:, 0].min():.4f}, "
              f"min n_feat {int(round(F[:, 1].min() * X.shape[1]))}, evals {n_eval}, {time.time() - t0:.0f}s"
              + (f", tuned C {extra['tuned_C']:.3g}" if extra else ""), flush=True)


def load_reps(path_dir, prefix, seeds, n_filters, n_features, f2f):
    masks, Cs = [], []
    for s in seeds:
        r = np.load(os.path.join(path_dir, f"{prefix}_seed{s}.npz"))
        i = R.representative(r["F"], R.MAIN_RULE)
        m, C = decode_solution(r["X"][i], n_filters, n_features, f2f)
        masks.append(m)
        Cs.append(C)
    return np.array(masks), np.array(Cs)


def stage_evaluate(seeds):
    pool_d = R.load_pool()
    X, y = pool_d["X_pool"], pool_d["y"]
    n_filters, f2f = int(pool_d["n_filters"]), pool_d["feature_to_filter_index"]
    s = R.load_splits()
    of, n_outer = s["outer_fold"], int(s["n_outer"])
    ref = np.load(os.path.join(R.V3, "evaluation/nested_oof.npz"))
    assert np.array_equal(ref["y"], y) and np.array_equal(ref["outer_fold"], of)
    oof = {MAIN_KEY: ref[MAIN_KEY.replace("-", "_")]}
    rows, hv_rows = [], []
    for v in VARIANTS:
        key = f"{v}-tol-thr50"
        oof[key] = np.full(len(y), np.nan)
        for k in range(n_outer):
            tr, te = of != k, of == k
            for sd in seeds:
                r = np.load(os.path.join(run_dir(v, k), f"{v}_seed{sd}.npz"))
                hv_rows.append({"variant": v, "fold": k, "seed": sd, "hv": HV(ref_point=R.HV_REF)(r["F"]),
                                "best_inner_auc": 1 - r["F"][:, 0].min(), "front_size": len(r["F"]),
                                "n_eval": int(r["n_eval"])})
            masks, Cs = load_reps(run_dir(v, k), v, seeds, n_filters, X.shape[1], f2f)
            mask, fb, _ = R.consensus(masks, R.MAIN_THRESHOLD)
            oof[key][te] = R.FittedLR(mask, float(np.median(Cs))).fit(X[tr], y[tr]).predict(X[te])
            rows.append({"method": key, "fold": k, "n_features": int(mask.sum()), "fallback": fb,
                         "rep_n_median": float(np.median(masks.sum(1))),
                         "outer_auc": roc_auc_score(y[te], oof[key][te])})
    for k in range(n_outer):
        for sd in seeds:
            F = np.load(os.path.join(R.run_dir("fold", k), f"CSDR_seed{sd}.npz"))["F"]
            hv_rows.append({"variant": "TRIAD-CSDR", "fold": k, "seed": sd, "hv": HV(ref_point=R.HV_REF)(F),
                            "best_inner_auc": 1 - F[:, 0].min(), "front_size": len(F), "n_eval": R.N_EVAL})
    summ, comps = [], []
    for key, p in oof.items():
        lo, hi = R.boot_ci(y, p)
        summ.append({"method": key, "oof_auc": roc_auc_score(y, p), "ci_low": lo, "ci_high": hi})
        if key != MAIN_KEY:
            (dlo, dhi), pv = R.paired_boot(y, oof[MAIN_KEY], p)
            comps.append({"triad": MAIN_KEY, "variant": key,
                          "delta": roc_auc_score(y, oof[MAIN_KEY]) - roc_auc_score(y, p),
                          "ci_low": dlo, "ci_high": dhi, "p": pv, "p_bonferroni": min(pv * len(VARIANTS), 1.0)})
    hdf = pd.DataFrame(hv_rows)
    from scipy.stats import wilcoxon
    main_hv = hdf[hdf.variant == "TRIAD-CSDR"].sort_values(["fold", "seed"]).hv.values
    hv_tests = []
    for v in VARIANTS:
        h = hdf[hdf.variant == v].sort_values(["fold", "seed"]).hv.values
        p = wilcoxon(main_hv, h).pvalue
        hv_tests.append({"variant": v, "hv_mean": h.mean(), "hv_sd": h.std(), "triad_hv_mean": main_hv.mean(),
                         "triad_higher": int((main_hv > h).sum()), "pairs": len(h), "wilcoxon_p": p,
                         "p_bonferroni": min(p * len(VARIANTS), 1.0)})
    out = os.path.join(REV, "evaluation")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(summ).to_csv(os.path.join(out, "variant_summary.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(out, "variant_per_fold.csv"), index=False)
    pd.DataFrame(comps).to_csv(os.path.join(out, "variant_vs_triad.csv"), index=False)
    hdf.to_csv(os.path.join(out, "variant_hv.csv"), index=False)
    pd.DataFrame(hv_tests).to_csv(os.path.join(out, "variant_hv_tests.csv"), index=False)
    np.savez(os.path.join(out, "variant_oof.npz"), y=y, outer_fold=of, **{k.replace("-", "_"): v for k, v in oof.items()})
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(pd.DataFrame(summ))
        print(pd.DataFrame(rows).groupby("method")[["n_features", "outer_auc"]].agg(["median", "mean"]))
        print(pd.DataFrame(comps))
        print(pd.DataFrame(hv_tests))


def stage_final(seeds):
    """Final-configuration checks: dominance by the full-cohort fronts, and re-tuning C within the outer folds."""
    pool_d = R.load_pool()
    X, y = pool_d["X_pool"], pool_d["y"]
    n_filters, f2f = int(pool_d["n_filters"]), pool_d["feature_to_filter_index"]
    fm = load_final()
    final = fm["models"]["thr50"]
    mask, C = final.mask, float(final.C)
    rows = []
    for sd in seeds:
        folds = build_inner_folds(X, y, n_splits=5, random_state=sd)
        aucs = [roc_auc_score(yb, make_lr(C).fit(A[:, mask], ya).decision_function(B[:, mask])) for A, ya, B, yb in folds]
        f = np.array([1 - np.mean(aucs), mask.sum() / len(mask)])
        F = np.load(os.path.join(R.run_dir("full"), f"{fm['algorithm']}_seed{sd}.npz"))["F"]
        dominated_by = int(((F <= f).all(1) & (F < f).any(1)).sum())
        rows.append({"seed": sd, "final_inner_auc": 1 - f[0], "final_n": int(mask.sum()),
                     "front_best_auc_at_or_below_n": float((1 - F[F[:, 1] <= f[1] + 1e-12, 0]).max())
                     if (F[:, 1] <= f[1] + 1e-12).any() else np.nan,
                     "n_front_solutions_dominating": dominated_by, "front_size": len(F)})
    bdf = pd.DataFrame(rows)
    # within outer folds: consensus mask + C tuned on the training folds (5 folds, random_state=0)
    s = R.load_splits()
    of, n_outer = s["outer_fold"], int(s["n_outer"])
    p_med, p_tuned, crow = np.full(len(y), np.nan), np.full(len(y), np.nan), []
    for k in range(n_outer):
        tr, te = of != k, of == k
        masks, Cs = R.load_representatives(R.run_dir("fold", k), fm["algorithm"], seeds, R.MAIN_RULE,
                                           n_filters, X.shape[1], f2f)
        m, _, _ = R.consensus(masks, R.MAIN_THRESHOLD)
        p_med[te] = R.FittedLR(m, float(np.median(Cs))).fit(X[tr], y[tr]).predict(X[te])
        inner = R.build_inner_folds(X[tr], y[tr], n_splits=5, random_state=0)
        Ct, _ = tune_c(m, inner)
        p_tuned[te] = R.FittedLR(m, Ct).fit(X[tr], y[tr]).predict(X[te])
        crow.append({"fold": k, "n": int(m.sum()), "median_C": float(np.median(Cs)), "tuned_C": Ct})
    (lo, hi), pv = R.paired_boot(y, p_tuned, p_med)
    out = os.path.join(REV, "evaluation")
    os.makedirs(out, exist_ok=True)
    bdf.to_csv(os.path.join(out, "final_config_dominance.csv"), index=False)
    pd.DataFrame(crow).to_csv(os.path.join(out, "final_config_retune_c.csv"), index=False)
    res = {"oof_median_C": roc_auc_score(y, p_med), "oof_tuned_C": roc_auc_score(y, p_tuned),
           "delta_tuned_minus_median": roc_auc_score(y, p_tuned) - roc_auc_score(y, p_med),
           "ci_low": lo, "ci_high": hi, "p": pv}
    pd.DataFrame([res]).to_csv(os.path.join(out, "final_config_retune_c_summary.csv"), index=False)
    print(bdf)
    print(pd.DataFrame(crow))
    print(res)


def vif(A):
    A = (A - A.mean(0)) / A.std(0)
    inv = np.linalg.pinv(np.corrcoef(A, rowvar=False))
    return np.diag(inv)


def stage_ablation():
    """Leave-out ablation of the final model (refit on Duke, evaluated on I-SPY) + redundancy on Duke."""
    pool_d = R.load_pool()
    X, y, names = pool_d["X_pool"], pool_d["y"], pool_d["feature_names"]
    f2f = pool_d["feature_to_filter_index"]
    fm = load_final()
    final = fm["models"]["thr50"]
    mask, C = final.mask.copy(), float(final.C)
    idx = np.flatnonzero(mask)
    kin = f2f == (int(pool_d["n_filters"]) - 1)
    variants = {"full_model": mask}
    for j in idx:
        m = mask.copy()
        m[j] = False
        variants[f"drop:{names[j]}"] = m
    variants["drop:kinetic_group"] = mask & ~kin
    variants["drop:radiomic_group"] = mask & kin
    ext = {c: R.load_external(c, "medsam2", names) for c in ["ISPY1", "ISPY2"]}
    rows = []
    for key, m in variants.items():
        if not m.any():
            continue
        model = final if key == "full_model" else R.FittedLR(m, C).fit(X, y)
        for c, (Xe, ye, _) in ext.items():
            p = model.predict(Xe)
            rows.append({"variant": key, "n_features": int(m.sum()), "cohort": c, "auc": roc_auc_score(ye, p), "_p": p})
    df = pd.DataFrame(rows)
    out_rows = []
    for c, (Xe, ye, _) in ext.items():
        base = df[(df.variant == "full_model") & (df.cohort == c)]._p.iloc[0]
        for _, r in df[df.cohort == c].iterrows():
            d = {k: r[k] for k in ["variant", "n_features", "cohort", "auc"]}
            if r.variant != "full_model":
                (lo, hi), pv = R.paired_boot(ye, base, r._p)
                d.update({"delta_full_minus_variant": roc_auc_score(ye, base) - r.auc, "ci_low": lo, "ci_high": hi,
                          "p": pv})
            out_rows.append(d)
    res = pd.DataFrame(out_rows)
    A = R.fit_transform_pre(X[:, idx])[0]
    rho = pd.DataFrame(A, columns=names[idx]).corr(method="spearman")
    v = pd.DataFrame({"feature": names[idx], "kinetic": kin[idx], "vif": vif(A),
                      "max_abs_spearman_with_other": (rho.abs() - np.eye(len(idx))).max(1).values})
    out = os.path.join(REV, "evaluation")
    os.makedirs(out, exist_ok=True)
    res.to_csv(os.path.join(out, "feature_ablation_external.csv"), index=False)
    rho.to_csv(os.path.join(out, "final_features_spearman.csv"))
    v.to_csv(os.path.join(out, "final_features_vif.csv"), index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 100, "display.max_colwidth", 60):
        print(res)
        print(v)


MAMAMIA = os.path.join(MAMAMIA_DIR, "clinical_and_imaging_info.xlsx")


def mamamia_id(cohort, pid):
    num = str(pid).split("-")[-1].split("_")[-1]
    return f"DUKE_{int(num):03d}" if cohort == "DUKE" else f"{cohort}_{num}"


def mamamia_label_check(cohort, pids, y):
    mm = pd.read_excel(MAMAMIA, sheet_name="dataset_info").set_index("patient_id")["hr"].dropna()
    ids = [mamamia_id(cohort, p) for p in pids]
    keep = [i for i, m in enumerate(ids) if m in mm.index]
    agree = sum(int(mm[ids[i]]) == int(y[i]) for i in keep)
    return agree, len(keep)


def stage_audit():
    """External-validation checks."""
    pool_d = R.load_pool()
    names = pool_d["feature_names"]
    fm = load_final()
    lines = []
    assert np.array_equal(fm["feature_names"], names)
    lines.append(f"final_models.pkl feature_names == Duke pool feature_names ({len(names)} columns): OK")
    ok, n = mamamia_label_check("DUKE", pool_d["patient_ids"], pool_d["y"])
    lines.append(f"Duke: labels agree with independent MAMA-MIA hr for {ok}/{n} overlapping patients")
    scores = pd.read_csv(os.path.join(R.V3, "evaluation/external_case_scores.csv"))
    for c in ["ISPY1", "ISPY2"]:
        d = R.V3 + "/data"
        r = np.load(os.path.join(d, f"{c.lower()}_radiomics_medsam2.npz"), allow_pickle=True)
        k = np.load(os.path.join(d, f"{c.lower()}_kinetic_medsam2.npz"), allow_pickle=True)
        rad_names, kin_names = list(r["feature_names"]), list(k["feature_names"])
        same = rad_names + kin_names == list(names)
        lines.append(f"{c}: external radiomic+kinetic column names identical to Duke order before reindexing: {same}; "
                     f"set-equal: {set(rad_names + kin_names) == set(names)}")
        Xe, ye, pids = R.load_external(c, "medsam2", names)
        meta = pd.read_csv(R.ISPY_META[c]).set_index("pid")
        if {"ER", "PR"} <= set(meta.columns):
            er, pr, hr = meta.loc[pids, "ER"], meta.loc[pids, "PR"], meta.loc[pids, "HR"]
            rule = ((er == 1) | (pr == 1)).astype(int)
            lines.append(f"{c}: HR == (ER+ or PR+) for {(rule.values == hr.values).sum()}/{len(hr)} patients")
        else:
            lines.append(f"{c}: BreastDCEDL metadata provides HR only (no ER/PR columns)")
        lines.append(f"{c}: HR+ prevalence {ye.mean():.3f}; Duke HR+ prevalence {pool_d['y'].mean():.3f}")
        mm_ok, mm_n = mamamia_label_check(c, pids, ye)
        lines.append(f"{c}: labels agree with independent MAMA-MIA hr for {mm_ok}/{mm_n} overlapping patients")
        p = fm["models"]["thr50"].predict(Xe)
        sc = scores[scores.cohort == c].set_index("pid").loc[pids]
        lines.append(f"{c}: reloaded model reproduces stored per-case predictions: max |diff| = "
                     f"{np.abs(sc.prob.values - p).max():.2e}; labels identical: {np.array_equal(sc.hr.values, ye)}; "
                     f"AUC {roc_auc_score(ye, p):.4f}")
    out = os.path.join(REV, "evaluation")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "external_audit.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["search", "evaluate", "final", "ablation", "audit"])
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--n-proc", type=int, default=8)
    a = ap.parse_args()
    seeds = R.parse_seeds(a.seeds)
    if a.stage == "search":
        stage_search(a.variant, a.fold, seeds, a.n_proc)
    elif a.stage == "evaluate":
        stage_evaluate(seeds)
    elif a.stage == "final":
        stage_final(seeds)
    elif a.stage == "ablation":
        stage_ablation()
    else:
        stage_audit()
