"""Feature selectors used by the mRMR and ElasticNet staged baselines."""
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler


def mrmr_select(X, y, k=30, random_state=0):
    """
    Greedy minimum-redundancy maximum-relevance (mRMR) selection: start with the feature of highest
    mutual information with the label, then repeatedly add the feature that is relevant to the label
    but has the lowest mean correlation with the features already selected, until k are selected.
    k=30 is a fixed value typical for mRMR pre-selection in radiomics, not a tuned one.
    """
    mi = mutual_info_classif(X, y, random_state=random_state)
    selected = [int(np.argmax(mi))]
    remaining = set(range(X.shape[1])) - set(selected)
    Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    while len(selected) < k and remaining:
        sel_arr = np.array(selected)
        best_feat, best_score = None, -np.inf
        for f in remaining:
            redundancy = np.mean(np.abs(Xn[:, sel_arr].T @ Xn[:, f]) / X.shape[0])
            score = mi[f] - redundancy
            if score > best_score:
                best_score, best_feat = score, f
        selected.append(best_feat)
        remaining.remove(best_feat)

    mask = np.zeros(X.shape[1], dtype=bool)
    mask[selected] = True
    return mask


def elasticnet_select(X, y, random_state=0, max_iter=5000):
    """l1_ratio chosen by cross-validation (0.1 to 1.0; 1.0 is plain LASSO).
    Features are standardised before fitting, as for the LASSO baseline; without standardisation the
    very different scales of PyRadiomics features distort the regularisation path."""
    X_scaled = StandardScaler().fit_transform(X)
    enet = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
        cv=5, random_state=random_state, max_iter=max_iter, n_jobs=-1,
    ).fit(X_scaled, y.astype(float))
    print(f"selected l1_ratio={enet.l1_ratio_}, alpha={enet.alpha_:.6f}", flush=True)
    selected = np.abs(enet.coef_) > 1e-6
    if selected.sum() == 0:
        selected = np.ones(X.shape[1], dtype=bool)
    return selected
