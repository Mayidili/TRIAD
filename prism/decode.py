"""
Chromosome decoding helpers.
All genes are real-valued in [0, 1]; the functions below map them to filter selections,
feature masks and classifier hyperparameters.
"""
import numpy as np


def linear_decode(g, low, high):
    return low + g * (high - low)


def log_decode(g, low, high):
    return 10.0 ** (np.log10(low) + g * (np.log10(high) - np.log10(low)))


def decode_filter_segment(g_filter, threshold=0.5):
    """
    g_filter: shape (n_filters,), values in [0, 1].
    Returns a boolean mask of the selected filter configurations.

    If all genes are below threshold, the gene with the largest value is selected so that
    the candidate feature pool is never empty.
    """
    active = g_filter > threshold
    if not active.any():
        active = np.zeros_like(g_filter, dtype=bool)
        active[np.argmax(g_filter)] = True
    return active


def decode_feature_mask(g_feat, active_filter_mask, feature_to_filter_index, threshold=0.5):
    """
    g_feat: shape (n_features,), values in [0, 1].
    active_filter_mask: shape (n_filters,) boolean array from decode_filter_segment.
    feature_to_filter_index: shape (n_features,) int array; entry j is the filter
        configuration of feature j (a feature can only be selected if its filter is active).

    Returns: shape (n_features,) boolean array of selected features.
    """
    eligible = active_filter_mask[feature_to_filter_index]
    selected = (g_feat > threshold) & eligible
    return selected


# ---- classifier hyperparameter decoding (six XGBoost hyperparameters) ----
XGB_HP_ORDER = ["max_depth", "learning_rate", "n_estimators",
                "subsample", "colsample_bytree", "reg_lambda"]

XGB_HP_SPEC = {
    #            (mapping,       low,   high)
    "max_depth":        ("linear_int", 2,     8),
    "learning_rate":    ("log",        1e-2,  3e-1),
    "n_estimators":     ("linear_int", 50,    500),
    "subsample":        ("linear",     0.5,   1.0),
    "colsample_bytree": ("linear",     0.5,   1.0),
    "reg_lambda":       ("log",        1e-3,  10.0),
}


def decode_classifier_hyperparams(g_hp):
    """g_hp: shape (6,), values in [0, 1], ordered as XGB_HP_ORDER."""
    assert len(g_hp) == len(XGB_HP_ORDER)
    params = {}
    for name, g in zip(XGB_HP_ORDER, g_hp):
        kind, low, high = XGB_HP_SPEC[name]
        if kind == "linear":
            params[name] = float(linear_decode(g, low, high))
        elif kind == "linear_int":
            params[name] = int(round(linear_decode(g, low, high)))
        elif kind == "log":
            params[name] = float(log_decode(g, low, high))
        else:
            raise ValueError(kind)
    return params
