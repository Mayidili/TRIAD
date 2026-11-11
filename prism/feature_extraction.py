"""
One-off offline feature extraction for the filter-configuration pool.
Input: a manifest (one case per row: patient_id, image_path, mask_path, label).
Output: X_pool (n_samples, n_features), feature_names, feature_to_filter_index,
        patient_ids and y, cached to disk; the evolutionary search only reads this cache.

Filter-configuration pool (18 derived images):
  original (1) + wavelet (8 sub-bands) + LoG (5 scales) + Square/SquareRoot/Logarithm/Exponential (4)
"""
import numpy as np
import pandas as pd
import radiomics
from radiomics.featureextractor import RadiomicsFeatureExtractor

LOG_SIGMAS = [1.0, 2.0, 3.0, 4.0, 5.0]


def _log_prefix(sigma):
    # PyRadiomics names LoG images like "log-sigma-1-0-mm-3D" (decimal point replaced by "-")
    return f"log-sigma-{sigma:.1f}".replace(".", "-") + "-mm-3D"


# PyRadiomics feature-name prefixes of the 18 filter configurations, in the fixed order used by
# feature_to_filter_index. They must match the image-type prefixes PyRadiomics generates exactly;
# otherwise features end up in the wrong filter bucket and filter selection loses its meaning.
FILTER_NAMES = (
    ["original"]
    + [f"wavelet-{band}" for band in ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]]
    + [_log_prefix(s) for s in LOG_SIGMAS]
    + ["square", "squareroot", "logarithm", "exponential"]
)
N_FILTERS = len(FILTER_NAMES)  # 18
_FILTER_NAME_TO_INDEX = {name: idx for idx, name in enumerate(FILTER_NAMES)}


def build_extractor():
    """
    MRI intensities have no absolute physical scale (unlike CT Hounsfield units), so grey levels are
    discretised with a fixed bin count (binCount) rather than a fixed bin width; normalize applies a
    z-score before discretisation, as is standard for MRI radiomics.
    """
    settings = {
        "binCount": 32,
        "resampledPixelSpacing": [1, 1, 1],
        "interpolator": "sitkBSpline",
        "normalize": True,
        "normalizeScale": 100,
    }
    extractor = RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.enableImageTypeByName("Wavelet")
    extractor.enableImageTypeByName("LoG", customArgs={"sigma": LOG_SIGMAS})
    extractor.enableImageTypeByName("Square")
    extractor.enableImageTypeByName("SquareRoot")
    extractor.enableImageTypeByName("Logarithm")
    extractor.enableImageTypeByName("Exponential")
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("gldm")
    extractor.enableFeatureClassByName("ngtdm")
    extractor.enableFeatureClassByName("shape")
    return extractor


def _assign_filter_index(feature_key):
    """
    PyRadiomics feature names look like "original_shape_Sphericity", "wavelet-LLH_glcm_Correlation",
    "log-sigma-1-0-mm-3D_firstorder_Mean", "square_glrlm_..."; the part before the first underscore is
    the image type and is matched exactly against FILTER_NAMES. Shape features are only computed on the
    original image, so they always map to index 0.
    """
    prefix = feature_key.split("_")[0]
    if prefix not in _FILTER_NAME_TO_INDEX:
        raise ValueError(f"Unrecognised filter prefix for feature: {feature_key} (prefix={prefix})")
    return _FILTER_NAME_TO_INDEX[prefix]


def extract_case(image_path, mask_path, extractor=None):
    extractor = extractor or build_extractor()
    result = extractor.execute(image_path, mask_path)
    features = {
        k: float(v) for k, v in result.items()
        if not k.startswith("diagnostics_")
    }
    return features


def _extract_one(args):
    patient_id, image_path, mask_path = args
    try:
        features = extract_case(image_path, mask_path)
        return patient_id, features, None
    except Exception as e:  # noqa: BLE001
        return patient_id, None, f"{type(e).__name__}: {e}"


def build_corpus(manifest_csv, out_npz, extractor=None, n_jobs=1):
    """
    manifest_csv columns: patient_id, image_path, mask_path, label
    With n_jobs > 1 extraction runs in parallel processes. Extractors cannot be shared across
    processes, so the extractor argument is ignored in that case and build_extractor() is used.
    """
    manifest = pd.read_csv(manifest_csv)

    if n_jobs <= 1:
        extractor = extractor or build_extractor()
        rows = []
        canonical_columns = None
        kept_ids, kept_labels = [], []
        for _, row in manifest.iterrows():
            try:
                features = extract_case(row["image_path"], row["mask_path"], extractor)
            except Exception as e:  # noqa: BLE001
                print(f"[{row['patient_id']}] FAILED: {type(e).__name__}: {e}")
                continue
            if canonical_columns is None:
                canonical_columns = sorted(features.keys())
            else:
                assert sorted(features.keys()) == canonical_columns, (
                    f"Feature columns of case {row['patient_id']} differ from the first case; check the extractor settings"
                )
            rows.append([features[c] for c in canonical_columns])
            kept_ids.append(row["patient_id"])
            kept_labels.append(row["label"])
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = [(row["patient_id"], row["image_path"], row["mask_path"]) for _, row in manifest.iterrows()]
        label_by_id = {row["patient_id"]: row["label"] for _, row in manifest.iterrows()}
        results = {}
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(_extract_one, t): t[0] for t in tasks}
            n_done = 0
            for fut in as_completed(futures):
                pid, features, error = fut.result()
                n_done += 1
                if error is not None:
                    print(f"[{pid}] FAILED: {error}")
                else:
                    results[pid] = features
                if n_done % 20 == 0:
                    print(f"progress: {n_done}/{len(tasks)}", flush=True)

        canonical_columns = None
        rows, kept_ids, kept_labels = [], [], []
        for row in manifest.itertuples():
            pid = row.patient_id
            if pid not in results:
                continue
            features = results[pid]
            if canonical_columns is None:
                canonical_columns = sorted(features.keys())
            else:
                assert sorted(features.keys()) == canonical_columns, (
                    f"Feature columns of case {pid} differ from the first case; check the extractor settings"
                )
            rows.append([features[c] for c in canonical_columns])
            kept_ids.append(pid)
            kept_labels.append(label_by_id[pid])

    X_pool = np.array(rows, dtype=np.float64)
    feature_to_filter_index = np.array([_assign_filter_index(c) for c in canonical_columns], dtype=np.int64)
    patient_ids = np.array(kept_ids)
    y = np.array(kept_labels).astype(np.int64)

    np.savez(
        out_npz,
        X_pool=X_pool,
        feature_names=np.array(canonical_columns),
        feature_to_filter_index=feature_to_filter_index,
        n_filters=N_FILTERS,
        patient_ids=patient_ids,
        y=y,
    )
    return X_pool, canonical_columns, feature_to_filter_index, patient_ids, y
