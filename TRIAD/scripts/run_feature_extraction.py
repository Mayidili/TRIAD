import os
import sys

from prism.feature_extraction import build_corpus
from prism.paths import DATA_ROOT

MANIFEST = os.path.join(DATA_ROOT, "duke_feature_manifest.csv")
OUT_NPZ = os.path.join(DATA_ROOT, "duke_feature_pool.npz")

n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 28

X_pool, feature_names, feature_to_filter_index, patient_ids, y = build_corpus(
    MANIFEST, OUT_NPZ, n_jobs=n_jobs
)
print(f"\nFINISHED: X_pool shape={X_pool.shape}, n_patients={len(patient_ids)}, "
      f"label distribution={dict(zip(*__import__('numpy').unique(y, return_counts=True)))}")
print(f"saved to {OUT_NPZ}")
