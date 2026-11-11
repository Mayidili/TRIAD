"""
DCE kinetic features inside the tumour mask (same definition for Duke and I-SPY).

Three phases: S0 = pre-contrast, S1 = early post-contrast (Duke post_1 / I-SPY post_early),
SL = late post-contrast (last Duke post phase / I-SPY post_late). All three must share the same voxel grid.

Voxel-wise maps (computed inside the mask only):
  PE_early = (S1 - S0) / S0'        early percentage enhancement
  PE_late  = (SL - S0) / S0'        late percentage enhancement
  SER      = (S1 - S0) / (SL - S0)  signal enhancement ratio, enhancing voxels only, clipped to [0, 3]
  DELTA    = (SL - S1) / S1'        relative early-to-late change (< -10% washout, > +10% persistent)
S0' = max(S0, 0.05 * 99th percentile of the whole pre-contrast image), to avoid amplifying noise in dark voxels.
Enhancing voxels: PE_early >= 0.5; if fewer than 10, the upper half of PE_early is used instead.

Feature names follow the PyRadiomics style "kinetic_<group>_<name>" and are all assigned to the
19th filter configuration, "kinetic".
"""
import numpy as np
from scipy.stats import kurtosis, skew

ENH_THRESHOLD = 0.5
WASHOUT_THRESHOLD = 0.1
SER_CLIP = 3.0
S0_FLOOR_FRAC = 0.05

_MAPS = ["PEearly", "PElate", "SER", "DELTA"]
_STATS = ["Mean", "Median", "Std", "P10", "P25", "P75", "P90", "Skewness", "Kurtosis"]
_EXTRA = ["EnhancingFraction", "WashoutFraction", "PlateauFraction", "PersistentFraction",
          "EnhancingVolume", "WashoutVolume", "HotspotPEearly", "HotspotSER"]

FEATURE_NAMES = [f"kinetic_{m}_{s}" for m in _MAPS for s in _STATS] + [f"kinetic_summary_{e}" for e in _EXTRA]


def _stats(v):
    if v.size < 3:
        return [np.nan] * len(_STATS)
    p10, p25, p75, p90 = np.percentile(v, [10, 25, 75, 90])
    return [v.mean(), np.median(v), v.std(), p10, p25, p75, p90, float(skew(v)), float(kurtosis(v))]


def kinetic_features(s0, s1, sl, mask, voxel_volume_mm3):
    """s0/s1/sl: float arrays of equal shape; mask: bool of the same shape. Returns a 1-D array aligned with FEATURE_NAMES."""
    assert s0.shape == s1.shape == sl.shape == mask.shape, (s0.shape, s1.shape, sl.shape, mask.shape)
    floor = S0_FLOOR_FRAC * np.percentile(s0, 99)
    m = mask > 0
    a0, a1, al = s0[m].astype(np.float64), s1[m].astype(np.float64), sl[m].astype(np.float64)
    b0 = np.maximum(a0, floor)
    pe_early = (a1 - a0) / b0
    pe_late = (al - a0) / b0
    delta = (al - a1) / np.maximum(a1, floor)

    enh = pe_early >= ENH_THRESHOLD
    if enh.sum() < 10:
        enh = pe_early >= np.median(pe_early)
    denom = al[enh] - a0[enh]
    ser = np.clip((a1[enh] - a0[enh]) / np.where(np.abs(denom) < 1e-6, 1e-6, denom), 0.0, SER_CLIP)
    d_enh = delta[enh]

    feats = []
    for v in (pe_early, pe_late, ser, d_enh):
        feats += _stats(v)

    n_enh = int(enh.sum())
    washout = d_enh < -WASHOUT_THRESHOLD
    persistent = d_enh > WASHOUT_THRESHOLD
    plateau = ~washout & ~persistent
    hot = pe_early >= np.percentile(pe_early, 90)
    hot_enh = hot[enh]
    feats += [
        n_enh / m.sum(),
        washout.mean(), plateau.mean(), persistent.mean(),
        n_enh * voxel_volume_mm3,
        washout.sum() * voxel_volume_mm3,
        pe_early[hot].mean(),
        ser[hot_enh].mean() if hot_enh.any() else np.nan,
    ]
    out = np.array(feats, dtype=np.float64)
    assert out.shape == (len(FEATURE_NAMES),)
    return out
