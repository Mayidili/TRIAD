"""
Segmentation quality check: MedSAM2 masks (and the official I-SPY FTV masks) against the MAMA-MIA expert masks.

Alignment: the MAMA-MIA post-contrast image of the phase we segmented differs from our image only by an axis
permutation and flips. Duke and I-SPY1 use the first post-contrast phase (<case>_0001); the BreastDCEDL "early"
phase of I-SPY2 is the second post-contrast phase, so <case>_0002 is used (r = 1.000).
For each case the transform among 6 axis permutations x 8 flips with the highest correlation to our image
(same shape required) is chosen and applied to the expert mask; cases with correlation < 0.99 are flagged as
alignment failures and excluded.

Our data:
  Duke   <data>/seg_images/<pid>_image.nii.gz + seg_masks/<pid>_mask.nii.gz (voxel grid as in DICOM);
         z spacing taken from duke_slice_spacing.csv
  I-SPY  <run>/data/<cohort>/{early,mask_medsam2,mask_official}/<pid>.nii.gz (true spacing in the header)
Output: <seg_qa>/mamamia_agreement.csv (per case) and summary.csv
"""
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

from prism.paths import DATA_ROOT, MAMAMIA_DIR, RUN_DATA, SEG_QA_DIR

MM = MAMAMIA_DIR
PD = DATA_ROOT
FIX = RUN_DATA
OUT = SEG_QA_DIR


def arr(p):
    return sitk.GetArrayFromImage(sitk.ReadImage(p))


def cases():
    out = []
    for c in sorted(os.listdir(os.path.join(MM, "images"))):
        coh, num = c.split("_", 1)
        if coh == "DUKE":
            pid = f"Breast_MRI_{int(num):03d}"
            out.append((c, "Duke", pid))
        elif coh in ("ISPY1", "ISPY2"):
            d = os.path.join(FIX, coh, "early")
            cand = [f"{coh}_{num}", f"ACRIN-6698-{num}", f"ISPY2-{num}"]
            pid = next((p for p in cand if os.path.exists(os.path.join(d, f"{p}.nii.gz"))), None)
            if pid:
                out.append((c, coh, pid))
    return out


def duke_spacing():
    sp = pd.read_csv(os.path.join(FIX, "duke_slice_spacing.csv")).set_index("pid")
    return sp


def best_transform(ours, theirs):
    best = (-2.0, None, None)
    for perm in itertools.permutations(range(3)):
        t = np.transpose(theirs, perm)
        if t.shape != ours.shape:
            continue
        for fl in itertools.product([0, 1], repeat=3):
            ax = tuple(i for i in range(3) if fl[i])
            b = np.flip(t, ax) if ax else t
            s = tuple(slice(None, None, max(1, n // 64)) for n in ours.shape)
            r = np.corrcoef(ours[s].ravel().astype(np.float64), b[s].ravel().astype(np.float64))[0, 1]
            if r > best[0]:
                best = (r, perm, ax)
    return best


def apply(x, perm, ax):
    t = np.transpose(x, perm)
    return np.flip(t, ax) if ax else t


def metrics(a, b, spacing_zyx):
    va, vb = a.sum(), b.sum()
    vox = float(np.prod(spacing_zyx)) / 1000.0  # cm^3
    out = {"dice": 2 * (a & b).sum() / (va + vb) if va + vb else np.nan,
           "vol_ours_cm3": va * vox, "vol_expert_cm3": vb * vox, "vol_ratio": va / vb if vb else np.nan,
           "expert_covered": (a & b).sum() / vb if vb else np.nan}
    if va and vb:
        ca, cb = np.array(ndimage.center_of_mass(a)), np.array(ndimage.center_of_mass(b))
        out["centroid_dist_mm"] = float(np.linalg.norm((ca - cb) * np.array(spacing_zyx)))
    return out


def one(args):
    case, coh, pid = args
    try:
        phase = "0002" if coh == "ISPY2" else "0001"
        theirs = arr(os.path.join(MM, "images", case, f"{case}_{phase}.nii.gz"))
        expert = arr(os.path.join(MM, "segmentations/expert", f"{case}.nii.gz")) > 0
        if coh == "Duke":
            ours = arr(os.path.join(PD, "seg_images", f"{pid}_image.nii.gz"))
            masks = {"medsam2": arr(os.path.join(PD, "seg_masks", f"{pid}_mask.nii.gz")) > 0}
            sp = sitk.ReadImage(os.path.join(PD, "seg_images", f"{pid}_image.nii.gz")).GetSpacing()[::-1]
            if pid in SP.index:  # actual slice spacing
                sp = (float(SP.loc[pid, "dz"]), sp[1], sp[2])
        else:
            img = sitk.ReadImage(os.path.join(FIX, coh, "early", f"{pid}.nii.gz"))
            ours, sp = sitk.GetArrayFromImage(img), img.GetSpacing()[::-1]
            masks = {k: arr(os.path.join(FIX, coh, f"mask_{k}", f"{pid}.nii.gz")) > 0
                     for k in ["medsam2", "official"] if os.path.exists(os.path.join(FIX, coh, f"mask_{k}", f"{pid}.nii.gz"))}
        r, perm, ax = best_transform(ours, theirs)
        row = {"case": case, "cohort": coh, "pid": pid, "phase": phase, "align_r": r, "perm": str(perm), "flip_axes": str(ax)}
        if perm is None or r < 0.99:
            row["status"] = "align_failed"
            return [row]
        e = apply(expert, perm, ax)
        rows = []
        for k, m in masks.items():
            rows.append({**row, "status": "ok", "mask": k, **metrics(m, e, sp)})
        return rows
    except Exception as ex:
        return [{"case": case, "cohort": coh, "pid": pid, "status": f"error: {ex!r}"[:200]}]


def summarize(df):
    ok = df[df.status == "ok"]
    g = ok.groupby(["cohort", "mask"])
    s = g.agg(n=("dice", "size"), dice_mean=("dice", "mean"), dice_median=("dice", "median"),
              dice_q1=("dice", lambda x: x.quantile(0.25)), dice_q3=("dice", lambda x: x.quantile(0.75)),
              vol_ratio_median=("vol_ratio", "median"), expert_covered_median=("expert_covered", "median"),
              centroid_mm_median=("centroid_dist_mm", "median"),
              dice_lt_0_5=("dice", lambda x: (x < 0.5).mean()), missed=("expert_covered", lambda x: (x == 0).mean()))
    return s.reset_index()


SP = None

if __name__ == "__main__":
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    os.makedirs(OUT, exist_ok=True)
    SP = duke_spacing()
    todo = cases()
    print(f"{len(todo)} cases with images", flush=True)
    with Pool(n_proc) as p:
        rows = [r for rs in p.imap_unordered(one, todo, chunksize=4) for r in rs]
    df = pd.DataFrame(rows).sort_values(["cohort", "case"])
    df.to_csv(os.path.join(OUT, "mamamia_agreement.csv"), index=False)
    print(df.status.value_counts())
    s = summarize(df)
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(s)
