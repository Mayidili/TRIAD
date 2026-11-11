"""
I-SPY geometry preparation. The BreastDCEDL NIfTI files all have an identity affine and pixdim=1, with arrays
in (column, row, slice) order.

Two stages:
  native    write the early post-contrast image in the original acquisition orientation (slice, row, column),
            the official mask and the MedSAM2 prompt boxes (json), for MedSAM2 segmentation along the
            acquired slices (segment_ispy_medsam2.py).
  finalize  reorder the pre / early / late phases, the official mask and the MedSAM2 mask into the same
            anatomical order as Duke (superior-inferior Z, anterior-posterior Y, left-right X), set the true
            voxel spacing (TCIA DICOM header; BreastDCEDL metadata when missing), and write them to
            <run>/data/<cohort>/{early,pre,late,mask_official,mask_medsam2}/<pid>.nii.gz.

Axis assignment: native array = (slice, row, column); row direction = IOP[3:6], column direction = IOP[0:3],
slice direction = their cross product. Each direction is mapped to the LPS axis with the largest absolute
component (0 = X left-right, 1 = Y anterior-posterior, 2 = Z superior-inferior), then reordered to (Z, Y, X).
Prompt box: the official mask (enhancement-threshold FTV, often fragmented) is closed in 3D (radius 2 voxels)
and the bounding box of the largest connected component is used; the key slice is the middle of that
component's slice range, as with the Duke annotation boxes.
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

from prism.paths import DATA_ROOT, INTERIM_DIR, RUN_DATA

ISPY = os.path.join(DATA_ROOT, "ISPY")
V3 = INTERIM_DIR
FIX = RUN_DATA
META = {"ISPY1": f"{ISPY}/ISPY1/BreastDCEDL_spy1_metadata.csv", "ISPY2": f"{ISPY}/ISPY2/BreastDCEDL_ISPY2_metadata.csv"}
DEFAULT_IOP = {"ISPY1": [0, 1, 0, 0, 0, -1], "ISPY2": [1, 0, 0, 0, 1, 0]}  # most common orientation in each cohort, used when the DICOM header is missing


def read(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def to_native(a):
    """BreastDCEDL array (column, row, slice) -> (slice, row, column)"""
    return np.ascontiguousarray(a.transpose(2, 1, 0))


def geometry(cohort):
    geo = pd.read_csv(os.path.join(FIX, f"{cohort.lower()}_geometry.csv")).set_index("pid")
    meta = pd.read_csv(META[cohort]).set_index("pid")
    out = {}
    for pid in sorted(f[: -len("_image.nii.gz")] for f in os.listdir(f"{ISPY}/{cohort}_extracted/images")):
        g = geo.loc[pid] if pid in geo.index and geo.loc[pid, "status"] == "ok" else None
        if g is not None:
            ps = float(g.ps_row)
            dz = float(g.sbs) if pd.notna(g.sbs) else float(g.thk)
            iop = json.loads(g.iop)
            src = "dicom"
        else:
            ps, dz = float(meta.loc[pid, "xy_spacing"]), float(meta.loc[pid, "slice_thick"])
            iop, src = DEFAULT_IOP[cohort], "metadata"
        if cohort == "ISPY2" and pid in meta.index and g is not None:
            # metadata takes precedence (it records the slice spacing, which matches DICOM SpacingBetweenSlices on spot checks)
            ps, dz = float(meta.loc[pid, "xy_spacing"]), float(meta.loc[pid, "slice_thick"])
        out[pid] = {"ps": ps, "dz": dz, "iop": iop, "src": src}
    return out


def anat_perm(iop):
    """LPS anatomical axis of each native (slice, row, column) axis; returns the transpose order from native to (Z, Y, X)."""
    col_dir = np.array(iop[:3], float)
    row_dir = np.array(iop[3:], float)
    sl_dir = np.cross(col_dir, row_dir)
    anat = [int(np.argmax(np.abs(d))) for d in (sl_dir, row_dir, col_dir)]  # native axis -> anatomical axis
    assert sorted(anat) == [0, 1, 2], (iop, anat)
    return [anat.index(k) for k in (2, 1, 0)], anat


def prompt_box(mask_native):
    closed = ndimage.binary_closing(mask_native, structure=ndimage.generate_binary_structure(3, 1), iterations=2)
    lab, n = ndimage.label(closed)
    if n == 0:
        return None
    sizes = ndimage.sum(closed, lab, range(1, n + 1))
    comp = lab == (int(np.argmax(sizes)) + 1)
    s, r, c = [np.flatnonzero(comp.any(axis=tuple(a for a in range(3) if a != ax))) for ax in range(3)]
    return {"start_slice": int(s.min()), "end_slice": int(s.max()), "start_row": int(r.min()), "end_row": int(r.max()),
            "start_col": int(c.min()), "end_col": int(c.max()), "n_components": int(n)}


def _native_one(args):
    cohort, pid = args
    out = os.path.join(FIX, cohort, "native")
    img = to_native(read(f"{ISPY}/{cohort}_extracted/images/{pid}_image.nii.gz")).astype(np.float32)
    msk = to_native(read(f"{ISPY}/{cohort}_extracted/masks/{pid}_mask.nii.gz") > 0)
    box = prompt_box(msk)
    sitk.WriteImage(sitk.GetImageFromArray(img), os.path.join(out, f"{pid}_early.nii.gz"))
    return pid, box


def stage_native(cohort, n_proc):
    out = os.path.join(FIX, cohort, "native")
    os.makedirs(out, exist_ok=True)
    pids = sorted(geometry(cohort))
    with Pool(n_proc) as p:
        res = p.map(_native_one, [(cohort, x) for x in pids])
    boxes = {pid: box for pid, box in res}
    with open(os.path.join(out, "prompt_boxes.json"), "w") as f:
        json.dump(boxes, f)
    ncomp = [b["n_components"] for b in boxes.values() if b]
    print(f"{cohort}: {len(boxes)} native images, {sum(b is None for b in boxes.values())} empty official masks, "
          f"official-mask components median {np.median(ncomp):.0f}")


def _final_one(args):
    cohort, pid, g = args
    perm, anat = anat_perm(g["iop"])
    native_spacing = [g["dz"], g["ps"], g["ps"]]           # (slice, row, column)
    zyx_spacing = [native_spacing[i] for i in perm]         # reordered (Z, Y, X)
    sitk_spacing = tuple(float(v) for v in zyx_spacing[::-1])  # sitk order (x, y, z)

    def write(arr, sub, dtype):
        a = np.ascontiguousarray(arr.transpose(perm)).astype(dtype)
        im = sitk.GetImageFromArray(a)
        im.SetSpacing(sitk_spacing)
        d = os.path.join(FIX, cohort, sub)
        os.makedirs(d, exist_ok=True)
        sitk.WriteImage(im, os.path.join(d, f"{pid}.nii.gz"))

    ph = f"{V3}/{cohort.lower()}_phases"
    write(to_native(read(f"{ISPY}/{cohort}_extracted/images/{pid}_image.nii.gz")), "early", np.float32)
    write(to_native(read(f"{ph}/{pid}_pre.nii.gz")), "pre", np.float32)
    write(to_native(read(f"{ph}/{pid}_late.nii.gz")), "late", np.float32)
    write(to_native(read(f"{ISPY}/{cohort}_extracted/masks/{pid}_mask.nii.gz") > 0), "mask_official", np.uint8)
    mp = os.path.join(FIX, cohort, "native_medsam2", f"{pid}_mask.nii.gz")
    has_m = os.path.exists(mp)
    if has_m:
        write(read(mp) > 0, "mask_medsam2", np.uint8)
    return {"pid": pid, "perm": str(perm), "native_anat": str(anat), "spacing_xyz": str(sitk_spacing),
            "src": g["src"], "medsam2": has_m}


def stage_finalize(cohort, n_proc):
    geo = geometry(cohort)
    with Pool(n_proc) as p:
        rows = p.map(_final_one, [(cohort, pid, g) for pid, g in geo.items()])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(FIX, f"{cohort.lower()}_fixed_geometry.csv"), index=False)
    print(df.perm.value_counts().to_dict(), df.src.value_counts().to_dict(), "medsam2:", int(df.medsam2.sum()), "/", len(df))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["native", "finalize"])
    ap.add_argument("--cohort", required=True, choices=["ISPY1", "ISPY2"])
    ap.add_argument("--n-proc", type=int, default=16)
    a = ap.parse_args()
    (stage_native if a.stage == "native" else stage_finalize)(a.cohort, a.n_proc)
