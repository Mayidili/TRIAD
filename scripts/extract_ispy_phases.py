"""
Extract the pre-contrast (S0) and late post-contrast (SL) phases of I-SPY1/I-SPY2 from the BreastDCEDL archives;
the early phase (S1) is images/{pid}_image.nii.gz from extract_ispy{1,2}_images.py.

Phases:
  I-SPY1  S0 = acq0, S1 = acq1, SL = acq2
          (the I-SPY1 metadata has no phase-index column; almost all patients have acq0-2 only, taken as pre/early/late)
  I-SPY2  S0/S1/SL = the pre / post_early / post_late indices in the metadata
Output: <interim>/{ispy1,ispy2}_phases/<pid>_{pre,late}.nii.gz

Usage:
  python3 extract_ispy_phases.py --cohort ISPY1 --n-proc 16
  python3 extract_ispy_phases.py --cohort ISPY2 --n-proc 14   # one process per tar archive
"""
import argparse
import os
import tarfile
import zipfile
from multiprocessing import Pool

import pandas as pd

from prism.paths import DATA_ROOT, INTERIM_DIR

ISPY = os.path.join(DATA_ROOT, "ISPY")
OUT_DIR = INTERIM_DIR


def extract_ispy1(phase_dir):
    z = zipfile.ZipFile(os.path.join(ISPY, "ISPY1/BreastDCEDL_spy1.zip"))
    names = set(z.namelist())
    pids = [f[: -len("_image.nii.gz")] for f in os.listdir(os.path.join(ISPY, "ISPY1_extracted/images"))]
    for pid in pids:
        for tag, acq in (("pre", 0), ("late", 2)):
            src = f"BreastDCEDL_spy1/spt1_dce/{pid}_spy1_vis1_acq{acq}.nii.gz"
            dst = os.path.join(phase_dir, f"{pid}_{tag}.nii.gz")
            if src in names and not os.path.exists(dst):
                with open(dst, "wb") as f:
                    f.write(z.read(src))
    return pids


def _scan_tar(args):
    path, want, phase_dir = args
    n = 0
    with tarfile.open(path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            base = os.path.basename(member.name)
            if base in want:
                dst = os.path.join(phase_dir, want[base])
                with tf.extractfile(member) as src, open(dst, "wb") as f:
                    f.write(src.read())
                n += 1
    return os.path.basename(path), n


def extract_ispy2(phase_dir, n_proc):
    meta = pd.read_csv(os.path.join(ISPY, "ISPY2/BreastDCEDL_ISPY2_metadata.csv"))
    have = {f[: -len("_image.nii.gz")] for f in os.listdir(os.path.join(ISPY, "ISPY2_extracted/images"))}
    meta = meta[meta.pid.isin(have)]
    want = {}
    for r in meta.itertuples():
        for tag, i in (("pre", r.pre), ("late", r.post_late)):
            if pd.notna(i) and not os.path.exists(os.path.join(phase_dir, f"{r.pid}_{tag}.nii.gz")):
                want[f"{r.pid}_spy2_vis1_dce_aqc_{int(i)}.nii.gz"] = f"{r.pid}_{tag}.nii.gz"
    parts = [f"BreastDCEDL_ISPY2_nopcr_dce_part{i}.tar.gz" for i in range(1, 9)] + \
            [f"BreastDCEDL_ISPY2_pcr_dce_part{i}.tar.gz" for i in range(1, 7)]
    if want:
        tasks = [(os.path.join(ISPY, "ISPY2", p), want, phase_dir) for p in parts]
        with Pool(min(n_proc, len(tasks))) as pool:
            for name, n in pool.imap_unordered(_scan_tar, tasks):
                print(f"{name}: extracted {n}", flush=True)
    return sorted(have)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["ISPY1", "ISPY2"])
    ap.add_argument("--n-proc", type=int, default=8)
    args = ap.parse_args()
    phase_dir = os.path.join(OUT_DIR, f"{args.cohort.lower()}_phases")
    os.makedirs(phase_dir, exist_ok=True)
    pids = extract_ispy1(phase_dir) if args.cohort == "ISPY1" else extract_ispy2(phase_dir, args.n_proc)


if __name__ == "__main__":
    main()
