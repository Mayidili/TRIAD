"""
Read the true voxel geometry of I-SPY cases from the original TCIA DICOM headers (the BreastDCEDL NIfTI files
all have an identity affine and pixdim=1).
For each case the original DCE series of the first examination (earliest StudyDate) is used; one image is
downloaded to read PixelSpacing / SpacingBetweenSlices / SliceThickness / Rows / Columns.
  ISPY1: series description ending in ": PE1" (a DCE-derived series with the same geometry as the original DCE;
         original series names differ between sites)
  ISPY2: used to check the xy_spacing / slice_thick values in the metadata (--cohort ISPY2 --limit N)
TCIA requires relaxed TLS settings (the server only accepts older cipher suites).
"""
import argparse
import io
import os
import ssl
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pydicom

from prism.paths import DATA_ROOT

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
CTX = ssl.create_default_context()
CTX.set_ciphers("DEFAULT@SECLEVEL=1")
CTX.maximum_version = ssl.TLSVersion.TLSv1_2


def get(url, raw=False):
    for _ in range(4):
        try:
            with urllib.request.urlopen(url, context=CTX, timeout=120) as r:
                b = r.read()
            return b if raw else json.loads(b)
        except Exception as e:  # noqa: BLE001
            err = e
    raise err


def one(args):
    collection, pid, desc_match = args
    try:
        series = get(f"{API}/getSeries?Collection={collection}&PatientID={pid}")
        dce = [s for s in series if desc_match(s.get("SeriesDescription", ""))]
        if not dce:
            return {"pid": pid, "status": "no_dce", "descs": "|".join(sorted({s.get('SeriesDescription', '') for s in series}))}
        dce.sort(key=lambda s: (s["StudyDate"], s["SeriesNumber"]))
        s = dce[0]
        sop = get(f"{API}/getSOPInstanceUIDs?SeriesInstanceUID={s['SeriesInstanceUID']}")[0]["SOPInstanceUID"]
        d = pydicom.dcmread(io.BytesIO(get(f"{API}/getSingleImage?SeriesInstanceUID={s['SeriesInstanceUID']}&SOPInstanceUID={sop}", raw=True)),
                            stop_before_pixels=True)
        return {"pid": pid, "status": "ok", "series": s["SeriesDescription"], "study_date": s["StudyDate"][:10],
                "image_count": s["ImageCount"], "rows": int(d.Rows), "cols": int(d.Columns),
                "ps_row": float(d.PixelSpacing[0]), "ps_col": float(d.PixelSpacing[1]),
                "thk": float(getattr(d, "SliceThickness", "nan")), "sbs": float(getattr(d, "SpacingBetweenSlices", "nan")),
                "iop": str([round(float(x), 2) for x in d.ImageOrientationPatient]),
                "field": float(getattr(d, "MagneticFieldStrength", "nan")), "manufacturer": str(getattr(d, "Manufacturer", ""))}
    except Exception as e:  # noqa: BLE001
        return {"pid": pid, "status": f"error {type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["ISPY1", "ISPY2"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    img_dir = f"{DATA_ROOT}/ISPY/{a.cohort}_extracted/images"
    pids = sorted(f[: -len("_image.nii.gz")] for f in os.listdir(img_dir))[: a.limit]
    if a.cohort == "ISPY1":
        # series names differ between sites; every case has a "...: PE1" or "...: PE2" series derived from the original DCE (same geometry, single phase)
        tasks = [("ISPY1", p, lambda d: d.endswith(": PE1") or d.endswith(": PE2")) for p in pids]
    else:
        coll = lambda p: "ACRIN-6698" if p.startswith("ACRIN") else "ISPY2"
        tasks = [(coll(p), p, lambda d: ("dce" in d.lower() or "dyn" in d.lower() or "vibrant" in d.lower())
                  and not any(k in d.lower() for k in ["ser", "pe1", "pe2", "mip", "sub", "seg", "analysis"])) for p in pids]
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one, tasks))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.status.str.split().str[0].value_counts())
    ok = df[df.status == "ok"]
    if len(ok):
        print(ok[["rows", "cols", "ps_row", "ps_col", "thk", "sbs"]].describe().round(3))
        print("thk != sbs:", (abs(ok.thk - ok.sbs) > 1e-3).sum(), " iop:", ok.iop.value_counts().head(3).to_dict())


if __name__ == "__main__":
    main()
