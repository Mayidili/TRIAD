"""
Download all DICOM images of a public TCIA collection through the TCIA REST API.
Files are stored as {out_dir}/{PatientID}/{SeriesInstanceUID}/; zip archives are extracted and discarded.

Usage:
  python scripts/download_tcia.py --collection "Duke-Breast-Cancer-MRI" \
      --out-dir data/Duke-Breast-Cancer-MRI --workers 6

Series already downloaded (checked against the zip size) are skipped, so the script can be rerun to resume.
Only requests is used for HTTP; curl may fail the TLS handshake with this server on some systems.
"""
import argparse
import io
import json
import os
import time
import zipfile

import requests

BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def fetch_series_list(collection, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    resp = requests.get(f"{BASE_URL}/getSeries", params={"Collection": collection, "format": "json"}, timeout=120)
    resp.raise_for_status()
    series_list = resp.json()
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(series_list, f)
    return series_list


def _already_done(series_dir, expected_file_size):
    marker = os.path.join(series_dir, ".done")
    if not os.path.exists(marker):
        return False
    if expected_file_size:
        try:
            with open(marker) as f:
                return int(f.read().strip()) == int(expected_file_size)
        except (ValueError, OSError):
            return False
    return True


def download_series(series_meta, out_dir, max_retries=3, timeout=600):
    uid = series_meta["SeriesInstanceUID"]
    patient_id = series_meta.get("PatientID", "unknown_patient")
    expected_size = series_meta.get("FileSize")
    series_dir = os.path.join(out_dir, patient_id, uid)

    if _already_done(series_dir, expected_size):
        return uid, "skipped"

    os.makedirs(series_dir, exist_ok=True)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                f"{BASE_URL}/getImage",
                params={"SeriesInstanceUID": uid},
                timeout=timeout,
                stream=True,
            )
            resp.raise_for_status()
            content = resp.content
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                zf.extractall(series_dir)
            with open(os.path.join(series_dir, ".done"), "w") as f:
                f.write(str(expected_size or len(content)))
            return uid, "ok"
        except Exception as e:  # noqa: BLE001 -- retry on any download/extract error, report only the last failure
            last_err = e
            time.sleep(2 * attempt)
    return uid, f"failed: {last_err}"


def run(collection, out_dir, workers, cache_path, log_path):
    series_list = fetch_series_list(collection, cache_path)
    print(f"{collection}: {len(series_list)} series to process")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {"ok": 0, "skipped": 0, "failed": 0}
    with open(log_path, "a") as log_f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_series, s, out_dir): s["SeriesInstanceUID"] for s in series_list}
        for i, future in enumerate(as_completed(futures), 1):
            uid, status = future.result()
            bucket = "failed" if status.startswith("failed") else status
            results[bucket] += 1
            log_f.write(f"{uid}\t{status}\n")
            log_f.flush()
            if i % 50 == 0 or i == len(series_list):
                print(f"[{i}/{len(series_list)}] ok={results['ok']} skipped={results['skipped']} failed={results['failed']}")

    print("done:", results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cache", default=None, help="JSON file caching the series list, so reruns skip the API query")
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    cache_path = args.cache or os.path.join(args.out_dir, "_series_list_cache.json")
    log_path = args.log or os.path.join(args.out_dir, "_download_log.tsv")
    os.makedirs(args.out_dir, exist_ok=True)

    run(args.collection, args.out_dir, args.workers, cache_path, log_path)
