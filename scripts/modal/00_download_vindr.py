"""00 — Download VinDr-CXR from Kaggle into a Modal Volume.

Replaces the gaze-v0.1 era REFLACX/MIMIC downloader. The Kaggle release
of VinDr-CXR (the "VinBigData Chest X-ray Abnormalities Detection"
competition) does NOT require PhysioNet credentialing or CITI training
— only a Kaggle account and acceptance of the competition rules.

Run prerequisites (one-time, do these on the Kaggle web UI):
  1. Kaggle account at https://www.kaggle.com (free).
  2. Accept the competition rules at
     https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/rules
     (one click on "I Understand and Accept").
  3. Generate an API token at https://www.kaggle.com/settings/account
     → click "Create New API Token". As of 2026, this produces a
     single opaque token of the form `KGAT_<32 hex chars>`.
  4. Create a Modal Secret named `kaggle-token` from the dashboard.
     There are two supported schemes; this script auto-detects which
     one is in use:
       (a) New scheme (recommended, current Kaggle UI):
             KAGGLE_API_TOKEN = KGAT_<32 hex chars>
       (b) Legacy username/key scheme (older accounts):
             KAGGLE_USERNAME = <username>
             KAGGLE_KEY      = <key>
     If both are set, KAGGLE_API_TOKEN takes priority.

Run:
    modal run scripts/modal/00_download_vindr.py

Expected runtime: 5-15 minutes for the ~6 GB competition zip + extract
+ optional DICOM→PNG conversion. The Modal Volume `gazeprobe-data`
persists the result so this script never needs to re-run.

Layout produced on the volume:
    /data/vindr/
        train.csv                 # bbox annotations (per-rater rows)
        train/<image_id>.dicom    # original DICOM frontal CXRs
        train_png/<image_id>.png  # 1024px PNGs (if --convert-png)
        sample_submission.csv
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

# ----------------------------------------------------------------------- #
# Modal app + image
# ----------------------------------------------------------------------- #

app = modal.App("gazeprobe-download-vindr")

# Image: just enough to download from Kaggle + parse + (optionally) DICOM
# convert. Generation/inference workloads use a different, heavier image
# defined in scripts/modal/_image.py (forthcoming).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("unzip")
    .pip_install(
        # Unpinned so we get a recent enough version to handle the
        # 2026 KAGGLE_API_TOKEN single-token scheme. 1.6.x was too old.
        "kaggle",
        "pandas==2.2.3",
        "pydicom==2.4.4",
        "pillow==10.4.0",
        "numpy==2.1.2",
        "tqdm==4.66.5",
    )
)

data_vol = modal.Volume.from_name("gazeprobe-data", create_if_missing=True)


# ----------------------------------------------------------------------- #
# Download function
# ----------------------------------------------------------------------- #

@app.function(
    image=image,
    volumes={"/data": data_vol},
    secrets=[modal.Secret.from_name("kaggle-token")],
    # Bumped from 1h after the previous run timed out partway through
    # the 15k-file DICOM→PNG conversion. 6h is plenty even worst-case.
    timeout=6 * 60 * 60,
    # The DICOM conversion is now parallelized; ask for real cores
    # so we actually get parallelism. Modal default is 0.125 CPU.
    cpu=8.0,
    # NOTE: deliberately NOT setting ephemeral_disk. Modal's minimum
    # ephemeral request is 512 GiB which we don't need — all the
    # actual data (zip, DICOMs, PNGs) lives on the `gazeprobe-data`
    # Volume mounted at /data, which has its own (larger) quota.
    # Only /tmp scratch uses ephemeral; default is enough.
)
def download_all(
    competition: str = "vinbigdata-chest-xray-abnormalities-detection",
    convert_png: bool = True,
    png_resolution: int = 1024,
    keep_test_split: bool = False,
) -> dict:
    """Download VinDr-CXR competition data into /data/vindr and
    optionally convert DICOMs to PNGs at the given resolution.

    Args:
        competition: Kaggle competition slug.
        convert_png: If True, convert each DICOM to a PNG at
            `png_resolution`. PNGs land under /data/vindr/train_png/.
            Recommended True — DICOM loading at inference time is
            slow and pydicom adds an unnecessary dependency.
        png_resolution: Edge length of the longest side after
            conversion. 1024 is plenty for our VLMs (which downsample
            to 336-896 internally).
        keep_test_split: VinDr-CXR competition ships a `test/` split
            without annotations. We don't need it (our audit uses only
            the annotated `train/` split). Default False = delete to
            save volume space.
    """
    target = Path("/data/vindr")
    target.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Phase 1 — Kaggle auth sanity probe
    # ------------------------------------------------------------------ #
    print("=" * 64)
    print("Phase 1/4 — Kaggle auth probe")
    print("=" * 64)
    api_token = os.environ.get("KAGGLE_API_TOKEN")
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True, parents=True)

    if api_token:
        # New 2026 single-token scheme.
        print(f"  scheme:                 KAGGLE_API_TOKEN (single-token, 2026 format)")
        print(f"  KAGGLE_API_TOKEN length: {len(api_token)}  (expected ~37, format KGAT_<32hex>)")
        print(f"  KAGGLE_API_TOKEN ends with whitespace: {api_token != api_token.rstrip()}")
        print(f"  KAGGLE_API_TOKEN starts with 'KGAT_':   {api_token.startswith('KGAT_')}")

        # Write the access_token file that the new kaggle CLI reads.
        access_token = kaggle_dir / "access_token"
        access_token.write_text(api_token.strip())
        os.chmod(access_token, 0o600)
        print(f"  wrote {access_token} (mode 600)")

        # Some kaggle CLI versions look at the env var directly; make
        # sure the (possibly-trimmed) value is exported so the
        # subprocess inherits it cleanly.
        os.environ["KAGGLE_API_TOKEN"] = api_token.strip()

    elif user and key:
        # Legacy username/key scheme.
        print(f"  scheme:                 KAGGLE_USERNAME + KAGGLE_KEY (legacy format)")
        print(f"  KAGGLE_USERNAME:        {user}")
        print(f"  KAGGLE_USERNAME length: {len(user)}")
        print(f"  KAGGLE_USERNAME ends with whitespace: {user != user.rstrip()}")
        print(f"  KAGGLE_KEY length:      {len(key)}  (expected ~32-40)")
        print(f"  KAGGLE_KEY ends with whitespace:      {key != key.rstrip()}")

        kaggle_json = kaggle_dir / "kaggle.json"
        import json as _json
        kaggle_json.write_text(_json.dumps({"username": user.strip(), "key": key.strip()}))
        os.chmod(kaggle_json, 0o600)
        print(f"  wrote {kaggle_json} (mode 600)")

    else:
        raise RuntimeError(
            "Modal Secret `kaggle-token` is missing both schemes:\n"
            "  EITHER KAGGLE_API_TOKEN (the 2026 single-token format,\n"
            "          e.g. KGAT_<32 hex chars>),\n"
            "  OR     KAGGLE_USERNAME + KAGGLE_KEY (legacy).\n"
            "Edit the Modal Secret to add KAGGLE_API_TOKEN with the\n"
            "value from Kaggle Account → Create New API Token."
        )

    # `kaggle competitions list` is a cheap auth probe.
    probe = subprocess.run(
        ["kaggle", "competitions", "list", "--page", "1"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        # Redact every credential value from any output before showing
        # it — covers both the new single-token and the legacy schemes.
        def _redact(s: str) -> str:
            needles = []
            if api_token:
                needles.extend([api_token, api_token.strip()])
            if key:
                needles.extend([key, key.strip()])
            if user:
                needles.extend([user, user.strip()])
            for needle in needles:
                if needle:
                    s = s.replace(needle, "<REDACTED>")
            return s
        stderr_red = _redact(probe.stderr or "")
        stdout_red = _redact(probe.stdout or "")
        print()
        print("---- stderr (redacted) ----")
        print(stderr_red[-2000:])
        print("---- stdout (redacted) ----")
        print(stdout_red[-2000:])
        raise RuntimeError(
            f"Kaggle auth probe failed (exit {probe.returncode}). "
            "See the redacted stderr above for the actual cause. "
            "Common: (a) wrong username/key in Modal Secret "
            "(check the lengths printed above against the values in "
            "kaggle.json), (b) Kaggle account not verified by phone, "
            "(c) corporate firewall blocking kaggle.com."
        )
    print("  ✓ auth OK")

    # ------------------------------------------------------------------ #
    # Phase 2 — Competition acceptance probe
    # ------------------------------------------------------------------ #
    print("=" * 64)
    print(f"Phase 2/4 — competition acceptance probe ({competition})")
    print("=" * 64)
    # Cheapest possible test: try to list the files. If the user hasn't
    # accepted rules, Kaggle returns "You must accept this competition's
    # rules before you'll be able to download files."
    list_probe = subprocess.run(
        ["kaggle", "competitions", "files", "-c", competition],
        capture_output=True, text=True,
    )
    if list_probe.returncode != 0 or "accept" in (list_probe.stdout + list_probe.stderr).lower():
        raise RuntimeError(
            f"Competition `{competition}` rules not accepted. Visit "
            f"https://www.kaggle.com/competitions/{competition}/rules "
            "in your browser, click 'I Understand and Accept', then re-run."
        )
    print("  ✓ competition accepted")

    # ------------------------------------------------------------------ #
    # Phase 3 — Download + unzip
    # ------------------------------------------------------------------ #
    print("=" * 64)
    print(f"Phase 3/4 — download + unzip ({competition})")
    print("=" * 64)
    if not (target / "train").exists() or not list(target.glob("train/*.dicom")):
        zip_path = target / f"{competition}.zip"
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", competition,
             "-p", str(target)],
            check=True,
        )
        print(f"  downloaded → {zip_path}")
        subprocess.run(
            ["unzip", "-o", "-q", str(zip_path), "-d", str(target)],
            check=True,
        )
        zip_path.unlink()
        print(f"  unzipped → {target}")
    else:
        print(f"  ✓ {target}/train already populated; skipping download")

    if not keep_test_split and (target / "test").exists():
        # Free volume space — we never use the unannotated test split
        import shutil
        shutil.rmtree(target / "test")
        print(f"  removed test/ to save volume space")
    if not keep_test_split and (target / "sample_submission.csv").exists():
        (target / "sample_submission.csv").unlink()

    # ------------------------------------------------------------------ #
    # Phase 4 — Optional DICOM → PNG conversion (parallel)
    # ------------------------------------------------------------------ #
    print("=" * 64)
    print("Phase 4/4 — DICOM → PNG conversion" if convert_png else "Phase 4/4 — skipped")
    print("=" * 64)
    converted = skipped = failed = 0
    if convert_png:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        png_root = target / "train_png"
        png_root.mkdir(parents=True, exist_ok=True)

        dicoms = sorted((target / "train").glob("*.dicom"))
        # Pre-filter so the worker pool only sees real work
        todo = [d for d in dicoms if not (png_root / f"{d.stem}.png").exists()]
        skipped = len(dicoms) - len(todo)
        print(f"  total DICOMs:    {len(dicoms)}")
        print(f"  already PNG:     {skipped}")
        print(f"  to convert:      {len(todo)}")

        # Heuristic: ThreadPoolExecutor with ~2x physical cores. The
        # bottleneck is pydicom file I/O + numpy normalization +
        # PIL encode, all of which release the GIL meaningfully.
        n_workers = 16
        commit_every = 1000

        def _convert(dcm_path):
            import numpy as np
            import pydicom
            from PIL import Image
            png_path = png_root / f"{dcm_path.stem}.png"
            try:
                arr = pydicom.dcmread(dcm_path).pixel_array
                arr = arr.astype(np.float32)
                arr -= arr.min()
                if arr.max() > 0:
                    arr = arr / arr.max() * 255
                arr = arr.astype(np.uint8)
                img = Image.fromarray(arr, mode="L")
                w, h = img.size
                if max(w, h) > png_resolution:
                    if w >= h:
                        new_w, new_h = png_resolution, int(h * png_resolution / w)
                    else:
                        new_h, new_w = png_resolution, int(w * png_resolution / h)
                    img = img.resize((new_w, new_h), Image.BILINEAR)
                img.save(png_path, format="PNG")
                return True, None
            except Exception as e:
                return False, f"{dcm_path.name}: {e}"

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_convert, d): d for d in todo}
            for i, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc="convert"), 1):
                ok, err = fut.result()
                if ok:
                    converted += 1
                else:
                    failed += 1
                    print(f"  [WARN] {err}", file=sys.stderr)
                # Periodic commit so a late crash doesn't lose work
                if commit_every and i % commit_every == 0:
                    data_vol.commit()
                    print(f"  intermediate commit at {i}/{len(futures)}")

    # Final commit
    data_vol.commit()

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    train_csv = target / "train.csv"
    train_dicom_count = len(list((target / "train").glob("*.dicom")))
    train_png_count = len(list((target / "train_png").glob("*.png"))) if convert_png else 0
    print()
    print("=" * 64)
    print("Summary")
    print("=" * 64)
    print(f"  train.csv exists: {train_csv.exists()}")
    print(f"  DICOM count:      {train_dicom_count}")
    if convert_png:
        print(f"  PNG count:        {train_png_count}")
        print(f"  Newly converted:  {converted}")
        print(f"  Already present:  {skipped}")
        print(f"  Failed:           {failed}")
    print(f"  Volume committed: gazeprobe-data")
    print("=" * 64)
    return {
        "train_csv_exists": train_csv.exists(),
        "dicom_count": train_dicom_count,
        "png_count": train_png_count,
        "converted": converted,
        "skipped": skipped,
        "failed": failed,
    }


# ----------------------------------------------------------------------- #
# Local entrypoint
# ----------------------------------------------------------------------- #

@app.local_entrypoint()
def main(
    competition: str = "vinbigdata-chest-xray-abnormalities-detection",
    convert_png: bool = True,
    png_resolution: int = 1024,
    keep_test_split: bool = False,
):
    result = download_all.remote(
        competition=competition,
        convert_png=convert_png,
        png_resolution=png_resolution,
        keep_test_split=keep_test_split,
    )
    print()
    print("Returned to local:")
    for k, v in result.items():
        print(f"  {k}: {v}")
