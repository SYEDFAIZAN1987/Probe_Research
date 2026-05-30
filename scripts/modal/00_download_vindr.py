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
     → click "Create New Token" → downloads `kaggle.json`.
  4. Create a Modal Secret named `kaggle-token` from the dashboard
     with two key-value pairs:
        KAGGLE_USERNAME = <username from kaggle.json>
        KAGGLE_KEY      = <key from kaggle.json>
     The Kaggle CLI reads these as environment variables.

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
        "kaggle==1.6.17",
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
    timeout=60 * 60,  # 1 hour cap; should finish in 5-15 min
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
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not user or not key:
        raise RuntimeError(
            "Modal Secret `kaggle-token` is missing KAGGLE_USERNAME or "
            "KAGGLE_KEY. Create the Secret from the Modal dashboard "
            "(Secrets → New → name=kaggle-token, fields=KAGGLE_USERNAME, "
            "KAGGLE_KEY) and re-run."
        )
    print(f"  Kaggle user: {user}")
    print("  KAGGLE_KEY:  <set, length="
          f"{len(key)}>")

    # `kaggle competitions list` is a cheap auth probe.
    probe = subprocess.run(
        ["kaggle", "competitions", "list", "--page", "1"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        # Don't print stderr — it may include the API key on auth failures
        raise RuntimeError(
            f"Kaggle auth probe failed (exit {probe.returncode}). "
            "Common causes: (a) wrong username/key in the Modal Secret, "
            "(b) Kaggle account not verified by phone, (c) corporate "
            "firewall blocking kaggle.com."
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
    # Phase 4 — Optional DICOM → PNG conversion
    # ------------------------------------------------------------------ #
    print("=" * 64)
    print("Phase 4/4 — DICOM → PNG conversion" if convert_png else "Phase 4/4 — skipped")
    print("=" * 64)
    converted = skipped = failed = 0
    if convert_png:
        import numpy as np
        import pydicom
        from PIL import Image
        from tqdm import tqdm

        png_root = target / "train_png"
        png_root.mkdir(parents=True, exist_ok=True)

        dicoms = sorted((target / "train").glob("*.dicom"))
        for dcm_path in tqdm(dicoms, desc="convert", unit="img"):
            png_path = png_root / f"{dcm_path.stem}.png"
            if png_path.exists():
                skipped += 1
                continue
            try:
                arr = pydicom.dcmread(dcm_path).pixel_array
                # Normalize to 0-255 uint8
                arr = arr.astype(np.float32)
                arr -= arr.min()
                if arr.max() > 0:
                    arr = arr / arr.max() * 255
                arr = arr.astype(np.uint8)
                img = Image.fromarray(arr, mode="L")
                # Resize so longest side = png_resolution, preserving aspect
                w, h = img.size
                if max(w, h) > png_resolution:
                    if w >= h:
                        new_w = png_resolution
                        new_h = int(h * png_resolution / w)
                    else:
                        new_h = png_resolution
                        new_w = int(w * png_resolution / h)
                    img = img.resize((new_w, new_h), Image.BILINEAR)
                img.save(png_path, format="PNG")
                converted += 1
            except Exception as e:
                print(f"  [WARN] {dcm_path.name}: {e}", file=sys.stderr)
                failed += 1

    # Commit volume so the downloaded data survives function teardown
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
