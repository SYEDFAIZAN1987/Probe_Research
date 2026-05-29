"""Cloud-side download of REFLACX + MIMIC-CXR-JPG subset into the
`gazeprobe-data` Modal Volume.

Runs as a Modal CPU function (no GPU) — much faster than uploading
from a home connection because Modal containers have gigabit+
bandwidth to PhysioNet.

Phases (all idempotent, safe to re-run):
  1. wget REFLACX 1.0.0 (~2 GB) into /data/reflacx/
  2. Parse REFLACX metadata, build the set of unique
     (subject_id, study_id, dicom_id) tuples it references
  3. wget the matching subset of MIMIC-CXR-JPG 2.0.0 (~2.6 GB)
     into /data/mimic-cxr-jpg/

Requires the Modal Secret `physionet-credentials` with keys
PHYSIONET_USER and PHYSIONET_PASS.

Run:
    modal run scripts/modal/00_download_data.py

Verify after:
    modal volume ls gazeprobe-data /
    modal volume ls gazeprobe-data /reflacx
    modal volume ls gazeprobe-data /mimic-cxr-jpg
"""

import os
import subprocess
from pathlib import Path

import modal


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget")
    .pip_install("pandas==2.2.3", "tqdm==4.66.5")
)

app = modal.App("gazeprobe-data-ingest", image=image)
data_vol = modal.Volume.from_name("gazeprobe-data")


@app.function(
    volumes={"/data": data_vol},
    secrets=[modal.Secret.from_name("physionet-credentials")],
    timeout=60 * 60,                   # 1 hour hard cap
    cpu=2,
    memory=4096,
)
def download_all() -> dict:
    import pandas as pd
    from tqdm import tqdm

    user = os.environ["PHYSIONET_USER"]
    pwd = os.environ["PHYSIONET_PASS"]
    auth = ["--user", user, "--password", pwd]

    reflacx_root = Path("/data/reflacx")
    mimic_root = Path("/data/mimic-cxr-jpg")
    reflacx_root.mkdir(parents=True, exist_ok=True)
    mimic_root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- #
    # Phase 0 — auth probe (fail fast on bad creds / missing DUA)
    # ---------------------------------------------------------------- #
    # Wget exit code 8 with --quiet hides the actual HTTP status. We
    # do a single tiny fetch (--server-response so the headers print)
    # of the dataset's LICENSE.txt before committing to a recursive
    # pull. If THIS errors, we surface the real 4xx/5xx code so you
    # can tell credentials-vs-DUA-vs-404 apart.
    print("=" * 64)
    print("Phase 0/3 — auth probe")
    print("=" * 64)
    probe_url = "https://physionet.org/files/reflacx-xray-localization/1.0.0/LICENSE.txt"
    probe_cmd = [
        "wget", "--server-response", "--tries=1", "--timeout=30",
        "-O", "/tmp/_probe.txt",
        *auth, probe_url,
    ]
    # Don't echo `pwd` if this prints; redact for the log.
    redacted = [c if c != pwd else "<REDACTED>" for c in probe_cmd]
    print(f"  probe: {' '.join(redacted)}")
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    # wget writes server response headers to stderr
    head = probe.stderr.splitlines()[-30:] if probe.stderr else []
    for line in head:
        print(f"    {line}")
    if probe.returncode != 0:
        raise RuntimeError(
            "Auth probe failed. Most likely cause: REFLACX data-use "
            "agreement not accepted on your PhysioNet account. Visit "
            "https://physionet.org/content/reflacx-xray-localization/1.0.0/ "
            "and accept the DUA at the bottom of the page. Same for "
            "https://physionet.org/content/mimic-cxr-jpg/2.0.0/ before "
            "phase 3."
        )
    print("  ✓ auth OK")

    # ---------------------------------------------------------------- #
    # Phase 1 — REFLACX recursive fetch
    # ---------------------------------------------------------------- #
    print("=" * 64)
    print("Phase 1/3 — REFLACX 1.0.0")
    print("=" * 64)
    cmd = [
        "wget",
        "--recursive", "--no-parent",
        "--timestamping", "--continue",
        "--no-host-directories", "--cut-dirs=3",
        "--reject", "index.html*,robots.txt",
        "-P", str(reflacx_root),
        *auth,
        "https://physionet.org/files/reflacx-xray-localization/1.0.0/",
    ]
    # No --quiet so server errors are visible. Output is moderate.
    subprocess.run(cmd, check=True)
    data_vol.commit()
    print("  ✓ REFLACX download complete")

    # ---------------------------------------------------------------- #
    # Phase 2 — parse metadata to build the image subset list
    # ---------------------------------------------------------------- #
    print("=" * 64)
    print("Phase 2/3 — building MIMIC-CXR-JPG URL list")
    print("=" * 64)
    frames = []
    for phase in (1, 2, 3):
        p = reflacx_root / f"metadata_phase_{phase}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
        else:
            print(f"  [WARN] {p} not found")
    if not frames:
        raise RuntimeError("no metadata_phase_*.csv found — REFLACX layout changed?")

    meta = pd.concat(frames, ignore_index=True)
    required = {"dicom_id", "subject_id", "study_id"}
    missing = required - set(meta.columns)
    if missing:
        raise RuntimeError(
            f"REFLACX metadata missing required columns {missing}; "
            f"present: {list(meta.columns)}"
        )
    unique = (
        meta[["dicom_id", "subject_id", "study_id"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"  {len(unique)} unique (subject, study, dicom) tuples to fetch")

    # ---------------------------------------------------------------- #
    # Phase 3 — fetch each JPG, skip existing
    # ---------------------------------------------------------------- #
    print("=" * 64)
    print("Phase 3/3 — MIMIC-CXR-JPG subset")
    print("=" * 64)

    # Phase-3 auth probe: separate DUA from REFLACX's. Fail fast.
    mimic_probe_url = "https://physionet.org/files/mimic-cxr-jpg/2.0.0/LICENSE.txt"
    probe = subprocess.run(
        ["wget", "--server-response", "--tries=1", "--timeout=30",
         "-O", "/tmp/_mimic_probe.txt", *auth, mimic_probe_url],
        capture_output=True, text=True,
    )
    for line in (probe.stderr.splitlines()[-10:] if probe.stderr else []):
        print(f"    {line}")
    if probe.returncode != 0:
        raise RuntimeError(
            "MIMIC-CXR-JPG auth probe failed. The REFLACX DUA does NOT "
            "cover MIMIC-CXR-JPG — they're separate datasets. Visit "
            "https://physionet.org/content/mimic-cxr-jpg/2.0.0/ and "
            "accept the data-use agreement, then re-run."
        )
    print("  ✓ MIMIC-CXR-JPG auth OK")

    base = "https://physionet.org/files/mimic-cxr-jpg/2.0.0/files"
    downloaded = skipped = failed = 0

    for i, row in tqdm(unique.iterrows(), total=len(unique), mininterval=2.0):
        sid = str(int(row["subject_id"]))
        stud = str(int(row["study_id"]))
        dcm = str(row["dicom_id"])
        rel_dir = f"p{sid[:2]}/p{sid}/s{stud}"
        out_dir = mimic_root / rel_dir
        out_path = out_dir / f"{dcm}.jpg"
        if out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        url = f"{base}/{rel_dir}/{dcm}.jpg"
        try:
            subprocess.run(
                ["wget", "--quiet", "--tries=3", "--timeout=30",
                 "-O", str(out_path), *auth, url],
                check=True, timeout=90,
            )
            downloaded += 1
        except Exception as e:
            failed += 1
            if out_path.exists() and out_path.stat().st_size == 0:
                out_path.unlink()
            print(f"  [WARN] {dcm}: {type(e).__name__}: {e}")

        # Commit the volume every 200 items so a container restart
        # doesn't lose progress.
        if (downloaded + skipped) % 200 == 0:
            data_vol.commit()

    data_vol.commit()
    summary = {"downloaded": downloaded, "skipped": skipped, "failed": failed,
               "n_unique": len(unique)}
    print()
    print(f"=== Done: {summary} ===")
    return summary


@app.local_entrypoint()
def main():
    result = download_all.remote()
    print()
    print("Final result:", result)
