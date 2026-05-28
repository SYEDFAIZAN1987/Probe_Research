"""01 - REFLACX schema exploration (day-1, read-only).

Goal: verify the local REFLACX dump matches the schema this project
assumes, so downstream extraction code does not break on a column
rename or a path quirk. The script does NOT modify data. It loads,
prints, and saves a small set of sanity plots to data/reports/.

Expected REFLACX layout (Lanfredi et al., 2022, PhysioNet v1.0.0):

    REFLACX_ROOT/
        metadata_phase_1.csv
        metadata_phase_2.csv
        metadata_phase_3.csv
        main_data/
            <id>/                           # one dir per trial
                anomaly_location_ellipses.csv   # per-finding ellipses (gold bboxes)
                chest_bounding_box.csv          # bbox of the chest in the image
                fixations.csv                   # (timestamp, x, y, duration)
                gaze.csv                        # raw gaze samples
                timestamps_transcription.csv    # word-level timestamps
                transcription.txt               # dictated report text
                tobii_calibration_log.csv

    MIMIC-CXR-JPG/
        files/p{XX}/p{patient}/s{study}/{dicom_id}.jpg

Linking REFLACX trials → MIMIC-CXR-JPG image files happens via the
`dicom_id` column in the REFLACX metadata.

Run:
    python scripts/01_reflacx_explore.py \\
        --reflacx-root /path/to/reflacx \\
        --mimic-jpg-root /path/to/mimic-cxr-jpg \\
        --out-dir data/reports

Outputs (committed to data/reports/, gitignored):
    schema_summary.txt              line-by-line printout of every CSV's
                                    columns + dtypes + head(3)
    fig_fixation_density.png        one example fixation density overlay
    fig_bbox_example.png            one example with gold ellipses drawn
    case_counts_by_phase.csv        N trials per (phase, reader)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reflacx-root", required=True, type=Path)
    ap.add_argument("--mimic-jpg-root", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("data/reports"), type=Path)
    ap.add_argument(
        "--n-trial-samples",
        default=3,
        type=int,
        help="Number of trial dirs to print full schema for. Default 3.",
    )
    return ap.parse_args()


# --------------------------------------------------------------------- #
# Schema printout helpers
# --------------------------------------------------------------------- #

def describe_csv(path: Path, n_head: int = 3) -> str:
    """Return a multi-line schema summary for a CSV file."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return f"[ERROR reading {path}]: {e}"
    parts = [
        f"=== {path} ===",
        f"  shape: {df.shape}",
        f"  columns: {list(df.columns)}",
        f"  dtypes: {dict(df.dtypes.astype(str))}",
        f"  head({n_head}):",
        df.head(n_head).to_string(index=False),
        "",
    ]
    return "\n".join(parts)


def describe_txt(path: Path, n_chars: int = 400) -> str:
    """Return a preview of a text file."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return f"[ERROR reading {path}]: {e}"
    preview = text[:n_chars] + ("..." if len(text) > n_chars else "")
    return f"=== {path} ===\n  length_chars: {len(text)}\n  preview: {preview!r}\n"


# --------------------------------------------------------------------- #
# Image linkage
# --------------------------------------------------------------------- #

def resolve_mimic_jpg_path(mimic_root: Path, dicom_id: str, subject_id: str | int, study_id: str | int) -> Path | None:
    """Map a REFLACX (dicom_id, subject_id, study_id) → MIMIC-CXR-JPG path.

    MIMIC-CXR-JPG groups patients by the first two chars of subject_id:
        files/p10/p10000032/s50414267/<dicom_id>.jpg
    """
    sid = str(subject_id)
    bucket = f"p{sid[:2]}"
    return mimic_root / "files" / bucket / f"p{sid}" / f"s{study_id}" / f"{dicom_id}.jpg"


# --------------------------------------------------------------------- #
# Sanity plots
# --------------------------------------------------------------------- #

def plot_fixations_on_image(
    image_path: Path,
    fixations: pd.DataFrame,
    out_path: Path,
) -> None:
    """Overlay fixation points on the CXR image. Read-only diagnostic."""
    import matplotlib.pyplot as plt  # local import — keeps cold-start fast

    img = np.array(Image.open(image_path).convert("L"))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap="gray")
    # NOTE: REFLACX fixation columns are typically `x_position`, `y_position`.
    # Confirm this on first run; the script will raise KeyError loudly if not.
    x_col = next(c for c in fixations.columns if c.lower().startswith("x"))
    y_col = next(c for c in fixations.columns if c.lower().startswith("y"))
    ax.scatter(
        fixations[x_col], fixations[y_col],
        s=15, c="red", alpha=0.5, edgecolors="none",
    )
    ax.set_title(f"Fixations on {image_path.name}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_bboxes_on_image(
    image_path: Path,
    ellipses: pd.DataFrame,
    out_path: Path,
) -> None:
    """Overlay gold anomaly ellipses on the CXR. Read-only diagnostic."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    img = np.array(Image.open(image_path).convert("L"))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap="gray")
    # Ellipse schema typically includes: xmin, ymin, xmax, ymax, and pathology label cols.
    # Use the bbox interpretation to draw a containing rectangle as a fallback.
    for _, row in ellipses.iterrows():
        if {"xmin", "ymin", "xmax", "ymax"}.issubset(row.index):
            cx = (row["xmin"] + row["xmax"]) / 2
            cy = (row["ymin"] + row["ymax"]) / 2
            w = row["xmax"] - row["xmin"]
            h = row["ymax"] - row["ymin"]
            ax.add_patch(Ellipse((cx, cy), w, h, fill=False, ec="lime", lw=2))
    ax.set_title(f"Anomaly ellipses on {image_path.name}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    schema_lines: list[str] = []

    # 1. Metadata files per phase ---------------------------------------
    metadata_frames: dict[int, pd.DataFrame] = {}
    for phase in (1, 2, 3):
        meta_path = args.reflacx_root / f"metadata_phase_{phase}.csv"
        if not meta_path.exists():
            schema_lines.append(f"[MISSING] {meta_path}")
            continue
        schema_lines.append(describe_csv(meta_path))
        metadata_frames[phase] = pd.read_csv(meta_path)

    if not metadata_frames:
        print("[FATAL] no metadata_phase_*.csv found under --reflacx-root", file=sys.stderr)
        return 1

    # 2. Per-phase, per-reader case counts ------------------------------
    case_counts = []
    for phase, df in metadata_frames.items():
        reader_col = next((c for c in df.columns if "reader" in c.lower() or "rater" in c.lower()), None)
        case_counts.append({
            "phase": phase,
            "n_trials": len(df),
            "n_unique_dicom": df["dicom_id"].nunique() if "dicom_id" in df.columns else None,
            "n_readers": df[reader_col].nunique() if reader_col else None,
        })
    pd.DataFrame(case_counts).to_csv(args.out_dir / "case_counts_by_phase.csv", index=False)

    # 3. Sample trial dirs and print schemas ----------------------------
    # REFLACX trial dirs under main_data/ are named with a participant+reading
    # code like "P102R009922" (NOT the dicom_id). The metadata `id` column
    # matches the directory name. Confirmed against the PhysioNet docs.
    main_data = args.reflacx_root / "main_data"
    if not main_data.exists():
        schema_lines.append(f"[WARN] {main_data} not found — checking alternate layouts")
        candidate_trials = [d for d in args.reflacx_root.iterdir() if d.is_dir()]
    else:
        candidate_trials = [d for d in main_data.iterdir() if d.is_dir()]

    sample_trials = candidate_trials[: args.n_trial_samples]
    for trial_dir in sample_trials:
        schema_lines.append(f"\n##### trial: {trial_dir.name} #####")
        for csv_name in [
            "anomaly_location_ellipses.csv",
            "chest_bounding_box.csv",
            "fixations.csv",
            "gaze.csv",
            "timestamps_transcription.csv",
            "tobii_calibration_log.csv",
        ]:
            p = trial_dir / csv_name
            if p.exists():
                schema_lines.append(describe_csv(p))
            else:
                schema_lines.append(f"[MISSING in {trial_dir.name}] {csv_name}")
        txt = trial_dir / "transcription.txt"
        if txt.exists():
            schema_lines.append(describe_txt(txt))

    # 4. Image-linkage sanity check -------------------------------------
    schema_lines.append("\n##### MIMIC-CXR-JPG linkage check #####")
    any_df = next(iter(metadata_frames.values()))
    if {"dicom_id", "subject_id", "study_id"}.issubset(any_df.columns):
        sample_row = any_df.iloc[0]
        candidate = resolve_mimic_jpg_path(
            args.mimic_jpg_root,
            sample_row["dicom_id"],
            sample_row["subject_id"],
            sample_row["study_id"],
        )
        schema_lines.append(f"  example resolved path: {candidate}")
        schema_lines.append(f"  exists: {candidate.exists() if candidate else False}")
    else:
        schema_lines.append("  [WARN] metadata missing one of {dicom_id, subject_id, study_id}")

    # 5. One sanity plot of fixations and one of bboxes -----------------
    if sample_trials:
        first = sample_trials[0]
        fix_path = first / "fixations.csv"
        ellipse_path = first / "anomaly_location_ellipses.csv"
        trial_id = first.name
        # Look up this trial via the metadata `id` column (REFLACX docs:
        # the trial directory name is the `id` field). Fall back to
        # dicom_id matching if `id` isn't present.
        if "id" in any_df.columns:
            link_row = any_df[any_df["id"].astype(str) == trial_id]
        else:
            link_row = any_df[any_df["dicom_id"].astype(str) == trial_id]
        if not link_row.empty and fix_path.exists():
            row = link_row.iloc[0]
            img_path = resolve_mimic_jpg_path(
                args.mimic_jpg_root, row["dicom_id"], row["subject_id"], row["study_id"],
            )
            if img_path and img_path.exists():
                try:
                    plot_fixations_on_image(
                        img_path, pd.read_csv(fix_path),
                        args.out_dir / "fig_fixation_density.png",
                    )
                    schema_lines.append(f"  wrote fig_fixation_density.png for {trial_id}")
                except Exception as e:
                    schema_lines.append(f"  [WARN] fixation plot failed: {e}")
                if ellipse_path.exists():
                    try:
                        plot_bboxes_on_image(
                            img_path, pd.read_csv(ellipse_path),
                            args.out_dir / "fig_bbox_example.png",
                        )
                        schema_lines.append(f"  wrote fig_bbox_example.png for {trial_id}")
                    except Exception as e:
                        schema_lines.append(f"  [WARN] bbox plot failed: {e}")
            else:
                schema_lines.append(f"  [WARN] image not found at expected path: {img_path}")

    # 6. Write everything to schema_summary.txt -------------------------
    summary_path = args.out_dir / "schema_summary.txt"
    summary_path.write_text("\n".join(schema_lines), encoding="utf-8")
    print(f"[OK] wrote {summary_path}")
    print(f"[OK] case_counts → {args.out_dir / 'case_counts_by_phase.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
