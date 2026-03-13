#!/usr/bin/env python3
"""
SLS (Single Leg Stance) Analysis - Command Line

Usage:
    python scripts/analyze_sls.py <path_to.c3d_or.json> [--side L|R] [--frame N]

Loads C3D or BTS Brain JSON, computes FPKPA and Limb Stability (LS) scores
at frame of maximal knee flexion (or specified frame).
"""
import argparse
import sys
from pathlib import Path

# โปรเจกต์ root (parent ของ scripts/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from biomech.c3d_loader import load_motion_file
from biomech.sls_analysis import (
    analyze_sls_limb_stability,
    format_limb_stability_report,
)


def main():
    ap = argparse.ArgumentParser(
        description="SLS 2D analysis: FPKPA and Limb Stability scoring"
    )
    ap.add_argument("file", help="Path to C3D or BTS Brain JSON file")
    ap.add_argument("--side", choices=["L", "R"], default="L", help="Stance leg (default: L)")
    ap.add_argument("--frame", type=int, default=None, help="Frame index (default: max knee flexion)")
    ap.add_argument("--no-max-frame", action="store_true", help="Use frame 0 instead of max knee flexion")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)

    print(f"Loading: {path.name}")
    data = load_motion_file(str(path))
    print(f"  Frames: {data.n_frames}, Points: {data.n_points}")
    print()

    result = analyze_sls_limb_stability(
        data,
        frame_idx=args.frame,
        side=args.side,
        use_max_knee_frame=not args.no_max_frame,
    )

    print(format_limb_stability_report(result, data=data))


if __name__ == "__main__":
    main()
