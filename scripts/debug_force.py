#!/usr/bin/env python3
"""
Debug force extraction - ตรวจสอบ force/cop จาก C3D

Usage:
    python scripts/debug_force.py <path_to.c3d>
"""
import sys
from pathlib import Path

# โปรเจกต์ root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

path = Path(sys.argv[1]) if len(sys.argv) > 1 else _PROJECT_ROOT / "c3d/test Exam_2026-03-06_135002.c3d"
if not path.exists():
    print("Usage: python scripts/debug_force.py <c3d_path>")
    sys.exit(1)

from biomech.c3d_loader import load_motion_file

data = load_motion_file(path)
print("=" * 60)
print("Force extraction debug")
print("=" * 60)
print(f"n_frames: {data.n_frames}")
print(f"force_forces: {data.force_forces is not None}")
print(f"force_cops: {data.force_cops is not None}")

if data.force_forces is not None and data.force_cops is not None:
    ff = data.force_forces
    fc = data.force_cops
    print(f"forces shape: {ff.shape}")
    print(f"cops shape: {fc.shape}")
    mags = (ff ** 2).sum(axis=2) ** 0.5
    max_mag = mags.max()
    print(f"Max force magnitude: {max_mag:.1f} N")
    n_with_force = (mags > 10).sum()
    print(f"Frames with force > 10 N: {n_with_force}")
    for frame in [0, 50, 100, 140]:
        if frame < ff.shape[0]:
            for p in range(ff.shape[1]):
                m = (ff[frame, p] ** 2).sum() ** 0.5
                print(f"  Frame {frame} plate {p}: F={ff[frame,p]}, mag={m:.1f}, COP={fc[frame,p]}")
else:
    print("force_forces or force_cops is None!")
