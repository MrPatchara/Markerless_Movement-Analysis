#!/usr/bin/env python3
"""
Inspect C3D file structure - FORCE_PLATFORM, analog data

Usage:
    python scripts/inspect_c3d.py <path_to.c3d>

ใช้ตรวจสอบโครงสร้าง C3D (BTS Podium, analog vs digital).
"""
import sys
from pathlib import Path

import numpy as np


def inspect_pyc3d(path: Path):
    """Inspect using py-c3d."""
    try:
        import c3d as c3d_lib
    except ImportError:
        print("py-c3d not installed: pip install c3d")
        return
    print("=" * 60)
    print("C3D Inspection (py-c3d)")
    print("=" * 60)
    with open(path, "rb") as f:
        reader = c3d_lib.Reader(f)
        print("\n--- Header ---")
        print("Point rate:", getattr(reader, "point_rate", "N/A"))
        print("First frame:", getattr(reader, "first_frame", "N/A"))
        print("Last frame:", getattr(reader, "last_frame", "N/A"))
        print("Analog used:", getattr(reader, "analog_used", "N/A"))
        print("Analog per frame:", getattr(reader, "analog_per_frame", "N/A"))
        print("Analog rate:", getattr(reader, "analog_rate", "N/A"))

        print("\n--- FORCE_PLATFORM ---")
        has_fp = False
        for gname, g in reader.group_items():
            if gname == "FORCE_PLATFORM":
                has_fp = True
                used = g.get_int16("USED") if hasattr(g, "get_int16") else None
                print(f"  USED (n_plates): {used}")
                for pname, p in g.param_items():
                    try:
                        if hasattr(p, "int_array") and p.int_array is not None:
                            arr = np.asarray(p.int_array)
                            print(f"  {pname}: int_array shape={arr.shape} {arr.ravel()[:16].tolist()}")
                        elif hasattr(p, "float_array") and p.float_array is not None:
                            arr = np.asarray(p.float_array)
                            print(f"  {pname}: float_array shape={arr.shape} {arr.ravel()[:12].tolist()}")
                        else:
                            print(f"  {pname}: (no array)")
                    except Exception as e:
                        print(f"  {pname}: error {e}")
        if not has_fp:
            print("  (FORCE_PLATFORM group not found)")

        print("\n--- ANALOG ---")
        for gname, g in reader.group_items():
            if gname == "ANALOG":
                for pname, p in g.param_items():
                    try:
                        if hasattr(p, "float_array") and p.float_array is not None:
                            arr = np.asarray(p.float_array)
                            print(f"  {pname}: shape={arr.shape} sample={arr.ravel()[:8].tolist()}")
                        elif hasattr(p, "int_array") and p.int_array is not None:
                            arr = np.asarray(p.int_array)
                            print(f"  {pname}: shape={arr.shape} sample={arr.ravel()[:8].tolist()}")
                        else:
                            print(f"  {pname}: (no array)")
                    except Exception as e:
                        print(f"  {pname}: error {e}")
                break

        print("\n--- POINT labels (looking for Force/COP) ---")
        pl = getattr(reader, "point_labels", None)
        if pl:
            labels = [str(l).strip() for l in pl]
            force_cop = [l for l in labels if "force" in l.lower() or "cop" in l.lower() or "grf" in l.lower()]
            if force_cop:
                print("  Found:", force_cop)
            else:
                print("  First 15:", labels[:15])
        else:
            print("  (no labels)")

        print("\n--- Sample frames (first 3) ---")
        for i, (frame_num, points, analog) in enumerate(reader.read_frames()):
            if i >= 3:
                break
            print(f"  Frame {frame_num}: points shape={points.shape if points is not None else None}")
            if analog is not None:
                print(f"    analog shape={analog.shape}, sample channels 0-5: {[float(analog[:, c].mean()) if c < analog.shape[1] else 'N/A' for c in range(6)]}")
            else:
                print("    analog: None")


def inspect_ezc3d(path: Path):
    """Inspect using ezc3d if available."""
    try:
        import ezc3d
    except ImportError:
        return
    try:
        c3d = ezc3d.c3d(str(path))
    except Exception as e:
        print("ezc3d load error:", e)
        return
    print("\n" + "=" * 60)
    print("C3D Inspection (ezc3d)")
    print("=" * 60)
    data = c3d.get("data", {})
    params = c3d.get("parameters", {})
    print("\n--- data keys ---", list(data.keys()))
    if "points" in data:
        print("  points shape:", data["points"].shape)
    if "analog" in data:
        ana = data["analog"]
        print("  analog:", type(ana), "shape:", getattr(ana, "shape", "N/A"))
        if hasattr(ana, "shape") and ana.size > 0:
            print("    sample [0,:5]:", ana[0, :5] if ana.ndim >= 2 else ana[:5])
    print("\n--- FORCE_PLATFORM params ---")
    fp = params.get("FORCE_PLATFORM", {})
    for k, v in fp.items():
        val = v.get("value", v) if isinstance(v, dict) else v
        print(f"  {k}: {type(val)} {val}")
    print("\n--- ANALOG params ---")
    an = params.get("ANALOG", {})
    for k, v in an.items():
        val = v.get("value", v) if isinstance(v, dict) else v
        if hasattr(val, "shape"):
            print(f"  {k}: shape={val.shape} sample={val.flat[:4].tolist() if val.size else 'empty'}")
        else:
            print(f"  {k}: {val}")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not path or not path.exists():
        print("Usage: python scripts/inspect_c3d.py <path_to.c3d>")
        print("Example: python scripts/inspect_c3d.py c3d/mytrial.c3d")
        sys.exit(1)
    inspect_pyc3d(path)
    inspect_ezc3d(path)


if __name__ == "__main__":
    main()
