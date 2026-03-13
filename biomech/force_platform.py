"""
Force platform / GRF extraction from C3D files.
Computes ground reaction force (GRF) vector and center of pressure (COP) from analog data.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple


def _param_to_float_array(p, name: str) -> Optional[np.ndarray]:
    """Get float array from param - handles int_array (float bit pattern) from some C3D writers."""
    val = getattr(p, "float_array", None)
    if val is None:
        val = getattr(p, "value", None)
    if val is None:
        val = getattr(p, "int_array", None)
    if val is None:
        return None
    arr = np.asarray(val)
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float64)
    # Reinterpret int32/uint32 as float32 (BTS Podium stores floats as int bit pattern)
    return np.asarray(arr, dtype=np.int32).view(np.float32).astype(np.float64).copy()


def compute_lab_axes_from_force_plate(
    corners_list: List[np.ndarray],
    lasis: Optional[np.ndarray] = None,
    rasis: Optional[np.ndarray] = None,
    points_sample: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute lab axes from force plate corners for accurate frontal plane projection.

    Uses the force plate surface to define true vertical (gravity) and derives
    medial-lateral direction for frontal plane (ML × Vertical).

    Args:
        corners_list: List of (4,3) corner arrays per plate (lab coordinates).
        lasis, rasis: Optional L/R ASIS positions for person's ML direction.
        points_sample: Optional (n,3) skeleton points to determine "up" direction.

    Returns:
        (vertical_unit, ml_unit) or None if unavailable.
        - vertical_unit: points up (normal to force plate)
        - ml_unit: medial-lateral in horizontal plane (left-right of person)
    """
    if not corners_list:
        return None
    crn = np.asarray(corners_list[0], dtype=float)
    if crn.shape == (3, 4):
        crn = crn.T  # -> (4, 3)
    if crn.shape[0] < 4 or crn.shape[1] < 3:
        return None
    # Two edges of the plate (order may vary by manufacturer)
    e1 = crn[1] - crn[0]
    e2 = crn[3] - crn[0]
    normal = np.cross(e1, e2)
    n = np.linalg.norm(normal)
    if n < 1e-8:
        return None
    vertical = normal / n

    # Ensure vertical points "up" using skeleton centroid if available
    plate_center = np.mean(crn[:4], axis=0)
    if points_sample is not None and points_sample.size >= 3:
        valid = np.isfinite(points_sample).all(axis=1)
        if np.any(valid):
            centroid = np.nanmean(points_sample[valid], axis=0)
            up_component = np.dot(centroid - plate_center, vertical)
            if up_component < 0:
                vertical = -vertical
    else:
        # Fallback: assume vertical points toward +Y or +Z (common lab conventions)
        if vertical[1] < 0 and vertical[2] < 0:
            vertical = -vertical

    # ML direction: prefer person's L-R (ASIS line) projected onto horizontal
    if lasis is not None and rasis is not None and np.isfinite(lasis).all() and np.isfinite(rasis).all():
        hip_line = np.asarray(rasis, dtype=float) - np.asarray(lasis, dtype=float)
        hip_line_h = hip_line - np.dot(hip_line, vertical) * vertical
        hn = np.linalg.norm(hip_line_h)
        if hn > 1e-8:
            ml_vec = hip_line_h / hn
        else:
            ml_vec = _ml_from_plate_edge(e1, vertical)
    else:
        ml_vec = _ml_from_plate_edge(e1, vertical)

    if ml_vec is None:
        return None
    return vertical.astype(np.float64), ml_vec.astype(np.float64)


def _ml_from_plate_edge(edge: np.ndarray, vertical: np.ndarray) -> Optional[np.ndarray]:
    """Project plate edge onto horizontal plane for ML direction."""
    e_h = edge - np.dot(edge, vertical) * vertical
    n = np.linalg.norm(e_h)
    if n < 1e-8:
        return None
    return e_h / n


def extract_force_cop_pyc3d(filepath: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract force and COP from C3D using py-c3d.
    Returns (forces, cops) or None if unavailable.
    - forces: (n_frames, n_plates, 3) - Fx, Fy, Fz in N
    - cops: (n_frames, n_plates, 3) - COP x, y, z in lab coords
    """
    try:
        import c3d as c3d_lib
    except ImportError:
        return None

    with open(filepath, "rb") as f:
        reader = c3d_lib.Reader(f)
        n_frames = getattr(reader, "frame_count", 0) or (
            getattr(reader, "last_frame", 0) - getattr(reader, "first_frame", 1) + 1
        )
        if n_frames <= 0:
            return None

        n_plates = 0
        channels = []
        origins = []
        corners_list = []

        for gname, g in reader.group_items():
            if gname == "FORCE_PLATFORM":
                n_plates = 0
                try:
                    n_plates = int(g.get_int16("USED"))
                except (TypeError, AttributeError):
                    p = g.get("USED", None)
                    if p is not None:
                        arr = getattr(p, "int_array", None)
                        if arr is None:
                            arr = getattr(p, "value", None)
                        if arr is not None:
                            n_plates = int(np.asarray(arr).flat[0])
                if n_plates <= 0:
                    return None
                ch = g.get("CHANNEL", None)
                if ch is not None:
                    val = getattr(ch, "int_array", None)
                    if val is None:
                        val = getattr(ch, "value", None)
                    if val is not None:
                        arr = np.asarray(val).flatten()
                        n_ch = min(n_plates * 6, len(arr))
                        for i in range(n_plates):
                            idx = i * 6
                            if idx + 6 <= n_ch:
                                channels.append([int(arr[idx + j]) for j in range(6)])
                org = g.get("ORIGIN", None)
                if org is not None:
                    arr = _param_to_float_array(org, "ORIGIN")
                    if arr is not None:
                        if arr.ndim == 2:
                            if arr.shape[0] == 3 and arr.shape[1] >= n_plates:
                                origins = [arr[:3, i].copy() for i in range(n_plates)]
                            elif arr.shape[0] >= n_plates and arr.shape[1] == 3:
                                origins = [arr[i, :3].copy() for i in range(n_plates)]
                            else:
                                origins = [np.array(arr.flat[:3]) for _ in range(n_plates)]
                        elif arr.size >= 3:
                            origins = [np.array(arr.flat[:3]) for _ in range(n_plates)]
                cor = g.get("CORNERS", None)
                if cor is not None:
                    arr = _param_to_float_array(cor, "CORNERS")
                    if arr is not None:
                        try:
                            if arr.ndim == 3:
                                corners_list = [arr[:, :, i].T for i in range(arr.shape[2])]
                            elif arr.ndim == 2:
                                r, c = arr.shape[0], arr.shape[1]
                                if r == 3 and c >= 4:
                                    n_plates_crn = max(1, c // 4)
                                    corners_list = [
                                        np.column_stack([arr[:, 4 * i + j] for j in range(4)]).T
                                        for i in range(n_plates_crn)
                                    ]
                                elif r >= 4 and c == 3:
                                    corners_list = [arr[:4, :]]
                                else:
                                    corners_list = []
                            else:
                                corners_list = []
                        except (IndexError, ValueError):
                            corners_list = []
                break

        if not channels:
            channels = [list(range(6 * p, 6 * p + 6)) for p in range(n_plates)]

        analog_scale = 1.0
        gen_scale = 1.0
        for gname, g in reader.group_items():
            if gname == "ANALOG":
                try:
                    s = g.get("SCALE", None)
                    if s is not None:
                        val = getattr(s, "float_array", None)
                        if val is None:
                            val = getattr(s, "value", None)
                        if val is not None:
                            analog_scale = float(np.asarray(val).flat[0]) if np.size(val) else 1.0
                except Exception:
                    pass
                try:
                    gs = g.get("GEN_SCALE", None)
                    if gs is not None:
                        val = getattr(gs, "float_array", None)
                        if val is None:
                            val = getattr(gs, "value", None)
                        if val is not None:
                            gen_scale = float(np.asarray(val).flat[0]) if np.size(val) else 1.0
                except Exception:
                    pass
                break

        f.close()

    with open(filepath, "rb") as f:
        reader = c3d_lib.Reader(f)
        forces_list = []
        cops_list = []
        analog_per_frame = getattr(reader, "analog_per_frame", 1)
        first_frame = getattr(reader, "first_frame", 1)

        for frame_num, points, analog in reader.read_frames():
            if analog is None or analog.size == 0:
                forces_list.append(np.zeros((n_plates, 3)))
                cops_list.append(np.zeros((n_plates, 3)))
                continue

            f_frame = np.zeros((n_plates, 3))
            c_frame = np.zeros((n_plates, 3))
            scale = analog_scale * gen_scale

            # py-c3d analog: (n_channels, n_samples) e.g. (12,10) or (n_samples, n_channels)
            if analog.ndim < 2:
                n_chan = 0
                def get_channel(c): return 0.0
            elif analog.shape[0] > analog.shape[1]:
                # (channels, samples) e.g. (12, 10) - more rows = channels
                n_chan = analog.shape[0]
                def get_channel(c): return float(analog[c, :].mean()) if c < n_chan else 0.0
            else:
                # (samples, channels) e.g. (10, 12) - more cols = channels
                n_chan = analog.shape[1]
                def get_channel(c): return float(analog[:, c].mean()) if c < n_chan else 0.0

            for p in range(n_plates):
                ch = channels[p] if p < len(channels) else list(range(6 * p, 6 * p + 6))
                if len(ch) < 6:
                    continue
                ci = [max(0, int(c) - 1) for c in ch[:6]]  # C3D channels 1-based
                fx = get_channel(ci[0]) * scale
                fy = get_channel(ci[1]) * scale
                fz = get_channel(ci[2]) * scale
                mx = get_channel(ci[3]) * scale
                my = get_channel(ci[4]) * scale
                mz = get_channel(ci[5]) * scale
                f_frame[p] = [fx, fy, fz]
                cop_x, cop_y = 0.0, 0.0
                if abs(fz) > 1.0:
                    cop_x = -my / fz
                    cop_y = mx / fz
                if corners_list and p < len(corners_list):
                    try:
                        crn = np.asarray(corners_list[p]).reshape(-1, 3)
                    except (ValueError, IndexError):
                        crn = np.zeros((0, 3))
                    if len(crn) >= 4:
                        center = np.mean(crn, axis=0)
                        c0, c1, c2, c3 = crn[0], crn[1], crn[2], crn[3]
                        x_dir = (c0 + c3) / 2 - center
                        y_dir = (c0 + c1) / 2 - center
                        nx = np.linalg.norm(x_dir)
                        ny = np.linalg.norm(y_dir)
                        if nx > 1e-6 and ny > 1e-6:
                            c_frame[p] = center + cop_x * (x_dir / nx) + cop_y * (y_dir / ny)
                        else:
                            if p < len(origins):
                                c_frame[p, :2] = [cop_x, cop_y]
                                c_frame[p] += origins[p]
                            c_frame[p, 2] = np.mean(crn[:, 2])
                    else:
                        c_frame[p, 0], c_frame[p, 1] = cop_x, cop_y
                        if p < len(origins):
                            c_frame[p] += origins[p]
                        if len(crn) > 0:
                            c_frame[p, 2] = np.mean(crn[:, 2])
                else:
                    c_frame[p, 0], c_frame[p, 1] = cop_x, cop_y
                    if p < len(origins):
                        c_frame[p] += origins[p]

            forces_list.append(f_frame)
            cops_list.append(c_frame)

        forces = np.array(forces_list)
        cops = np.array(cops_list)
        return forces, cops


def extract_force_cop_ezc3d(filepath: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract force and COP from C3D using ezc3d."""
    try:
        import ezc3d  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        c3d = ezc3d.c3d(str(filepath))
    except Exception:
        return None
    data = c3d.get("data", {})
    if "analog" not in data:
        return None
    # ezc3d structure varies; try common layout
    ana = data["analog"]
    if hasattr(ana, "shape"):
        pass
    return None


def extract_force_cop(filepath: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract force and COP from C3D. Uses py-c3d (ezc3d optional)."""
    result = extract_force_cop_pyc3d(filepath)
    if result is not None:
        return result
    return extract_force_cop_ezc3d(filepath)
