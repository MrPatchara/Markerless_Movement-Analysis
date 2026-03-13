"""
GRF metrics – inclination, body weight, %BW
"""
from __future__ import annotations

import numpy as np
from typing import Optional

# Try C3DData type
try:
    from biomech.c3d_loader import C3DData
except ImportError:
    from .c3d_loader import C3DData


G = 9.81  # m/s²


def grf_inclination_deg(fx: float, fy: float, fz: float) -> float:
    """
    มุมเอียงของ GRF จากแนวดิ่ง (degrees).
    0° = ตั้งฉากกับพื้น, 90° = แนวนอน.
    """
    mag = np.sqrt(fx * fx + fy * fy + fz * fz)
    if mag < 1.0:
        return 0.0
    # แกนที่มีแรงมากสุด = แนวหลัก (มักเป็น vertical)
    ax = np.argmax(np.abs([fx, fy, fz]))
    f_vertical = [fx, fy, fz][ax]
    cos_angle = abs(f_vertical) / mag
    cos_angle = np.clip(cos_angle, 0.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def get_body_weight_from_c3d(filepath) -> Optional[float]:
    """
    อ่าน body weight (kg) จาก C3D – SUBJECT:WEIGHT.
    Returns None ถ้าไม่มีหรืออ่านไม่ได้.
    """
    filepath = __import__("pathlib").Path(filepath)
    if filepath.suffix.lower() != ".c3d":
        return None
    try:
        import c3d as c3d_lib
        with open(filepath, "rb") as f:
            reader = c3d_lib.Reader(f)
            for gname, g in reader.group_items():
                if gname.upper() == "SUBJECT":
                    try:
                        w = g.get("WEIGHT", None)
                        if w is not None:
                            val = getattr(w, "float_array", None)
                            if val is None:
                                val = getattr(w, "value", None)
                            if val is not None:
                                arr = np.asarray(val).flatten()
                                if arr.size > 0:
                                    return float(arr[0])
                    except Exception:
                        pass
                    break
    except Exception:
        pass
    try:
        import ezc3d
        c3d = ezc3d.c3d(str(filepath))
        params = c3d.get("parameters", {})
        subj = params.get("SUBJECT", params.get("SUBJECTS", {}))
        wp = subj.get("WEIGHT", None)
        if wp is None:
            pass
        elif isinstance(wp, dict):
            w = wp.get("value", wp.get("values", []))
            arr = np.asarray(w).flatten()
            if arr.size > 0:
                return float(arr[0])
        else:
            try:
                return float(np.asarray(wp).flat[0])
            except (TypeError, ValueError):
                pass
    except Exception:
        pass
    return None


def estimate_body_weight_from_force(
    forces: np.ndarray,
    fps: float,
    n_seconds_static: float = 1.0,
    min_force_n: float = 100.0,
) -> Optional[float]:
    """
    ประมาณน้ำหนักตัว (N) จากช่วงยืนนิ่ง (static standing).
    ใช้เฟรมแรก n_seconds_static วินาที ที่ |F| > min_force_N
    มาหา median เป็น BW.
    """
    if forces is None or forces.size == 0:
        return None
    n_frames = forces.shape[0]
    n_static = min(int(fps * n_seconds_static), n_frames)
    if n_static <= 0:
        return None
    mags = np.linalg.norm(forces[:n_static], axis=2)
    flat = mags.ravel()
    valid = flat >= min_force_n
    if not np.any(valid):
        return None
    return float(np.median(flat[valid]))


def get_body_weight_n(
    data: C3DData,
    filepath: Optional[str] = None,
) -> Optional[float]:
    """
    น้ำหนักตัวเป็น N (สำหรับ % BW).
    ลำดับ: 1) จาก C3D SUBJECT:WEIGHT  2) ประมาณจากช่วงยืนนิ่ง
    """
    info = get_body_weight_info(data, filepath)
    return info["bw_n"] if info else None


def get_body_weight_info(
    data: C3DData,
    filepath: Optional[str] = None,
    manual_bw_kg: Optional[float] = None,
) -> Optional[dict]:
    """
    คืน (bw_n, bw_kg, source) สำหรับแสดงผล.
    ลำดับ: 1) manual_bw_kg  2) C3D SUBJECT:WEIGHT  3) ประมาณจากช่วงยืนนิ่ง
    """
    if manual_bw_kg is not None and manual_bw_kg > 0:
        return {
            "bw_n": manual_bw_kg * G,
            "bw_kg": manual_bw_kg,
            "source": "กรอกเอง (Manual)",
        }
    if filepath:
        bw_kg = get_body_weight_from_c3d(filepath)
        if bw_kg is not None and bw_kg > 0:
            return {"bw_n": bw_kg * G, "bw_kg": bw_kg, "source": "C3D (SUBJECT:WEIGHT)"}
    ff = getattr(data, "force_forces", None)
    if ff is not None:
        est_n = estimate_body_weight_from_force(ff, data.frame_rate)
        if est_n is not None and est_n > 0:
            return {
                "bw_n": est_n,
                "bw_kg": est_n / G,
                "source": "ประมาณจากช่วงยืนนิ่ง 1s",
            }
    return None
