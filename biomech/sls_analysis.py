"""
SLS (Single Leg Stance) 2D Video Analysis Module
อ้างอิง: Di Paolo et al. The Knee 2024 – การวิเคราะห์ทั้งหมดใช้ระนาบ frontal (frontal plane).

--- Frontal plane ตาม paper ---
• ระนาบ frontal = แกน ML (medial–lateral) × แกนแนวตั้ง (vertical).
• เมื่อมี force plate: ใช้แกนจาก force plate (vertical = gravity, ML จากแผ่น) เพื่อให้ตรงกับโลก.
• เมื่อไม่มี force plate: ใช้แกนจาก skeleton (axis ที่มี range มาก = ML, แกนขึ้น = vertical).

1) Limb Stability (LS) – FPKPA
   - มุมระหว่างเส้น KJC→AJC กับ ASIS→KJC ใน frontal plane.
   - คะแนน: 0/2 ถ้า >25°, 1/2 ถ้า 10–25°, 2/2 ถ้า <10°.

2) Pelvis Stability (PS) – PA (Pelvis Tilt Angle)
   - มุมของเส้น LASIS–RASIS จากแนวนอน ใน frontal plane.
   - PA+ = contralateral drop, PA- = ipsilateral drop.
   - คะแนน: 0/2 ถ้า |PA| >25°, 1/2 ถ้า 10–25°, 2/2 ถ้า <10°.

3) Trunk Stability (TS) – TA (Trunk Tilt Angle)
   - มุมระหว่างแนว trunk (clavicular notch → midline pelvis) กับแนวตั้ง (gravity) ใน frontal plane เท่านั้น.
   - Clavicular notch = midpoint L/R shoulder; midline pelvis = midpoint LASIS–RASIS.
   - แนวตั้ง = แกนโลก (จาก force plate เมื่อมี).
   - คะแนน: 0/2 ถ้า |TA| >10°, 1/2 ถ้า 5–10°, 2/2 ถ้า <5°.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

# Try to use C3DData from c3d_loader
try:
    from biomech.c3d_loader import C3DData
except ImportError:
    from .c3d_loader import C3DData

try:
    from biomech.force_platform import compute_lab_axes_from_force_plate
except ImportError:
    from .force_platform import compute_lab_axes_from_force_plate


# --- Marker / keypoint mapping ---
# VICON-style C3D marker names (common in gait labs)
VICON_MARKERS = {
    "LKJC": ["LKNE", "LKNEE", "LKnee", "L_Knee", "LKNE"],
    "RKJC": ["RKNE", "RKNEE", "RKnee", "R_Knee", "RKNE"],
    "LAJC": ["LANK", "LANKLE", "LAnkle", "L_Ankle", "LANK"],
    "RAJC": ["RANK", "RANKLE", "RAnkle", "R_Ankle", "RANK"],
    "LASIS": ["LASI", "LASIS", "L_ASIS", "LHIP", "LHip"],
    "RASIS": ["RASI", "RASIS", "R_ASIS", "RHIP", "RHip"],
    "LLFC": ["LLFC", "L_LFC", "LLateralFemoralCondyle"],
    "RLFC": ["RLFC", "R_LFC", "RLateralFemoralCondyle"],
}

# HALPE 26 / BTS Brain keypoint indices (26 points)
# 5=L shoulder, 6=R shoulder (midpoint = clavicular notch proxy); 11=L hip, 12=R hip
HALPE_26_INDICES = {
    "LASIS": 11,
    "RASIS": 12,
    "LKJC": 13,
    "RKJC": 14,
    "LAJC": 15,
    "RAJC": 16,
}
HALPE_26_TRUNK = {"LSHOULDER": 5, "RSHOULDER": 6}


@dataclass
class FPKPAResult:
    """Result of FPKPA calculation at one frame."""
    angle_deg: float
    side: str  # "L" or "R"
    frame_idx: int
    score: int  # 0, 1, or 2
    interpretation: str  # "valgus", "varus", "neutral"
    raw_angle: float  # signed angle
    score_note: str = ""  # e.g. "adequate (<10°)"


@dataclass
class TrunkStabilityResult:
    """Trunk stability (TS). TA = angle of trunk segment (clavicular→midline pelvis) from vertical."""
    ta_deg: Optional[float] = None
    score: Optional[int] = None
    score_note: str = ""
    frame_idx: int = 0
    side: str = ""
    clavicular_2d: Optional[np.ndarray] = None
    midline_pelvis_2d: Optional[np.ndarray] = None
    used_world_vertical: bool = False  # True if TA was computed from force plate (gravity)


@dataclass
class PelvisStabilityResult:
    """Pelvis stability analysis result (PS criterion). PA = pelvis tilt angle in frontal plane."""
    pa_deg: Optional[float] = None  # Pelvis tilt angle (angle of ASIS line from horizontal)
    score: Optional[int] = None  # 0, 1, or 2
    score_note: str = ""
    frame_idx: int = 0
    side: str = ""  # stance leg
    lasis_2d: Optional[np.ndarray] = None
    rasis_2d: Optional[np.ndarray] = None


@dataclass
class LimbStabilityResult:
    """Limb stability analysis result (LS criterion)."""
    fpkpa: Optional[FPKPAResult] = None
    grf_knee_score: Optional[int] = None
    grf_knee_note: Optional[str] = None
    total_ls_score: Optional[float] = None  # 0-2 per paper (LS = FPKPA & GRF combined)
    frame_idx: int = 0
    side: str = ""


@dataclass
class MovementStrategyResult:
    """
    Movement strategy analysis result.
    α = มุมในที่เข่า (knee flexion). β = hip–trunk flexion.
    Scoring: 2/2 แล้วหัก 1 ถ้า α>120° (งอน้อยเกิน) หรือ β>100° (โน้มน้อยเกิน).
    """
    alpha_deg: Optional[float] = None
    beta_deg: Optional[float] = None
    score: Optional[int] = None  # 0–2
    frame_idx: int = 0
    side: str = ""
    alpha_flag: bool = False  # True if α > threshold
    beta_flag: bool = False  # True if β > threshold


def _find_marker_index(labels: List[str], keys: List[str], aliases: Dict[str, List[str]]) -> Optional[int]:
    """Find index of marker by name. Labels may be from C3D or generic (Kpt_0, ...)."""
    labels_lower = [str(l).strip().upper() for l in labels]
    for key in keys:
        for alias in aliases.get(key, [key]):
            alias_upper = alias.upper()
            for i, lb in enumerate(labels_lower):
                if alias_upper in lb or lb == alias_upper:
                    return i
    return None


def _get_point_indices(data: C3DData, side: str) -> Dict[str, Optional[int]]:
    """
    Get indices for KJC, AJC, ASIS, (LFC) for given side (L or R).
    Supports both VICON-style labels and HALPE 26 (Kpt_0..Kpt_25).
    """
    labels = data.labels
    n = data.n_points
    prefix = "L" if side.upper() == "L" else "R"
    out: Dict[str, Optional[int]] = {}

    # --- 1) พยายามจับตามชื่อ marker ก่อน (รองรับทั้งรูปแบบ test exam และ 0001 bb) ---
    norm_labels = [str(l).strip().lower().replace("_", "-") for l in labels]

    def _find_by_contains(candidates) -> Optional[int]:
        for i, name in enumerate(norm_labels):
            for c in candidates:
                if c in name:
                    return i
        return None

    if prefix == "L":
        asis_idx = _find_by_contains(["left-hip", "l-hip"])
        kjc_idx = _find_by_contains(["left-knee", "l-knee"])
        ajc_idx = _find_by_contains(["left-ankle", "l-ankle"])
    else:
        asis_idx = _find_by_contains(["right-hip", "r-hip"])
        kjc_idx = _find_by_contains(["right-knee", "r-knee"])
        ajc_idx = _find_by_contains(["right-ankle", "r-ankle"])

    if asis_idx is not None and kjc_idx is not None and ajc_idx is not None:
        out["ASIS"] = asis_idx
        out["KJC"] = kjc_idx
        out["AJC"] = ajc_idx
        out["LFC"] = None
        return out

    # --- 2) HALPE 26 / BTS Brain: Kpt_0 .. Kpt_25 (no LFC in standard set) ---
    if n == 26:
        out["ASIS"] = HALPE_26_INDICES.get(f"{prefix}ASIS")
        out["KJC"] = HALPE_26_INDICES.get(f"{prefix}KJC")
        out["AJC"] = HALPE_26_INDICES.get(f"{prefix}AJC")
        out["LFC"] = None  # not in HALPE 26
        return out

    # MediaPipe 33, OpenPose, or custom labels
    # Common mappings: 23=L hip, 24=R hip, 25=L knee, 26=R knee, 27=L ankle, 28=R ankle
    if n == 33:
        mp_lower = {"ASIS": 23, "KJC": 25, "AJC": 27} if prefix == "L" else {"ASIS": 24, "KJC": 26, "AJC": 28}
        for k, idx in mp_lower.items():
            out[k] = idx if idx < n else None
        out["LFC"] = None
        return out

    # Try VICON-style names
    keys = [f"{prefix}ASIS", f"{prefix}KJC", f"{prefix}AJC", f"{prefix}LFC"]
    for key in keys:
        name = key[1:] if len(key) > 1 else key
        out[name] = _find_marker_index(labels, [key], VICON_MARKERS)

    return out


def _get_trunk_indices(data: C3DData) -> Dict[str, Optional[int]]:
    """Indices for Trunk Stability: L/R shoulder (midpoint = clavicular notch) and LASIS/RASIS (midline pelvis)."""
    n = data.n_points
    out: Dict[str, Optional[int]] = {"LSHOULDER": None, "RSHOULDER": None, "LASIS": None, "RASIS": None}

    # ใช้ _get_point_indices เพื่อหา hip (แทน ASIS) ก่อน
    idx_l = _get_point_indices(data, "L")
    idx_r = _get_point_indices(data, "R")
    out["LASIS"] = idx_l.get("ASIS") if idx_l else None
    out["RASIS"] = idx_r.get("ASIS") if idx_r else None

    # พยายามหา L/R shoulder จากชื่อ marker ก่อน
    labels = [str(l).strip().lower().replace("_", "-") for l in data.labels]

    def _find_idx(cands) -> Optional[int]:
        for i, name in enumerate(labels):
            for c in cands:
                if c in name:
                    return i
        return None

    l_sh = _find_idx(["left-shoulder", "l-shoulder"])
    r_sh = _find_idx(["right-shoulder", "r-shoulder"])
    if l_sh is not None and r_sh is not None:
        out["LSHOULDER"] = l_sh
        out["RSHOULDER"] = r_sh
        return out

    # fallback: HALPE 26
    if n == 26:
        out["LSHOULDER"] = HALPE_26_TRUNK.get("LSHOULDER")
        out["RSHOULDER"] = HALPE_26_TRUNK.get("RSHOULDER")
        return out
    if n == 33:
        out["LSHOULDER"] = 11
        out["RSHOULDER"] = 12
        return out
    labels = data.labels
    for key, aliases in [
        ("LSHOULDER", ["LSHO", "LSHOULDER", "L_SHOULDER", "LACR"]),
        ("RSHOULDER", ["RSHO", "RSHOULDER", "R_SHOULDER", "RACR"]),
    ]:
        out[key] = _find_marker_index(labels, [key], {key: aliases})
    return out


def _to_frontal_plane(points_3d: np.ndarray, axis_vertical: int = 2, axis_ml: int = 0) -> np.ndarray:
    """
    Project 3D points onto frontal plane (medial-lateral vs vertical) by column selection.
    """
    return points_3d[:, [axis_ml, axis_vertical]]


def _project_to_frontal_from_vectors(
    points_3d: np.ndarray,
    vertical: np.ndarray,
    ml: np.ndarray,
    v_flip: float = 1.0,
) -> np.ndarray:
    """
    Project 3D points onto frontal plane using force-plate-derived axes.
    Returns (n,2): [:,0]=ML, [:,1]=vertical (head up when v_flip>0).
    """
    pts = np.asarray(points_3d, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    ml_coord = np.dot(pts, ml)
    v_coord = v_flip * np.dot(pts, vertical)
    return np.column_stack([ml_coord, v_coord])


def _get_world_vertical_from_force_plate(
    data: C3DData,
    frame_idx: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Get world vertical (gravity) and ML as 3D unit vectors from force plate.
    Returns (vertical_3d, ml_3d, v_flip) or None if force plate unavailable.
    Use this for TA so vertical = แกนแนวตั้งของโลก.
    """
    fp = getattr(data, "force_plates", None)
    if not fp or len(fp) == 0:
        return None
    # Lab-based vertical / ML from force plate geometry
    result = compute_lab_axes_from_force_plate(
        fp, lasis=None, rasis=None, points_sample=None
    )
    if result is None:
        return None
    vertical, ml = result
    # ใช้ตำแหน่ง skeleton ทั้ง trial เทียบกับจุดกึ่งกลาง force plate
    # เพื่อให้แน่ใจว่า v_flip ทำให้หัวอยู่ "เหนือ" แผ่น
    crn = np.asarray(fp[0], dtype=float)
    if crn.shape == (3, 4):
        crn = crn.T
    plate_center = np.nanmean(crn[:4], axis=0)
    all_pts = data.points.reshape(-1, 3)
    valid = np.isfinite(all_pts).all(axis=1)
    if np.any(valid):
        centroid = np.nanmean(all_pts[valid], axis=0)
        # ถ้า dot(centroid - plate_center, vertical) < 0 แสดงว่า vertical ชี้ลง → กลับทิศ
        dot = float(np.dot(centroid - plate_center, vertical))
        v_flip = 1.0 if dot > 0 else -1.0
    else:
        v_flip = 1.0
    return vertical, ml, v_flip


def _get_frontal_projection_vectors(
    data: C3DData,
    frame_idx: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Get (vertical, ml, v_flip) from force plate when available.
    Returns None to fall back to axis-based projection.
    """
    fp = getattr(data, "force_plates", None)
    if not fp or len(fp) == 0:
        return None
    # ดึงแกนแนวตั้ง/ML จาก geometry ของ force plate โดยไม่พึ่งพาเฟรมเดียว
    result = compute_lab_axes_from_force_plate(
        fp, lasis=None, rasis=None, points_sample=None
    )
    if result is None:
        return None
    vertical, ml = result
    # ใช้ centroid ของ skeleton ทั้ง trial เทียบกับจุดกึ่งกลางแผ่นเพื่อกำหนดทิศขึ้น
    crn = np.asarray(fp[0], dtype=float)
    if crn.shape == (3, 4):
        crn = crn.T
    plate_center = np.nanmean(crn[:4], axis=0)
    all_pts = data.points.reshape(-1, 3)
    valid = np.isfinite(all_pts).all(axis=1)
    if np.any(valid):
        centroid = np.nanmean(all_pts[valid], axis=0)
        dot = float(np.dot(centroid - plate_center, vertical))
        v_flip = 1.0 if dot > 0 else -1.0
    else:
        v_flip = 1.0
    return vertical, ml, v_flip


def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray, signed: bool = True) -> float:
    """
    Angle between 2D vectors in degrees.
    If signed: positive = valgus (knee in), negative = varus (knee out).
    Uses cross product for sign (counterclockwise = positive).
    """
    v1 = np.asarray(v1, dtype=float).ravel()[:2]
    v2 = np.asarray(v2, dtype=float).ravel()[:2]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    v1 = v1 / n1
    v2 = v2 / n2
    cos_a = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle_rad = np.arccos(cos_a)
    if signed:
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        angle_rad = np.sign(cross) * angle_rad
    return float(np.degrees(angle_rad))


def compute_fpkpa_from_2d(
    asis_2d: np.ndarray,
    kjc_2d: np.ndarray,
    ajc_2d: np.ndarray,
) -> float:
    """
    Compute FPKPA from 2D points (e.g. from image pixel coords).
    Angle at KJC between (KJC→AJC) and (KJC→ASIS).
    Returns signed angle in degrees. Same convention as compute_fpkpa.
    """
    v1 = np.asarray(ajc_2d, dtype=float).ravel()[:2] - np.asarray(kjc_2d, dtype=float).ravel()[:2]
    v2 = np.asarray(asis_2d, dtype=float).ravel()[:2] - np.asarray(kjc_2d, dtype=float).ravel()[:2]
    return _angle_between_vectors(v1, v2, signed=True)


def compute_fpkpa(
    kjc: np.ndarray,
    ajc: np.ndarray,
    asis: np.ndarray,
    axis_vertical: int = 2,
    axis_ml: int = 0,
) -> float:
    """
    Compute Frontal Plane Knee Projection Angle (FPKPA).

    FPKPA = angle between (1) KJC→AJC and (2) ASIS→KJC, in frontal plane.

    Convention: FPKPA (+) = valgus, FPKPA (-) = varus.
    Returns angle in degrees (signed).
    """
    pts = np.array([kjc, ajc, asis], dtype=float)
    if np.any(~np.isfinite(pts)):
        return float("nan")
    fr = _to_frontal_plane(pts, axis_vertical, axis_ml)
    kjc_f, ajc_f, asis_f = fr[0], fr[1], fr[2]
    v1 = ajc_f - kjc_f   # KJC → AJC
    v2 = asis_f - kjc_f  # ASIS → KJC
    return _angle_between_vectors(v1, v2, signed=True)


def _project_fpkpa_points(
    data: C3DData,
    frame_idx: int,
    kjc: np.ndarray,
    ajc: np.ndarray,
    asis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray, float]]]:
    """
    Project KJC, AJC, ASIS onto frontal plane. Prefer force-plate axes when available.
    Returns (kjc_2d, ajc_2d, asis_2d, fp_config).
    fp_config = (vertical, ml, v_flip) when force plate used, else None.
    """
    pts = np.array([kjc, ajc, asis], dtype=float)
    if np.any(~np.isfinite(pts)):
        return pts[:, :2], pts[:, :2], pts[:, :2], None

    fp_cfg = _get_frontal_projection_vectors(data, frame_idx)
    if fp_cfg is not None:
        vertical, ml, v_flip = fp_cfg
        fr = _project_to_frontal_from_vectors(pts, vertical, ml, v_flip)
        return fr[0], fr[1], fr[2], fp_cfg

    axis_ml, axis_v = _get_frontal_axes(data)
    _, _, v_flip = _compute_frontal_orientation(data)
    fr = _to_frontal_plane(pts, axis_vertical=axis_v, axis_ml=axis_ml)
    fr[:, 1] *= v_flip
    return fr[0], fr[1], fr[2], None


def score_fpkpa(angle_deg: float) -> Tuple[int, str]:
    """
    Score FPKPA:
    - 0/2 if deviation > 25°
    - 1/2 if 10° ≤ deviation ≤ 25°
    - 2/2 if deviation < 10°

    Angle is deviation from straight (180°) in frontal plane.
    """
    if np.isnan(angle_deg):
        return 0, "N/A (invalid)"
    a = abs(angle_deg)
    deviation = min(a, 180 - a) if a <= 180 else 0
    if deviation > 25:
        return 0, "non-adequate (>25°)"
    if 10 <= deviation <= 25:
        return 1, "partially adequate (10–25°)"
    return 2, "adequate (<10°)"


def score_trunk_angle(angle_deg: float) -> Tuple[int, str]:
    """Score Trunk Tilt Angle (TA): 0/2 if >10°, 1/2 if 5–10°, 2/2 if <5°."""
    if np.isnan(angle_deg):
        return 0, "N/A (invalid)"
    a = abs(angle_deg)
    if a > 10:
        return 0, "non-adequate (>10°)"
    if 5 <= a <= 10:
        return 1, "partially adequate (5–10°)"
    return 2, "adequate (<5°)"


def _get_frontal_axes(data: C3DData) -> Tuple[int, int]:
    """
    Return (axis_ml, axis_vertical) for frontal plane projection.
    BTS Brain: X=lateral, Y=vertical, Z=depth → use (0,1).
    VICON/C3D: often X=forward, Y=up, Z=lateral → use (2,1) for ML vs vertical.
    If results look wrong, try axis_config in compute_fpkpa/analyze_sls_limb_stability.
    """
    if getattr(data, "source", "") == "json":
        return 0, 2  # X = lateral, Z = vertical
    return 0, 2  # X = lateral, Z = vertical (common C3D)


def _compute_frontal_orientation(data: C3DData) -> Tuple[int, int, float]:
    """
    Same logic as 3D viewer: detect vertical axis and flip for upright.
    Returns (axis_ml, axis_v, v_flip) for frontal plane with head up.
    """
    all_pts = data.points.reshape(-1, 3)
    valid = np.isfinite(all_pts).all(axis=1)
    if not np.any(valid):
        return 0, 2, 1.0
    pts = all_pts[valid]
    extents = np.ptp(pts, axis=0)
    up_axis = int(np.argmax(extents))
    others = [i for i in range(3) if i != up_axis]
    up_vals = pts[:, up_axis]
    mid = np.median(up_vals)
    above = np.nanmean(up_vals[up_vals > mid]) if np.any(up_vals > mid) else mid
    below = np.nanmean(up_vals[up_vals <= mid]) if np.any(up_vals <= mid) else mid
    v_flip = 1.0 if above > below else -1.0
    axis_ml = others[0]
    axis_v = up_axis
    return axis_ml, axis_v, v_flip


def analyze_frame_fpkpa(
    data: C3DData,
    frame_idx: int,
    side: str = "L",
) -> Optional[FPKPAResult]:
    """
    Compute FPKPA for one frame and one leg.

    Args:
        data: C3D motion data
        frame_idx: Frame index
        side: "L" or "R" for stance leg

    Returns:
        FPKPAResult or None if required markers missing
    """
    pts = data.get_frame(frame_idx)
    indices = _get_point_indices(data, side)
    kjc_i = indices.get("KJC")
    ajc_i = indices.get("AJC")
    asis_i = indices.get("ASIS")
    if kjc_i is None or ajc_i is None or asis_i is None:
        return None
    kjc = pts[kjc_i]
    ajc = pts[ajc_i]
    asis = pts[asis_i]
    if np.any(~np.isfinite([kjc, ajc, asis])):
        return None
    kjc_2d, ajc_2d, asis_2d, _ = _project_fpkpa_points(data, frame_idx, kjc, ajc, asis)
    angle = compute_fpkpa_from_2d(asis_2d, kjc_2d, ajc_2d)
    score, note = score_fpkpa(angle)
    interp = "valgus" if angle > 0 else ("varus" if angle < 0 else "neutral")
    if side == "L":
        interp = "varus" if interp == "valgus" else "valgus"
    # Report deviation from straight for clarity
    dev = min(abs(angle), 180 - abs(angle)) if abs(angle) <= 180 else 0
    return FPKPAResult(
        angle_deg=dev,
        side=side,
        frame_idx=frame_idx,
        score=score,
        interpretation=interp,
        raw_angle=angle,
        score_note=note,
    )


def _compute_knee_angle_sagittal(
    h: np.ndarray, k: np.ndarray, a: np.ndarray,
    axis_vertical: int = 2, axis_ap: int = 1,
) -> float:
    """
    Angle at knee (hip–knee–ankle) in sagittal plane = มุมใน (interior) = knee flexion.
    Project onto sagittal plane. Returns degrees: ยิ่งงอมาก ค่ายิ่งลด (e.g. ~77° when bent, 180° when straight).
    """
    pts = np.array([h, k, a])
    sag = pts[:, [axis_ap, axis_vertical]]  # 2D in sagittal plane
    v1 = sag[0] - sag[1]  # thigh: hip→knee
    v2 = sag[2] - sag[1]  # shank: ankle→knee
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return np.nan
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _angle_sagittal_three_points(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    axis_vertical: int,
    axis_ap: int,
) -> float:
    """
    Generic sagittal-plane angle at p2 between p1→p2 and p3→p2.
    Used for Movement Strategy (side view) angles.
    Returns angle in degrees (0–180).
    """
    pts = np.array([p1, p2, p3])
    if np.any(~np.isfinite(pts)):
        return float("nan")
    sag = pts[:, [axis_ap, axis_vertical]]  # (AP, Vertical)
    v1 = sag[0] - sag[1]
    v2 = sag[2] - sag[1]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return float("nan")
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def get_frames_on_force_plate(
    data: C3DData,
    threshold_n: float = 50.0,
) -> Optional[np.ndarray]:
    """
    คืนค่า mask (n_frames,) = True เมื่อมีแรงบน force plate เกิน threshold (เท้าแตะแผ่น).
    ใช้กับ Single Leg Hop เพื่อจำกัดการหาเฟรมเข่างอสุดเฉพาะช่วงที่เท้าแตะแผ่นแล้ว.
    Returns None ถ้าไม่มีข้อมูล force.
    """
    ff = getattr(data, "force_forces", None)
    if ff is None or ff.size == 0:
        return None
    # (n_frames, n_plates, 3) -> magnitude per frame per plate, then max over plates
    mags = np.linalg.norm(ff, axis=2)  # (n_frames, n_plates)
    max_mag_per_frame = np.nanmax(mags, axis=1)
    return (max_mag_per_frame >= threshold_n).astype(bool)


def _compute_knee_flexion_frame(
    data: C3DData,
    side: str,
    frame_mask: Optional[np.ndarray] = None,
) -> Tuple[int, float]:
    """
    Find frame with maximum knee flexion for one leg (raw values, no smoothing).
    Angle from _compute_knee_angle_sagittal = มุมใน (interior) = knee flexion; ยิ่งงอมาก ค่ายิ่งน้อย.
    frame_mask: optional (n_frames,) bool; if given, only consider frames where True.
    Returns (frame_idx, knee_flexion_deg). Best frame = frame where angle is minimum (most bent).
    """
    indices = _get_point_indices(data, side)
    hi, ki, ai = indices.get("ASIS"), indices.get("KJC"), indices.get("AJC")
    if hi is None or ki is None or ai is None:
        return 0, 0.0
    axis_ml, axis_v = _get_frontal_axes(data)
    axis_ap = 3 - axis_ml - axis_v
    angles = []
    for f in range(data.n_frames):
        if frame_mask is not None and f < len(frame_mask) and not frame_mask[f]:
            angles.append(np.nan)
            continue
        pts = data.get_frame(f)
        h, k, a = pts[hi], pts[ki], pts[ai]
        if np.any(~np.isfinite([h, k, a])):
            angles.append(np.nan)
            continue
        angle = _compute_knee_angle_sagittal(h, k, a, axis_vertical=axis_v, axis_ap=axis_ap)
        angles.append(angle)
    angles = np.array(angles, dtype=float)
    valid = np.isfinite(angles)
    if not np.any(valid):
        return 0, 0.0
    # มุมใน: ยิ่งงอมาก ค่ายิ่งน้อย → เฟรมที่งอสุด = argmin
    angles_for_min = np.where(valid, angles, np.inf)
    best_frame = int(np.argmin(angles_for_min))
    return best_frame, float(angles[best_frame])


def find_frame_max_knee_flexion(
    data: C3DData,
    side: str = "L",
    restrict_to_on_plate: bool = False,
) -> int:
    """
    Find frame with maximal knee flexion for given leg.
    restrict_to_on_plate: if True (e.g. Single Leg Hop), only consider frames where
    the stance foot is on the force plate (vertical force above threshold).
    """
    frame_mask = None
    if restrict_to_on_plate:
        frame_mask = get_frames_on_force_plate(data, threshold_n=50.0)
        if frame_mask is not None and not np.any(frame_mask):
            frame_mask = None  # no frame on plate -> fall back to all frames
    frame, _ = _compute_knee_flexion_frame(data, side, frame_mask=frame_mask)
    return frame


def get_knee_flexion_deg(data: C3DData, frame_idx: int, side: str = "L") -> Optional[float]:
    """
    Knee flexion = มุมในที่เข่า (interior angle, sagittal). Same as Alpha in MS.
    Returns degrees: ค่าที่ได้ = มุมงอเข่า (ยิ่งงอมาก ค่ายิ่งน้อย). None if markers missing.
    """
    indices = _get_point_indices(data, side)
    hi, ki, ai = indices.get("ASIS"), indices.get("KJC"), indices.get("AJC")
    if hi is None or ki is None or ai is None:
        return None
    pts = data.get_frame(frame_idx)
    h, k, a = pts[hi], pts[ki], pts[ai]
    if np.any(~np.isfinite([h, k, a])):
        return None
    axis_ml, axis_v = _get_frontal_axes(data)
    axis_ap = 3 - axis_ml - axis_v
    angle = _compute_knee_angle_sagittal(h, k, a, axis_vertical=axis_v, axis_ap=axis_ap)
    if np.isnan(angle):
        return None
    return float(angle)


def _get_movement_strategy_points(
    data: C3DData,
    frame_idx: int,
    side: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Get 3D points (LM, LFC, GT, SJC) for Movement Strategy.
    LM: distal shank/ankle (approx AJC). LFC: lateral femoral condyle (approx knee joint).
    GT: greater trochanter (approx hip joint). SJC: shoulder joint center (midpoint of L/R shoulder).
    รองรับทั้ง HALPE 26 และโมเดล markerless อื่น (เช่น 2GRF) โดยพยายามจับตามชื่อก่อน แล้วจึง fallback.
    """
    pts = data.get_frame(frame_idx)
    side = side.upper()
    n = data.n_points

    # --- 1) พยายามจับตามชื่อ marker ก่อน (เช่น 2GRF: Left-hip, Left-knee, Left-ankle, Left-shoulder, Right-shoulder) ---
    labels_norm = [str(l).strip().lower().replace("_", "-") for l in data.labels]

    def _find_idx(cands: List[str]) -> Optional[int]:
        for i, name in enumerate(labels_norm):
            for c in cands:
                if c in name:
                    return i
        return None

    if side == "L":
        lm_i = _find_idx(["left-ankle", "l-ankle"])
        lfc_i = _find_idx(["left-knee", "l-knee"])
        gt_i = _find_idx(["left-hip", "l-hip"])
    else:
        lm_i = _find_idx(["right-ankle", "r-ankle"])
        lfc_i = _find_idx(["right-knee", "r-knee"])
        gt_i = _find_idx(["right-hip", "r-hip"])
    lsho_i = _find_idx(["left-shoulder", "l-shoulder"])
    rsho_i = _find_idx(["right-shoulder", "r-shoulder"])

    if None not in (lm_i, lfc_i, gt_i, lsho_i, rsho_i):
        lm = pts[lm_i]
        lfc = pts[lfc_i]
        gt = pts[gt_i]
        sjc = (pts[lsho_i] + pts[rsho_i]) / 2.0
        arr = np.array([lm, lfc, gt, sjc], dtype=float)
        if np.isfinite(arr).all():
            return lm.astype(float), lfc.astype(float), gt.astype(float), sjc.astype(float)

    # --- 2) HALPE 26 / BTS Brain markerless (fallback) ---
    if n == 26:
        if side == "R":
            lm_i = HALPE_26_INDICES.get("RAJC")
            lfc_i = HALPE_26_INDICES.get("RKJC")
            gt_i = HALPE_26_INDICES.get("RASIS")
        else:
            lm_i = HALPE_26_INDICES.get("LAJC")
            lfc_i = HALPE_26_INDICES.get("LKJC")
            gt_i = HALPE_26_INDICES.get("LASIS")
        lsho_i = HALPE_26_TRUNK.get("LSHOULDER")
        rsho_i = HALPE_26_TRUNK.get("RSHOULDER")
        if None not in (lm_i, lfc_i, gt_i, lsho_i, rsho_i):
            lm = pts[lm_i]
            lfc = pts[lfc_i]
            gt = pts[gt_i]
            sjc = (pts[lsho_i] + pts[rsho_i]) / 2.0
            arr = np.array([lm, lfc, gt, sjc], dtype=float)
            if np.isfinite(arr).all():
                return lm.astype(float), lfc.astype(float), gt.astype(float), sjc.astype(float)

    # --- 3) VICON / C3D with named markers – approximate using existing helpers ---
    idx_side = _get_point_indices(data, side)
    trunk_idx = _get_trunk_indices(data)
    if not idx_side or not trunk_idx:
        return None
    lm_i = idx_side.get("AJC")  # use ankle as LM
    lfc_i = idx_side.get("LFC") or idx_side.get("KJC")  # prefer LFC, fallback to knee joint
    gt_i = idx_side.get("ASIS")  # approximate GT with hip/ASIS
    lsho_i = trunk_idx.get("LSHOULDER")
    rsho_i = trunk_idx.get("RSHOULDER")
    if None in (lm_i, lfc_i, gt_i, lsho_i, rsho_i):
        return None
    lm = pts[lm_i]
    lfc = pts[lfc_i]
    gt = pts[gt_i]
    sjc = (pts[lsho_i] + pts[rsho_i]) / 2.0
    arr = np.array([lm, lfc, gt, sjc], dtype=float)
    if not np.isfinite(arr).all():
        return None
    return lm.astype(float), lfc.astype(float), gt.astype(float), sjc.astype(float)


def find_stance_leg_and_frame(data: C3DData) -> Tuple[int, str]:
    """
    Auto-detect stance leg and frame of max knee flexion for SLS.
    Scans both legs, picks the leg with highest knee flexion (stance leg in SLS).
    Returns (frame_idx, side) e.g. (42, "L").
    """
    frame_l, angle_l = _compute_knee_flexion_frame(data, "L")
    frame_r, angle_r = _compute_knee_flexion_frame(data, "R")
    if angle_r > angle_l:
        return frame_r, "R"
    return frame_l, "L"


def analyze_movement_strategy(
    data: C3DData,
    frame_idx: int,
    side: str = "L",
) -> MovementStrategyResult:
    """
    Analyze Movement Strategy using sagittal (side-view) angles from paper:

    - Alpha (α) = knee flexion = มุมในที่เข่า (LM→LFC กับ LFC→GT). ยิ่งงอมาก α ยิ่งน้อย.
    - Hip–Trunk flexion (Beta, β):
        Angle between LFC→GT and GT→SJC (เข่า–สะโพก–ลำตัว/ไหล่) ในระนาบ sagittal.

    Scoring (ตาม paper): เริ่มจาก 2/2
        - α > 120° → หัก 1 (มุมมาก = งอน้อย/เกือบตรง)
        - β > 100° → หัก 1 (โน้มน้อยเกิน)
    """
    out = MovementStrategyResult(frame_idx=frame_idx, side=side)
    pts = _get_movement_strategy_points(data, frame_idx, side)
    if pts is None:
        return out
    lm, lfc, gt, sjc = pts
    axis_ml, axis_v = _get_frontal_axes(data)
    axis_ap = 3 - axis_ml - axis_v

    alpha = _angle_sagittal_three_points(lm, lfc, gt, axis_vertical=axis_v, axis_ap=axis_ap)
    beta = _angle_sagittal_three_points(lfc, gt, sjc, axis_vertical=axis_v, axis_ap=axis_ap)
    if np.isnan(alpha) or np.isnan(beta):
        return out
    out.alpha_deg = float(alpha)
    out.beta_deg = float(beta)

    # Scoring: start from 2, penalize violations
    score = 2
    if alpha > 120.0:
        score -= 1
        out.alpha_flag = True
    if beta > 100.0:
        score -= 1
        out.beta_flag = True
    out.score = max(0, score)
    return out


def analyze_shock_absorption(
    data: C3DData,
    frame_idx: int,
    side: str = "L",
) -> MovementStrategyResult:
    """
    Shock Absorption (Single Leg Hop): ประเมินจาก Knee flexion (Alpha) เท่านั้น.
    α = มุมที่ LFC ระหว่างเส้น LM→LFC กับ LFC→GT (interior angle ที่เข่า).
    Alpha is the supplementary of KFA (α = 180° − KFA ในบางนิยาม; ที่นี่เราใช้ α = มุมในโดยตรง).

    Scoring (จาก α เท่านั้น):
      - 2/2 if α < 100°
      - 1/2 if 100° ≤ α ≤ 120°
      - 0/2 if α > 120°
    """
    out = MovementStrategyResult(frame_idx=frame_idx, side=side)
    pts = _get_movement_strategy_points(data, frame_idx, side)
    if pts is None:
        return out
    lm, lfc, gt, sjc = pts
    axis_ml, axis_v = _get_frontal_axes(data)
    axis_ap = 3 - axis_ml - axis_v
    alpha = _angle_sagittal_three_points(lm, lfc, gt, axis_vertical=axis_v, axis_ap=axis_ap)
    beta = _angle_sagittal_three_points(lfc, gt, sjc, axis_vertical=axis_v, axis_ap=axis_ap)
    if np.isnan(alpha):
        return out
    out.alpha_deg = float(alpha)
    out.beta_deg = float(beta) if not np.isnan(beta) else None
    # Shock Absorption: score from alpha only
    if alpha > 120.0:
        out.score = 0
        out.alpha_flag = True
    elif alpha >= 100.0:
        out.score = 1
        out.alpha_flag = True
    else:
        out.score = 2
        out.alpha_flag = False
    out.beta_flag = False  # ไม่ใช้ β ในการให้คะแนน SA
    return out


def _append_ms_recommendations(lines: List[str], result: MovementStrategyResult) -> None:
    """Add Movement Strategy recommendations – expert-level."""
    alpha = result.alpha_deg or 0
    beta = result.beta_deg or 0
    score = result.score or 0
    a_flag = result.alpha_flag or False
    b_flag = result.beta_flag or False
    if score == 2:
        lines.append("  [สรุป] กลยุทธ์การเคลื่อนไหวภายในเกณฑ์ – ใช้ knee/hip flexion และการโน้มตัวเหมาะสมกับ SLS")
        lines.append("")
        lines.append("  • รักษาสถานะปัจจุบัน: ฝึก SLS, step-down โดยรักษา movement pattern (โน้มมาพอดี ทรงตัวได้)")
        lines.append("  • การป้องกัน: พิจารณาโปรแกรม neuromuscular control เป็นประจำ")
    elif score == 1:
        lines.append("  [สรุป] มีการใช้ knee หรือ trunk flexion นอกเกณฑ์ – ควรปรับ movement pattern ให้สอดคล้องกับ SLS")
        lines.append("")
        if a_flag and b_flag:
            lines.append("  ■ การตีความเชิงคลินิก")
            lines.append("    α > 120°: งอน้อยเกิน (เกือบตรง); β > 100°: โน้มน้อยเกิน ใน SLS ถ้าไม่โน้มมาพอจะทรงตัวยาก")
            lines.append("    ร่วมกันอาจเป็น quad-dominant, ไม่ใช้ hip/trunk ร่วมอย่างเหมาะสม")
            lines.append("")
            lines.append("  ■ โปรแกรมการฝึกที่แนะนำ")
            lines.append("    • Cue ให้โน้มลำตัว/สะโพกไปข้างหน้าพอสมควรใน SLS (hip hinge) เพื่อทรงตัวได้ดี")
            lines.append("    • ฝึก hip hinge, RDL เพื่อใช้ hip flexion ร่วม ไม่พึ่งแต่เข่า")
            lines.append("    • จำกัดความลึก SLS ก่อน (shallow) แล้วค่อยเพิ่มเมื่อ pattern ดีขึ้น")
            lines.append("    • เสริม gluteus, hamstring เพื่อกระจายภาระจาก quad")
        elif a_flag:
            lines.append("  ■ การตีความเชิงคลินิก")
            lines.append("    α > 120°: งอน้อยเกิน (เข่าเกือบตรง) – ใน SLS ควรงอเข่าพอให้เห็นการควบคุม")
            lines.append("")
            lines.append("  ■ โปรแกรมการฝึก")
            lines.append("    • Cue ให้งอเข่าพอใน SLS (\"ย่อเข่า\" \"สะโพกย่อ\") ไม่ต้องลึกมาก แต่ต้องมีมุมงอชัด")
            lines.append("    • ฝึก SLS แบบ shallow แล้วค่อยเพิ่มความลึกเมื่อ pattern ดี")
            lines.append("    • เสริม quad, hamstring, gluteus ให้ควบคุมเข่าได้ดี")
        else:
            lines.append("  ■ การตีความเชิงคลินิก")
            lines.append("    β > 100°: โน้มน้อยเกิน (ตัวตรงเกิน) – ใน SLS ต้องโน้มมาบ้างถึงจะทรงตัวได้ ไม่โน้มพอจะล้มหรือทรงตัวยาก")
            lines.append("")
            lines.append("  ■ โปรแกรมการฝึก")
            lines.append("    • Cue ให้โน้มลำตัวไปข้างหน้าพอสมควรใน SLS (\"สะโพกย่อไปด้านหลัง\" \"hip hinge\")")
            lines.append("    • ฝึก hip hinge, deadlift เพื่อให้ใช้ trunk/hip flexion ได้อย่างเหมาะสม")
            lines.append("    • Step-down, SLS พร้อมกระจก/วิดีโอ feedback – เน้นโน้มมาพอดี ไม่ตัวตรงเกิน")
        lines.append("")
        lines.append("  ■ ความถี่: 2–4 ครั้ง/สัปดาห์ นาน 6–12 สัปดาห์")
    else:
        lines.append("  [สรุป] กลยุทธ์การเคลื่อนไหวนอกเกณฑ์มาก – อาจงอน้อยเกิน (α>120°) หรือโน้มน้อยเกิน (β>100°) ไม่เหมาะสมกับ SLS")
        lines.append("")
        lines.append("  ■ การส่งต่อและการประเมิน")
        lines.append("    • ประเมิน trunk control (TS), hip strength ร่วม")
        lines.append("    • ตรวจ movement pattern ทั้ง chain (ankle–knee–hip–trunk) และการโน้มตัวใน SLS")
        lines.append("")
        lines.append("  ■ โปรแกรมรักษา (ต้องดูแลโดยผู้เชี่ยวชาญ)")
        lines.append("    • Movement pattern retraining: ฝึกโน้มตัว/hip hinge ให้พอดีใน SLS เพื่อทรงตัวได้")
        lines.append("    • Cueing, strengthening, motor control ร่วมกัน")
        lines.append("    • หลีกเลี่ยง SLS ลึกจนกว่าจะได้คะแนน ≥ 1/2")


def format_movement_strategy_report(
    result: MovementStrategyResult,
    data: Optional[C3DData] = None,
) -> str:
    """
    Format Movement Strategy result similar to paper:
    - Knee flexion (Alpha, α)
    - Hip–Trunk flexion (Beta, β)
    กระทำในระนาบ sagittal (2D side view).
    """
    lines: List[str] = [
        "Movement Strategy (α, β)",
        "",
        f"Stance leg: {result.side}   │   Frame: {result.frame_idx}",
        "",
    ]
    if result.alpha_deg is not None and result.beta_deg is not None and result.score is not None:
        result_lines = [
            f"Alpha (Knee flexion):   {result.alpha_deg:.4f}\u00B0",
            f"Beta  (Hip–Trunk):      {result.beta_deg:.4f}\u00B0",
            f"Score (Movement):       {result.score}/2",
        ]
        flags: List[str] = []
        if result.alpha_flag:
            flags.append("α > 120°")
        if result.beta_flag:
            flags.append("β > 100°")
        if flags:
            result_lines.append("Penalties:             " + ", ".join(flags))
        lines.extend(_box_results(54, "■ ผลการวิเคราะห์ MS", result_lines))
        lines.append("")
        a = result.alpha_deg
        b = result.beta_deg
        lines.append("─── สรุปความหมายผล ───")
        summary_parts = []
        if a is not None:
            summary_parts.append(f"เข่างอ ~{a:.0f}° (α=มุมใน)")
        if result.beta_flag:
            summary_parts.append("โน้มน้อยเกิน → ควรโน้มมาพอดีใน SLS")
        elif b is not None and result.score == 2:
            summary_parts.append("การโน้มตัวอยู่ในเกณฑ์")
        if result.alpha_flag:
            summary_parts.append("α>120° = งอน้อยเกิน/เกือบตรง")
        if summary_parts:
            lines.append("  " + "  ·  ".join(summary_parts))
        lines.append("")
        lines.append("─── นิยามมุม ───")
        lines.append("  α = มุมในที่เข่า (LM→LFC กับ LFC→GT) = knee flexion. ยิ่งงอมาก α ยิ่งน้อย.")
        lines.append("  β = มุมที่สะโพก (LFC→GT กับ GT→SJC). ตัวตรง≈180° ยิ่งโน้มมาข้างหน้า βยิ่งลด")
        lines.append("  จุด: LM=ข้อเท้า  LFC=เข่า  GT=สะโพก  SJC=กลางไหล่")
        lines.append("")
        lines.append("─── บริบท SLS ───")
        lines.append("  ใน SLS ต้องโน้มลำตัวมาบ้างถึงจะทรงตัวได้ ไม่โน้มพอจะล้ม")
        lines.append("  เกณฑ์: หัก 1 เมื่อ α>120° (งอน้อยเกิน) หรือ β>100° (โน้มน้อยเกิน)")
        if a is not None:
            lines.append("")
            lines.append(f"  α = {a:.2f}° = มุมงอเข่า (มุมใน). มุมภายนอก = 180° − α = {180.0 - a:.2f}°")
        if b is not None:
            lines.append(f"  อธิบาย β: {b:.2f}° → β>100° = โน้มน้อยเกิน; β≤100° = โน้มพอดี")
        lines.append("")
        lines.append("─── คำแนะนำ ───")
        _append_ms_recommendations(lines, result)
    else:
        lines.append("Movement Strategy: N/A (required markers not found or invalid)")
    return "\n".join(lines)


def _append_sa_recommendations(lines: List[str], result: MovementStrategyResult) -> None:
    """คำแนะนำตามหลักชีวกลศาสตร์สำหรับ Shock Absorption (Single Leg Hop)."""
    score = result.score or 0
    a = result.alpha_deg or 0
    if score == 2:
        lines.append("  [สรุป] การดูดซับแรงที่เข่า (knee flexion) อยู่ในเกณฑ์ดี – งอเข่าพอในจังหวะลงพื้น")
        lines.append("")
        lines.append("  • รักษา pattern การลงพื้น: ฝึก single-leg hop landing โดยเน้นงอเข่าและสะโพกรับน้ำหนัก")
        lines.append("  • การป้องกัน: พิจารณาโปรแกรม neuromuscular control และ plyometric อย่างต่อเนื่อง")
    elif score == 1:
        lines.append("  [สรุป] มุมงอเข่า (α) อยู่ในช่วง 100°–120° – ดูดซับแรงได้ปานกลาง ควรเพิ่ม knee flexion ในจังหวะลงพื้น")
        lines.append("")
        lines.append("  ■ การตีความเชิงคลินิก")
        lines.append("    α มาก (ใกล้ 120°) = งอเข่าน้อยในจังหวะลง landing → แรงส่งผ่านข้อเข่าและข้อต่ออื่นมากขึ้น")
        lines.append("    ความเสี่ยง: เพิ่มภาระที่ ACL, meniscus และข้อเข่าหากลงแข็งซ้ำๆ")
        lines.append("")
        lines.append("  ■ โปรแกรมการฝึกที่แนะนำ")
        lines.append("    • Cue: \"งอเข่ารับ\" \"ย่อเข่าตอนลง\" ใน single-leg hop landing")
        lines.append("    • ฝึก squat-to-land, drop jump โดยเน้น soft landing (งอเข่า–สะโพก)")
        lines.append("    • เสริม quadriceps eccentrics และ hamstring เพื่อควบคุมการงอเข่าใน landing")
        lines.append("")
        lines.append("  ■ ความถี่: 2–4 ครั้ง/สัปดาห์ นาน 6–12 สัปดาห์ แล้วประเมินซ้ำ")
    else:
        lines.append("  [สรุป] มุมงอเข่าน้อยเกิน (α > 120°) – ลงแข็ง (stiff landing) เสี่ยงบาดเจ็บ ควรปรับ technique การลงพื้น")
        lines.append("")
        lines.append("  ■ การตีความเชิงคลินิก")
        lines.append("    α > 120° = เข่าเกือบตรงในจังหวะลง → shock absorption ต่ำ แรงส่งผ่านข้อเข่าสูง")
        lines.append("")
        lines.append("  ■ โปรแกรมการฝึก (แนะนำภายใต้การดูแล)")
        lines.append("    • เน้น soft landing: ลงแล้วงอเข่า–สะโพกทันที (absorb)")
        lines.append("    • เริ่มจาก drop จากความสูงต่ำ แล้วค่อยเพิ่มเมื่อ pattern ดีขึ้น")
        lines.append("    • หลีกเลี่ยงการลงแข็งหรือเข่าเกือบตรงในกิจกรรมกระโดดจนกว่าจะได้คะแนน ≥ 1/2")


def format_shock_absorption_report(
    result: MovementStrategyResult,
    data: Optional[C3DData] = None,
) -> str:
    """
    Format Shock Absorption report (Single Leg Hop) – มีกรอบผล สรุป และคำแนะนำเหมือน LS/PS/TS/MS.
    """
    lines: List[str] = [
        "Shock Absorption (Single Leg Hop)",
        "",
        f"Stance leg: {result.side}   │   Frame: {result.frame_idx}",
        "",
    ]
    if result.alpha_deg is not None and result.score is not None:
        a = result.alpha_deg
        result_lines = [
            f"α (Knee flexion):        {a:.4f}\u00B0",
            f"Score (Shock Absorption): {result.score}/2",
        ]
        lines.extend(_box_results(54, "■ ผลการวิเคราะห์ SA", result_lines))
        lines.append("")
        lines.append("─── สรุปความหมายผล ───")
        if result.score == 2:
            lines.append("  งอเข่าพอในจังหวะลงพื้น  ·  ดูดซับแรงได้ดี  ·  อยู่ในเกณฑ์")
        elif result.score == 1:
            lines.append(f"  มุมงอเข่า α = {a:.1f}° (100°–120°)  ·  ดูดซับแรงปานกลาง  ·  ควรเพิ่ม knee flexion ตอนลง")
        else:
            lines.append(f"  มุมงอเข่า α = {a:.1f}° (>120°)  ·  ลงแข็ง  ·  ควรปรับ technique การลงพื้น")
        lines.append("")
        lines.append("─── นิยามมุม ───")
        lines.append("  Knee flexion (Alpha): มุมที่ LFC ระหว่างเส้น LM→LFC กับ LFC→GT (มุมในที่เข่า).")
        lines.append("  Alpha is the supplementary of KFA. ยิ่งงอเข่ามาก α ยิ่งน้อย.")
        lines.append("  จุด: LM = ข้อเท้า  LFC = เข่า  GT = สะโพก")
        lines.append("")
        lines.append("─── การให้คะแนน ───")
        lines.append("  0/2  if α > 120°  ·  1/2  if 100° ≤ α ≤ 120°  ·  2/2  if α < 100°")
        lines.append("")
        lines.append("─── คำแนะนำ ───")
        _append_sa_recommendations(lines, result)
    else:
        lines.append("Shock Absorption: N/A (required markers not found or invalid)")

    return "\n".join(lines)


def get_ms_sagittal_2d_geometry(
    data: C3DData,
    result: MovementStrategyResult,
) -> Dict[str, Any]:
    """
    2D geometry for Movement Strategy (sagittal / side view).
    Returns points in sagittal plane (AP, Vertical) for LM, LFC, GT, SJC and angles α, β.
    """
    out: Dict[str, Any] = {
        "analysis_type": "MS",
        "lm": None,
        "lfc": None,
        "gt": None,
        "sjc": None,
        "line_shank": None,
        "line_thigh": None,
        "line_trunk": None,
        "alpha_deg": None,
        "beta_deg": None,
        "skeleton_2d": None,
        "connections": None,
        "sagittal_y_down": True,  # widget: larger y = bottom of screen so foot (LM) at bottom
    }
    pts_3d = _get_movement_strategy_points(data, result.frame_idx, result.side)
    if pts_3d is None:
        return out
    lm, lfc, gt, sjc = pts_3d
    axis_ml, axis_v = _get_frontal_axes(data)
    axis_ap = 3 - axis_ml - axis_v
    _, _, v_flip = _compute_frontal_orientation(data)
    sag = np.column_stack([
        np.array([lm[axis_ap], lfc[axis_ap], gt[axis_ap], sjc[axis_ap]]),
        v_flip * np.array([lm[axis_v], lfc[axis_v], gt[axis_v], sjc[axis_v]]),
    ])
    # Sagittal 2D: foot (LM) at bottom, head (SJC) at top. We use sagittal_y_down in widget so
    # larger data y = bottom of screen. So we need LM to have larger y than SJC in our data.
    # Flip when head has larger y than foot (sag[3,1] > sag[0,1]) so after flip sag[0,1] > sag[3,1].
    if sag[3, 1] > sag[0, 1]:
        upright_flip = -1
    else:
        upright_flip = 1
    sag[:, 1] = sag[:, 1] * upright_flip
    # Rotate right 90° (clockwise): (x, y) → (y, -x) so view orientation matches correct sagittal
    sag = np.column_stack([sag[:, 1], -sag[:, 0]])
    pts_all = data.get_frame(result.frame_idx).astype(float)
    skel_2d = np.column_stack([
        pts_all[:, axis_ap],
        upright_flip * v_flip * pts_all[:, axis_v],
    ])
    skel_2d = np.column_stack([skel_2d[:, 1], -skel_2d[:, 0]])
    # ให้โมเดลหันหน้าไปทางซ้ายทั้งขาซ้ายและขาขวา: เมื่อเลือกขาขวา (R) สะท้อนแกน x
    if result.side.upper() == "R":
        sag[:, 0] = -sag[:, 0]
        skel_2d[:, 0] = -skel_2d[:, 0]
    out["lm"] = sag[0].copy()
    out["lfc"] = sag[1].copy()
    out["gt"] = sag[2].copy()
    out["sjc"] = sag[3].copy()
    out["line_shank"] = (sag[0].copy(), sag[1].copy())
    out["line_thigh"] = (sag[1].copy(), sag[2].copy())
    out["line_trunk"] = (sag[2].copy(), sag[3].copy())
    out["alpha_deg"] = result.alpha_deg
    out["beta_deg"] = result.beta_deg
    out["skeleton_2d"] = skel_2d
    out["connections"] = data.connections or []
    return out


def get_sa_sagittal_2d_geometry(
    data: C3DData,
    result: MovementStrategyResult,
) -> Dict[str, Any]:
    """
    2D geometry สำหรับ Shock Absorption – แสดงเฉพาะมุม α (Knee flexion): LM, LFC, GT และเส้น shank/thigh.
    ไม่มี trunk / SJC / β.
    """
    out_ms = get_ms_sagittal_2d_geometry(data, result)
    out: Dict[str, Any] = {
        "analysis_type": "SA",
        "lm": out_ms["lm"],
        "lfc": out_ms["lfc"],
        "gt": out_ms["gt"],
        "line_shank": out_ms["line_shank"],
        "line_thigh": out_ms["line_thigh"],
        "alpha_deg": out_ms["alpha_deg"],
        "skeleton_2d": out_ms["skeleton_2d"],
        "connections": out_ms["connections"],
        "sagittal_y_down": True,
        "flip_ml": out_ms.get("flip_ml", False),
    }
    # ไม่ส่ง sjc, line_trunk, beta_deg เพื่อให้ 2D แสดงแค่ α
    return out


def get_ms_overlay_geometry(
    data: C3DData,
    result: MovementStrategyResult,
) -> Dict[str, Any]:
    """
    3D overlay geometry for Movement Strategy: lines LM–LFC, LFC–GT, GT–SJC and points LM, LFC, GT, SJC.
    All points in raw C3D coordinates for viewer to transform.
    """
    out: Dict[str, Any] = {
        "line_shank": None,
        "line_thigh": None,
        "line_trunk": None,
        "lm": None,
        "lfc": None,
        "gt": None,
        "sjc": None,
    }
    pts_3d = _get_movement_strategy_points(data, result.frame_idx, result.side)
    if pts_3d is None:
        return out
    lm, lfc, gt, sjc = pts_3d
    out["line_shank"] = (lm.copy(), lfc.copy())
    out["line_thigh"] = (lfc.copy(), gt.copy())
    out["line_trunk"] = (gt.copy(), sjc.copy())
    out["lm"] = lm.copy()
    out["lfc"] = lfc.copy()
    out["gt"] = gt.copy()
    out["sjc"] = sjc.copy()
    return out


def analyze_sls_limb_stability(
    data: C3DData,
    frame_idx: Optional[int] = None,
    side: str = "L",
    use_max_knee_frame: bool = True,
) -> LimbStabilityResult:
    """
    Run FPKPA analysis for SLS trial (FPKPA only, no GRF/COP).

    Args:
        data: C3D motion data
        frame_idx: Frame to analyze; if None and use_max_knee_frame, uses frame of max knee flexion
        side: "L" or "R" stance leg
        use_max_knee_frame: If True and frame_idx is None, auto-select max knee flexion frame

    Returns:
        LimbStabilityResult with FPKPA score (0–2)
    """
    if frame_idx is None and use_max_knee_frame:
        frame_idx = find_frame_max_knee_flexion(data, side)
    if frame_idx is None:
        frame_idx = 0
    frame_idx = max(0, min(frame_idx, data.n_frames - 1))

    fpkpa_res = analyze_frame_fpkpa(data, frame_idx, side)
    ls = LimbStabilityResult(frame_idx=frame_idx, side=side)

    if fpkpa_res is not None:
        ls.fpkpa = fpkpa_res
        ls.total_ls_score = float(fpkpa_res.score)
    else:
        ls.total_ls_score = 0.0

    return ls


def get_ls_overlay_geometry(
    data: C3DData,
    result: LimbStabilityResult,
) -> Dict[str, Any]:
    """
    Extract 3D geometry for Limb Stability visualization (FPKPA lines, angle arc, COP).
    All points in raw C3D coordinates for viewer to transform.
    """
    out: Dict[str, Any] = {
        "line_thigh": None,
        "line_shank": None,
        "arc_points": None,
        "kjc": None,
        "lfc": None,
        "asis": None,
        "ajc": None,
        "angle_deg": None,
    }
    frame_idx = result.frame_idx
    side = result.side
    indices = _get_point_indices(data, side)
    pts = data.get_frame(frame_idx)
    kjc_i = indices.get("KJC")
    ajc_i = indices.get("AJC")
    asis_i = indices.get("ASIS")
    lfc_i = indices.get("LFC")
    if kjc_i is None or ajc_i is None or asis_i is None:
        return out
    kjc = pts[kjc_i].astype(float)
    ajc = pts[ajc_i].astype(float)
    asis = pts[asis_i].astype(float)
    if not (np.isfinite(kjc).all() and np.isfinite(ajc).all() and np.isfinite(asis).all()):
        return out
    out["line_thigh"] = (asis.copy(), kjc.copy())
    out["line_shank"] = (kjc.copy(), ajc.copy())
    out["kjc"] = kjc.copy()
    out["asis"] = asis.copy()
    out["ajc"] = ajc.copy()
    if result.fpkpa:
        out["angle_deg"] = result.fpkpa.raw_angle
    kjc_2d, ajc_2d, asis_2d, fp_cfg = _project_fpkpa_points(data, frame_idx, kjc, ajc, asis)
    kjc_f, ajc_f, asis_f = kjc_2d, ajc_2d, asis_2d
    v1 = ajc_f - kjc_f
    v2 = asis_f - kjc_f
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 > 1e-8 and n2 > 1e-8:
        v1_u = v1 / n1
        v2_u = v2 / n2
        ang = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        n_arc = max(8, int(np.degrees(ang) / 3))
        t_vals = np.linspace(0, 1, n_arc + 1)
        arc_2d = []
        for t in t_vals:
            theta = (1 - t) * np.arctan2(v2_u[1], v2_u[0]) + t * np.arctan2(v1_u[1], v1_u[0])
            r = 0.12 * min(n1, n2)
            pt = kjc_f + r * np.array([np.cos(theta), np.sin(theta)])
            arc_2d.append(pt)
        arc_2d = np.array(arc_2d)
        if fp_cfg is not None:
            vertical, ml, v_flip = fp_cfg
            ap_dir = np.cross(ml, vertical)
            ap_n = np.linalg.norm(ap_dir)
            if ap_n > 1e-8:
                ap_dir = ap_dir / ap_n
            kjc_ap = np.dot(kjc, ap_dir)
            arc_3d = arc_2d[:, 0:1] * ml + arc_2d[:, 1:2] * (v_flip * vertical) + kjc_ap * ap_dir
        else:
            axis_ml, axis_v = _get_frontal_axes(data)
            axis_ap = 3 - axis_ml - axis_v
            kjc_ap = kjc[axis_ap]
            arc_3d = np.zeros((len(arc_2d), 3))
            arc_3d[:, axis_ml] = arc_2d[:, 0]
            arc_3d[:, axis_v] = arc_2d[:, 1]
            arc_3d[:, axis_ap] = kjc_ap
        out["arc_points"] = arc_3d
    if lfc_i is not None and np.isfinite(pts[lfc_i]).all():
        out["lfc"] = pts[lfc_i].astype(float).copy()
    return out


def get_ls_frontal_2d_geometry(
    data: C3DData,
    result: LimbStabilityResult,
) -> Dict[str, Any]:
    """
    Extract 2D frontal plane geometry used for FPKPA calculation.
    Returns points in the EXACT same 2D projection as compute_fpkpa.
    Includes full skeleton for front-view diagram.
    """
    out: Dict[str, Any] = {
        "analysis_type": "LS",
        "asis": None,
        "kjc": None,
        "ajc": None,
        "lfc": None,
        "arc_points": None,
        "angle_deg": None,
        "axis_ml": 0,
        "axis_v": 2,
        "flip_ml": True,  # frontal view: person facing us → their right = our left
        "skeleton_2d": None,
        "connections": None,
        "line_thigh_ext": None,
        "line_shank_ext": None,
    }
    frame_idx = result.frame_idx
    side = result.side
    indices = _get_point_indices(data, side)
    pts = data.get_frame(frame_idx)
    kjc_i = indices.get("KJC")
    ajc_i = indices.get("AJC")
    asis_i = indices.get("ASIS")
    lfc_i = indices.get("LFC")
    if kjc_i is None or ajc_i is None or asis_i is None:
        return out
    kjc = pts[kjc_i].astype(float)
    ajc = pts[ajc_i].astype(float)
    asis = pts[asis_i].astype(float)
    if not (np.isfinite(kjc).all() and np.isfinite(ajc).all() and np.isfinite(asis).all()):
        return out
    kjc_2d, ajc_2d, asis_2d, fp_cfg = _project_fpkpa_points(data, frame_idx, kjc, ajc, asis)
    pts_all = data.get_frame(frame_idx).astype(float)
    if fp_cfg is not None:
        vertical, ml, v_flip = fp_cfg
        skel_2d = _project_to_frontal_from_vectors(pts_all, vertical, ml, v_flip)
        out["axis_ml"], out["axis_v"] = 0, 1  # ML=0, V=1 in our 2D output
    else:
        ax_ml, ax_v, v_flip = _compute_frontal_orientation(data)
        out["axis_ml"], out["axis_v"] = ax_ml, ax_v
        skel_2d = np.column_stack([
            pts_all[:, ax_ml],
            v_flip * pts_all[:, ax_v],
        ])
    out["skeleton_2d"] = skel_2d
    out["connections"] = data.connections or []
    fr = np.array([kjc_2d, ajc_2d, asis_2d])
    out["asis"] = fr[2].copy()
    out["kjc"] = fr[0].copy()
    out["ajc"] = fr[1].copy()
    if result.fpkpa:
        out["angle_deg"] = result.fpkpa.angle_deg  # deviation (for 2D display)
        out["angle_interpretation"] = result.fpkpa.interpretation
    v1 = fr[1] - fr[0]
    v2 = fr[2] - fr[0]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 > 1e-8 and n2 > 1e-8:
        v1_u = v1 / n1
        v2_u = v2 / n2
        ang = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        n_arc = max(8, int(np.degrees(ang) / 3))
        t_vals = np.linspace(0, 1, n_arc + 1)
        r = 0.12 * min(n1, n2)
        arc = []
        for t in t_vals:
            theta = (1 - t) * np.arctan2(v2_u[1], v2_u[0]) + t * np.arctan2(v1_u[1], v1_u[0])
            pt = fr[0] + r * np.array([np.cos(theta), np.sin(theta)])
            arc.append(pt)
        out["arc_points"] = np.array(arc)
    ext_shank = 1.2
    out["line_thigh_ext"] = (fr[2].copy(), fr[0].copy())
    out["line_shank_ext"] = (fr[1].copy(), fr[0] + ext_shank * (fr[0] - fr[1]))
    if lfc_i is not None and np.isfinite(pts[lfc_i]).all():
        lfc3 = pts[lfc_i]
        if fp_cfg is not None:
            out["lfc"] = _project_to_frontal_from_vectors(lfc3.reshape(1, 3), vertical, ml, v_flip).ravel()
        else:
            out["lfc"] = np.array([lfc3[ax_ml], v_flip * lfc3[ax_v]])
    return out


def _box_results(width: int, header: str, result_lines: List[str]) -> List[str]:
    """Wrap result lines in a Unicode box for visual emphasis."""
    w = max(width, len(header) + 2, max((len(ln) + 2 for ln in result_lines), default=0))
    def pad(s: str) -> str:
        return s + " " * max(0, w - 2 - len(s))
    out = [
        "╔" + "═" * (w - 2) + "╗",
        "║ " + pad(header) + "║",
        "╠" + "═" * (w - 2) + "╣",
    ]
    for ln in result_lines:
        out.append("║ " + pad(ln) + "║")
    out.append("╚" + "═" * (w - 2) + "╝")
    return out


def _append_ls_recommendations(lines: List[str], r: "FPKPAResult", side: str) -> None:
    """Add Limb Stability recommendations – expert-level, detailed."""
    interp = (r.interpretation or "").lower()
    if r.score == 2:
        lines.append("  [สรุป] การจัดเรียงแนวขาในเกณฑ์ adequate – ควบคุม frontal plane ได้ดี")
        lines.append("")
        lines.append("  • รักษาสถานะปัจจุบัน: ฝึก neuromuscular control เป็นประจำ (2–3 ครั้ง/สัปดาห์)")
        lines.append("  • ท่าที่แนะนำ: single-leg stance, mini-squat, step-down โดยเน้นรักษาแนวเข่า–ข้อเท้า")
        lines.append("  • การป้องกัน: พิจารณาโปรแกรม ACL injury prevention (NMT) เพื่อลดความเสี่ยงในอนาคต")
    elif r.score == 1:
        lines.append("  [สรุป] มี deviation ปานกลาง (10–25°) – ควรปรับปรุงก่อนนำไปใช้กิจกรรมที่มีความเสี่ยง")
        lines.append("")
        if "valgus" in interp:
            lines.append("  ■ การตีความเชิงคลินิก")
            lines.append("    Valgus (knee-in) สัมพันธ์กับ hip internal rotation + adduction มากเกินไป")
            lines.append("    อาจมาจาก: Gluteus medius อ่อนแรง, femoral internal rotation, หรืองอ trunk มาก")
            lines.append("    ความเสี่ยง: เพิ่มภาระที่ ACL, MCL และ patellofemoral joint")
            lines.append("")
            lines.append("  ■ การประเมินเพิ่มเติมที่แนะนำ")
            lines.append("    • วัด hip external rotator strength (resisted external rotation ในท่านั่ง/นอน)")
            lines.append("    • วัด hip abductor strength (side-lying abduction, single-leg squat endurance)")
            lines.append("    • ตรวจ trunk control (ตัวเอียง, rotation ใน single-leg)")
            lines.append("    • พิจารณา foot posture (pronation มากอาจส่งผลต่อ tibial rotation)")
            lines.append("")
            lines.append("  ■ โปรแกรมการฝึกที่แนะนำ (ตามหลักฐานเชิงประจักษ์)")
            lines.append("    • Phase 1 – เสริมความแข็งแรง: Clam shell, side-lying hip abduction,")
            lines.append("      glute bridge, single-leg bridge เน้นเกร็ง gluteus ก่อนเคลื่อนไหว")
            lines.append("    • Phase 2 – ควบคุมการเคลื่อนไหว: Step-down (ควบคุมเข่าไม่ให้หุบเข้า),")
            lines.append("      single-leg squat ค่อยๆ เพิ่มความลึก พร้อม feedback (กระจก/วิดีโอ)")
            lines.append("    • Phase 3 – ทักษะ: Lateral step, single-leg hop, landing โดยเน้น soft landing")
            lines.append("    • Cue ที่ใช้: \"เข่าเหยียดออกตามนิ้วเท้า\" \"บีบก้นก่อนย่อ\" \"อกยก หลังตรง\"")
            lines.append("")
            lines.append("  ■ ความถี่และระยะเวลา: 2–4 ครั้ง/สัปดาห์ นาน 6–12 สัปดาห์ แล้วประเมินซ้ำ")
        elif "varus" in interp:
            lines.append("  ■ การตีความเชิงคลินิก")
            lines.append("    Varus (knee-out) พบน้อยกว่า valgus ใน SLS มักสัมพันธ์กับ hip abduction มาก")
            lines.append("")
            lines.append("  ■ โปรแกรมการฝึก")
            lines.append("    • เสริมความแข็งแรง hip adductors (Copenhagen adduction, adductor machine)")
            lines.append("    • ฝึก single-leg control พร้อม cue ให้เข่าไม่หุบออก")
            lines.append("    • พิจารณาประเมิน IT band, TFL ตึง และ hip internal rotation range")
        else:
            lines.append("  ■ แนะนำเสริมความแข็งแรงและ neuromuscular control รอบสะโพก–เข่า")
            lines.append("    • ฝึกท่าที่มี single-leg loading พร้อม real-time feedback (กระจก)")
            lines.append("    • ประเมิน hip strength และ trunk control ร่วม")
    else:
        lines.append("  [สรุป] Deviation สูง (>25°) – เสี่ยงต่อการบาดเจ็บ ควรส่งต่อประเมินลึก")
        lines.append("")
        if "valgus" in interp:
            lines.append("  ■ การส่งต่อและการประเมินลึก")
            lines.append("    • วัด hip strength (isokinetic/HHD): ER, abduction, extension")
            lines.append("    • ตรวจ Q-angle, foot alignment (navicular drop, FPI)")
            lines.append("    • ประเมิน movement pattern ทั้ง chain (ankle → knee → hip → trunk)")
            lines.append("")
            lines.append("  ■ โปรแกรมรักษา (ต้องดูแลโดยผู้เชี่ยวชาญ)")
            lines.append("    • Neuromuscular training (NMT) แบบเข้มข้น: cueing, strengthening,")
            lines.append("      plyometrics, agility ร่วมกัน")
            lines.append("    • พิจารณา patellar taping, knee sleeve หรือ foot orthotic ถ้ามี contributing factors")
            lines.append("    • หลีกเลี่ยงกิจกรรม high-load จนกว่าจะได้คะแนน ≥ 1/2 และทดสอบ functional pass")
        elif "varus" in interp:
            lines.append("  ■ แนะนำส่งต่อประเมินโครงสร้างและความแข็งแรงรอบข้อต่อ")
            lines.append("    • โปรแกรม: hip adductor strengthening, movement pattern retraining")
        else:
            lines.append("  ■ แนะนำส่งต่อผู้เชี่ยวชาญ (PT/orthopedic) เพื่อประเมินสาเหตุและวางแผนรักษา")


def format_limb_stability_report(
    result: LimbStabilityResult,
    data: Optional[C3DData] = None,
) -> str:
    """Format LimbStabilityResult as readable report (FPKPA only)."""
    lines = [
        "Limb Stability (FPKPA)",
        "",
        f"Stance leg: {result.side}   │   Frame: {result.frame_idx}",
        "",
    ]
    if data is not None:
        idx = _get_point_indices(data, result.side)
        asis_i, kjc_i, ajc_i = idx.get("ASIS"), idx.get("KJC"), idx.get("AJC")
        if asis_i is not None and kjc_i is not None and ajc_i is not None:
            lines.append(f"Markers: ASIS={asis_i}  KJC={kjc_i}  AJC={ajc_i}")
            lines.append("")
    if result.fpkpa:
        r = result.fpkpa
        raw_display = -abs(r.raw_angle) if r.interpretation == "varus" else r.raw_angle
        dev_display = -r.angle_deg if r.interpretation == "varus" else r.angle_deg
        result_lines = [
            f"Raw angle:      {raw_display:+.4f}\u00B0  ({r.interpretation})",
            f"Deviation:      {dev_display:+.4f}\u00B0  from straight",
            f"Score:          {r.score}/2  {r.score_note or r.interpretation}",
        ]
        lines.extend(_box_results(54, "■ ผลการวิเคราะห์ FPKPA", result_lines))
        lines.append("")
        lines.append("─── สรุปความหมายผล ───")
        summary_parts = []
        dev_show = -r.angle_deg if r.interpretation == "varus" else r.angle_deg
        summary_parts.append(f"เบี่ยง {dev_show:.1f}° จากแนวตรง")
        if r.interpretation == "valgus":
            summary_parts.append("knee-in (valgus)")
        elif r.interpretation == "varus":
            summary_parts.append("knee-out (varus)")
        if r.score == 2:
            summary_parts.append("อยู่ในเกณฑ์")
        elif r.score == 1:
            summary_parts.append("ควรปรับ")
        lines.append("  " + "  ·  ".join(summary_parts))
        lines.append("")
        lines.append("─── นิยามมุม ───")
        lines.append("  FPKPA = มุมระนาบด้าน (frontal) ระหว่าง KJC→AJC กับ ASIS→KJC")
        lines.append("  Valgus (+): knee-in  ·  Varus (-): knee-out")
        lines.append("")
        lines.append("─── คำแนะนำ ───")
        _append_ls_recommendations(lines, r, result.side)
    else:
        lines.append("FPKPA: N/A (required markers not found)")
    return "\n".join(lines)


# --- Pelvis Stability (PS) - Di Paolo et al. The Knee 2024 ---


def compute_pelvis_tilt_angle(
    lasis: np.ndarray,
    rasis: np.ndarray,
    axis_vertical: int = 2,
    axis_ml: int = 0,
) -> float:
    """
    Pelvis tilt angle (PA) in frontal plane.
    Angle of the line LASIS–RASIS from horizontal.
    Positive = contralateral pelvic drop (per paper convention).
    Returns angle in degrees (signed).
    """
    pts = np.array([lasis, rasis], dtype=float)
    if np.any(~np.isfinite(pts)):
        return float("nan")
    fr = _to_frontal_plane(pts, axis_vertical, axis_ml)
    vec = fr[1] - fr[0]  # RASIS - LASIS in (ML, Vertical)
    if np.linalg.norm(vec) < 1e-8:
        return 0.0
    # Angle from horizontal (ML axis): atan2(vertical, ml)
    # Positive = right ASIS higher (contralateral drop when stance=L)
    return float(np.degrees(np.arctan2(vec[1], vec[0])))


def analyze_pelvis_stability(
    data: C3DData,
    frame_idx: int,
    side: str = "L",
) -> PelvisStabilityResult:
    """
    Compute Pelvis Stability (PS) score from Pelvis Tilt Angle (PA).
    PA = angle of LASIS–RASIS line from horizontal in frontal plane.
    Score: 0/2 if |PA| > 25°, 1/2 if 10° ≤ |PA| ≤ 25°, 2/2 if |PA| < 10°.
    """
    out = PelvisStabilityResult(frame_idx=frame_idx, side=side)
    pts = data.get_frame(frame_idx)
    idx_l = _get_point_indices(data, "L")
    idx_r = _get_point_indices(data, "R")
    lasis_i = idx_l.get("ASIS")
    rasis_i = idx_r.get("ASIS")
    if lasis_i is None or rasis_i is None:
        return out
    lasis = pts[lasis_i]
    rasis = pts[rasis_i]
    if np.any(~np.isfinite(lasis)) or np.any(~np.isfinite(rasis)):
        return out
    fp_cfg = _get_frontal_projection_vectors(data, frame_idx)
    if fp_cfg is not None:
        vertical, ml, v_flip = fp_cfg
        fr = _project_to_frontal_from_vectors(
            np.array([lasis, rasis]), vertical, ml, v_flip
        )
    else:
        ax_ml, ax_v, v_flip = _compute_frontal_orientation(data)
        fr = np.column_stack([
            np.array([lasis[ax_ml], rasis[ax_ml]]),
            v_flip * np.array([lasis[ax_v], rasis[ax_v]]),
        ])
    lasis_2d = fr[0]
    rasis_2d = fr[1]
    vec = rasis_2d - lasis_2d
    if np.linalg.norm(vec) < 1e-8:
        pa = 0.0
    else:
        pa = float(np.degrees(np.arctan2(vec[1], vec[0])))
    out.pa_deg = pa
    out.lasis_2d = lasis_2d
    out.rasis_2d = rasis_2d
    dev = abs(pa)
    score, note = score_fpkpa(dev)  # same thresholds: >25→0, 10-25→1, <10→2
    out.score = score
    out.score_note = note
    return out


def get_ps_frontal_2d_geometry(
    data: C3DData,
    result: PelvisStabilityResult,
) -> Dict[str, Any]:
    """
    Extract 2D frontal plane geometry for Pelvis Stability.
    Draws LASIS, RASIS, pelvis line, horizontal reference, angle.
    """
    out: Dict[str, Any] = {
        "analysis_type": "PS",
        "lasis": None,
        "rasis": None,
        "pelvis_line": None,
        "horizontal_ref": None,
        "angle_deg": None,
        "axis_ml": 0,
        "axis_v": 2,
        "flip_ml": True,
        "skeleton_2d": None,
        "connections": None,
    }
    if result.lasis_2d is None or result.rasis_2d is None:
        return out
    lasis_2d = result.lasis_2d
    rasis_2d = result.rasis_2d
    out["lasis"] = lasis_2d.copy()
    out["rasis"] = rasis_2d.copy()
    out["pelvis_line"] = (lasis_2d.copy(), rasis_2d.copy())
    mid = (lasis_2d + rasis_2d) / 2
    span = np.linalg.norm(rasis_2d - lasis_2d)
    if span < 1e-8:
        span = 0.1
    h_len = span * 1.5
    out["horizontal_ref"] = (
        mid + np.array([-h_len / 2, 0]),
        mid + np.array([h_len / 2, 0]),
    )
    out["angle_deg"] = result.pa_deg
    pts_all = data.get_frame(result.frame_idx).astype(float)
    fp_cfg = _get_frontal_projection_vectors(data, result.frame_idx)
    if fp_cfg is not None:
        vertical, ml, v_flip = fp_cfg
        out["skeleton_2d"] = _project_to_frontal_from_vectors(
            pts_all, vertical, ml, v_flip
        )
    else:
        ax_ml, ax_v, v_flip = _compute_frontal_orientation(data)
        out["skeleton_2d"] = np.column_stack([
            pts_all[:, ax_ml],
            v_flip * pts_all[:, ax_v],
        ])
    out["connections"] = data.connections or []
    return out


def get_ps_overlay_geometry(
    data: C3DData,
    result: PelvisStabilityResult,
) -> Dict[str, Any]:
    """
    3D geometry for Pelvis Stability overlay: LASIS-RASIS line + แกนแนวนอน (horizontal ref) ที่ใช้คำนวณ PA.
    """
    out: Dict[str, Any] = {"line_pelvis": None, "lasis": None, "rasis": None, "horizontal_ref_3d": None}
    idx_l = _get_point_indices(data, "L")
    idx_r = _get_point_indices(data, "R")
    lasis_i = idx_l.get("ASIS") if idx_l else None
    rasis_i = idx_r.get("ASIS") if idx_r else None
    if lasis_i is None or rasis_i is None:
        return out
    pts = data.get_frame(result.frame_idx)
    lasis = pts[lasis_i].astype(float)
    rasis = pts[rasis_i].astype(float)
    if not (np.isfinite(lasis).all() and np.isfinite(rasis).all()):
        return out
    out["line_pelvis"] = (lasis.copy(), rasis.copy())
    out["lasis"] = lasis.copy()
    out["rasis"] = rasis.copy()
    mid = (lasis + rasis) / 2
    span = np.linalg.norm(rasis - lasis)
    if span < 1e-8:
        span = 0.1
    h_len = span * 1.5
    fp_cfg = _get_frontal_projection_vectors(data, result.frame_idx)
    if fp_cfg is not None:
        vertical, ml, v_flip = fp_cfg
        out["horizontal_ref_3d"] = (mid - h_len * ml, mid + h_len * ml)
    else:
        ax_ml, ax_v = _get_frontal_axes(data)
        ml_vec = np.zeros(3)
        ml_vec[ax_ml] = 1.0
        out["horizontal_ref_3d"] = (mid - h_len * ml_vec, mid + h_len * ml_vec)
    return out


def _append_ps_recommendations(lines: List[str], result: PelvisStabilityResult) -> None:
    """Add Pelvis Stability recommendations – expert-level."""
    pa = result.pa_deg or 0
    score = result.score or 0
    side = (result.side or "L").upper()
    drop_dir = "contralateral" if (pa > 0 and side == "L") or (pa < 0 and side == "R") else "ipsilateral"
    if score == 2:
        lines.append("  [สรุป] การควบคุมเชิงกรานในเกณฑ์ adequate – hip abductor control ดี")
        lines.append("")
        lines.append("  • รักษาสถานะปัจจุบัน: ฝึก single-leg stance, step-down โดยรักษาระดับเชิงกราน")
        lines.append("  • การป้องกัน: พิจารณาโปรแกรม hip strengthening เป็นประจำเพื่อคงความสามารถ")
    elif score == 1:
        lines.append("  [สรุป] มี pelvic tilt ปานกลาง (10–25°) – ควรเสริมความแข็งแรงก่อนกิจกรรมที่มีความเสี่ยง")
        lines.append("")
        lines.append("  ■ การตีความเชิงคลินิก")
        lines.append(f"    Pelvic drop ด้าน {drop_dir} สัมพันธ์กับ hip abductor (gluteus medius) อ่อนแรงหรือ")
        lines.append("    ทำงานล่าช้า (delayed onset) ใน stance phase")
        lines.append("    ความเสี่ยง: Trendelenburg, ภาระเพิ่มที่เข่าและหลังส่วนล่าง")
        lines.append("")
        lines.append("  ■ การประเมินเพิ่มเติมที่แนะนำ")
        lines.append("    • วัด hip abductor strength: side-lying abduction, single-leg squat endurance")
        lines.append("    • ทดสอบ Trendelenburg sign, Single-leg squat test")
        lines.append("    • ตรวจ trunk lateral flexor control และ foot posture ร่วม")
        lines.append("")
        lines.append("  ■ โปรแกรมการฝึกที่แนะนำ")
        lines.append("    • Phase 1: Side-lying hip abduction, clam shell, single-leg bridge เน้นเกร็ง gluteus")
        lines.append("    • Phase 2: Single-leg stance (รักษาระดับเชิงกราน), step-down ควบคุม pelvic alignment")
        lines.append("    • Phase 3: Step-up, lateral step, single-leg squat พร้อม feedback")
        lines.append("    • Cue: \"สะโพกตรง\" \"ยกเชิงกรานขึ้น\" \"ไม่เอียงตัว\"")
        lines.append("")
        lines.append("  ■ ความถี่: 2–4 ครั้ง/สัปดาห์ นาน 6–12 สัปดาห์ แล้วประเมินซ้ำ")
    else:
        lines.append("  [สรุป] Pelvic tilt สูง (>25°) – เสี่ยงต่อการชดเชยและบาดเจ็บ ควรส่งต่อประเมินลึก")
        lines.append("")
        lines.append("  ■ การส่งต่อและการประเมินลึก")
        lines.append("    • วัด hip strength (abduction, ER) – HHD หรือ isokinetic")
        lines.append("    • ประเมิน trunk control, lumbar stability")
        lines.append("    • พิจารณา leg length discrepancy, scoliosis ถ้ามี")
        lines.append("")
        lines.append("  ■ โปรแกรมรักษา (ต้องดูแลโดยผู้เชี่ยวชาญ)")
        lines.append("    • Hip abductor strengthening แบบเข้มข้น")
        lines.append("    • Pelvic control training, gait retraining")
        lines.append("    • หลีกเลี่ยงกิจกรรม high-load จนกว่าจะได้คะแนน ≥ 1/2")


def format_pelvis_stability_report(
    result: PelvisStabilityResult,
    data: Optional[C3DData] = None,
) -> str:
    """Format PelvisStabilityResult as readable report."""
    lines = [
        "Pelvis Stability (PA)",
        "",
        f"Stance leg: {result.side}   │   Frame: {result.frame_idx}",
        "",
    ]
    if data is not None:
        idx_l = _get_point_indices(data, "L")
        idx_r = _get_point_indices(data, "R")
        lasis_i = idx_l.get("ASIS") if idx_l else None
        rasis_i = idx_r.get("ASIS") if idx_r else None
        if lasis_i is not None and rasis_i is not None:
            lines.append(f"Markers: LASIS={lasis_i}  RASIS={rasis_i}")
            lines.append("")
    if result.pa_deg is not None:
        result_lines = [
            f"PA:            {result.pa_deg:+.4f}\u00B0",
            f"Score:         {result.score}/2  {result.score_note}",
        ]
        lines.extend(_box_results(54, "■ ผลการวิเคราะห์ PA", result_lines))
        lines.append("")
        pa = result.pa_deg
        score = result.score or 0
        side = (result.side or "L").upper()
        drop_dir = "contralateral" if (pa > 0 and side == "L") or (pa < 0 and side == "R") else "ipsilateral"
        lines.append("─── สรุปความหมายผล ───")
        summary_parts = [f"PA = {pa:+.1f}°", f"เชิงกรานเอียง ({drop_dir} drop)"]
        if score == 2:
            summary_parts.append("อยู่ในเกณฑ์")
        elif score == 1:
            summary_parts.append("ควรเสริมความแข็งแรง")
        lines.append("  " + "  ·  ".join(summary_parts))
        lines.append("")
        lines.append("─── นิยามมุม ───")
        lines.append("  PA = มุมเส้น LASIS–RASIS จากแนวนอน (frontal plane)")
        lines.append("  PA+ = contralateral drop  ·  PA- = ipsilateral drop  ·  เกี่ยวกับ gluteus medius")
        lines.append("")
        lines.append("─── คำแนะนำ ───")
        _append_ps_recommendations(lines, result)
    else:
        lines.append("PA: N/A (required markers not found)")
    return "\n".join(lines)


# --- Trunk Stability (TS) - Di Paolo et al. The Knee 2024 ---
# TA = angle between (clavicular notch to midline pelvis line) and vertical neutral reference.
# Vertical = แกนแนวตั้งของโลก (gravity จาก force plate). TA วัดในระนาบ frontal เท่านั้น:
#   project trunk vector ลง frontal plane (ML–Vertical) แล้ววัดมุมกับแนวตั้ง → ตรงกับ BTS Smart Analyzer / หลัก clinical.
# ถ้าวัดมุมใน 3D จะได้ค่าสูงกว่า (รวมเอียงหน้า-หลังด้วย).
# Clavicular notch = midpoint of L and R shoulder. Score: 0/2 if >10°, 1/2 if 5–10°, 2/2 if <5°.


def analyze_trunk_stability(
    data: C3DData,
    frame_idx: int,
    side: str = "L",
) -> TrunkStabilityResult:
    """
    TA = angle between trunk segment and vertical.
    Trunk segment = midline pelvis to clavicular notch (clavicular = midpoint of L/R shoulder).
    Score: 0/2 if |TA|>10°, 1/2 if 5–10°, 2/2 if <5° (either OL or CL trunk tilt).
    """
    out = TrunkStabilityResult(frame_idx=frame_idx, side=side)
    pts = data.get_frame(frame_idx)
    idx = _get_trunk_indices(data)
    lasis_i = idx.get("LASIS")
    rasis_i = idx.get("RASIS")
    lsho_i = idx.get("LSHOULDER")
    rsho_i = idx.get("RSHOULDER")
    if lasis_i is None or rasis_i is None or lsho_i is None or rsho_i is None:
        return out
    lasis = pts[lasis_i]
    rasis = pts[rasis_i]
    lsho = pts[lsho_i]
    rsho = pts[rsho_i]
    if np.any(~np.isfinite(lasis)) or np.any(~np.isfinite(rasis)) or np.any(~np.isfinite(lsho)) or np.any(~np.isfinite(rsho)):
        return out
    clav_pt = (lsho + rsho) / 2
    midline_pelvis = (lasis + rasis) / 2
    world_cfg = _get_world_vertical_from_force_plate(data, frame_idx)
    if world_cfg is not None:
        vertical_3d, ml_3d, v_flip = world_cfg
        world_up = vertical_3d if v_flip > 0 else -vertical_3d
        trunk_vec_3d = clav_pt - midline_pelvis
        ml_c = np.dot(trunk_vec_3d, ml_3d)
        v_c = np.dot(trunk_vec_3d, world_up)
        in_plane_len = np.sqrt(ml_c * ml_c + v_c * v_c)
        if in_plane_len < 1e-8:
            ta = 0.0
        else:
            cos_frontal = np.clip(v_c / in_plane_len, -1.0, 1.0)
            angle_from_vert_rad = np.arccos(cos_frontal)
            ta_abs = float(np.degrees(angle_from_vert_rad))
            ta = np.sign(ml_c) * ta_abs
        clav_2d = _project_to_frontal_from_vectors(clav_pt.reshape(1, 3), vertical_3d, ml_3d, v_flip).ravel()
        mid_2d = _project_to_frontal_from_vectors(midline_pelvis.reshape(1, 3), vertical_3d, ml_3d, v_flip).ravel()
        out.used_world_vertical = True
    else:
        fp_cfg = _get_frontal_projection_vectors(data, frame_idx)
        if fp_cfg is not None:
            vertical, ml, v_flip = fp_cfg
            clav_2d = _project_to_frontal_from_vectors(clav_pt.reshape(1, 3), vertical, ml, v_flip).ravel()
            mid_2d = _project_to_frontal_from_vectors(midline_pelvis.reshape(1, 3), vertical, ml, v_flip).ravel()
        else:
            ax_ml, ax_v, v_flip = _compute_frontal_orientation(data)
            clav_2d = np.array([clav_pt[ax_ml], v_flip * clav_pt[ax_v]])
            mid_2d = np.array([midline_pelvis[ax_ml], v_flip * midline_pelvis[ax_v]])
        vec = clav_2d - mid_2d
        if np.linalg.norm(vec) < 1e-8:
            ta = 0.0
        else:
            cos_vert = np.clip(vec[1] / np.linalg.norm(vec), -1.0, 1.0)
            angle_from_vert_rad = np.arccos(cos_vert)
            ta_abs = float(np.degrees(angle_from_vert_rad))
            ta = np.sign(vec[0]) * ta_abs
    out.ta_deg = ta
    out.clavicular_2d = clav_2d
    out.midline_pelvis_2d = mid_2d
    dev = abs(ta)
    score, note = score_trunk_angle(dev)
    out.score = score
    out.score_note = note
    return out


def get_ts_frontal_2d_geometry(
    data: C3DData,
    result: TrunkStabilityResult,
) -> Dict[str, Any]:
    """2D geometry for Trunk Stability: trunk segment (midline pelvis → clavicular) and vertical reference."""
    out: Dict[str, Any] = {
        "analysis_type": "TS",
        "lasis": None,
        "rasis": None,
        "pelvis_line": None,
        "horizontal_ref": None,
        "angle_deg": None,
        "axis_ml": 0,
        "axis_v": 2,
        "flip_ml": True,
        "skeleton_2d": None,
        "connections": None,
    }
    if result.midline_pelvis_2d is None or result.clavicular_2d is None:
        return out
    mid_2d = result.midline_pelvis_2d
    clav_2d = result.clavicular_2d
    out["lasis"] = mid_2d.copy()
    out["rasis"] = clav_2d.copy()
    out["pelvis_line"] = (mid_2d.copy(), clav_2d.copy())  # trunk segment
    span = np.linalg.norm(clav_2d - mid_2d)
    if span < 1e-8:
        span = 0.1
    v_len = span * 1.2
    out["horizontal_ref"] = (
        mid_2d + np.array([0, -v_len / 2]),
        mid_2d + np.array([0, v_len / 2]),
    )
    out["vertical_ref"] = out["horizontal_ref"]
    out["vertical_ref_label"] = "vertical (gravity)"  # แนวตั้งของโลก สำหรับ 2D
    out["angle_deg"] = result.ta_deg
    pts_all = data.get_frame(result.frame_idx).astype(float)
    fp_cfg = _get_frontal_projection_vectors(data, result.frame_idx)
    if fp_cfg is not None:
        vertical, ml, v_flip = fp_cfg
        out["skeleton_2d"] = _project_to_frontal_from_vectors(
            pts_all, vertical, ml, v_flip
        )
    else:
        ax_ml, ax_v, v_flip = _compute_frontal_orientation(data)
        out["skeleton_2d"] = np.column_stack([
            pts_all[:, ax_ml],
            v_flip * pts_all[:, ax_v],
        ])
    out["connections"] = data.connections or []
    return out


def get_ts_overlay_geometry(
    data: C3DData,
    result: TrunkStabilityResult,
) -> Dict[str, Any]:
    """
    3D geometry for Trunk Stability overlay: trunk segment (clavicular → midline pelvis) + แกนแนวตั้ง (vertical/gravity ref).
    """
    out: Dict[str, Any] = {"line_trunk": None, "clavicular": None, "midline_pelvis": None, "vertical_ref_3d": None}
    idx = _get_trunk_indices(data)
    lasis_i = idx.get("LASIS")
    rasis_i = idx.get("RASIS")
    lsho_i = idx.get("LSHOULDER")
    rsho_i = idx.get("RSHOULDER")
    if lasis_i is None or rasis_i is None or lsho_i is None or rsho_i is None:
        return out
    pts = data.get_frame(result.frame_idx)
    lasis = pts[lasis_i].astype(float)
    rasis = pts[rasis_i].astype(float)
    lsho = pts[lsho_i].astype(float)
    rsho = pts[rsho_i].astype(float)
    if not (np.isfinite(lasis).all() and np.isfinite(rasis).all() and np.isfinite(lsho).all() and np.isfinite(rsho).all()):
        return out
    midline_pelvis = (lasis + rasis) / 2
    clavicular = (lsho + rsho) / 2
    trunk_len = np.linalg.norm(clavicular - midline_pelvis)
    if trunk_len < 1e-8:
        trunk_len = 0.1
    v_half = trunk_len * 0.7
    world_cfg = _get_world_vertical_from_force_plate(data, result.frame_idx)
    if world_cfg is not None:
        vertical_3d, ml_3d, v_flip = world_cfg
        world_up = vertical_3d if v_flip > 0 else -vertical_3d
        base = midline_pelvis.copy()
        out["vertical_ref_3d"] = (base - world_up * v_half, base + world_up * v_half)
    else:
        fp_cfg = _get_frontal_projection_vectors(data, result.frame_idx)
        if fp_cfg is not None:
            vertical, ml, v_flip = fp_cfg
            world_up = vertical if v_flip > 0 else -vertical
            base = midline_pelvis.copy()
            out["vertical_ref_3d"] = (base - world_up * v_half, base + world_up * v_half)
        else:
            ax_ml, ax_v = _get_frontal_axes(data)
            up_vec = np.zeros(3)
            up_vec[ax_v] = 1.0
            base = midline_pelvis.copy()
            out["vertical_ref_3d"] = (base - up_vec * v_half, base + up_vec * v_half)
    out["line_trunk"] = (midline_pelvis.copy(), clavicular.copy())
    out["clavicular"] = clavicular.copy()
    out["midline_pelvis"] = midline_pelvis.copy()
    return out


def _append_ts_recommendations(lines: List[str], result: TrunkStabilityResult) -> None:
    """Add Trunk Stability recommendations – expert-level."""
    ta = result.ta_deg or 0
    score = result.score or 0
    tilt_dir = "ขวา" if ta > 0 else "ซ้าย"
    if score == 2:
        lines.append("  [สรุป] การควบคุมลำตัวในเกณฑ์ adequate – trunk lateral control ดี")
        lines.append("")
        lines.append("  • รักษาสถานะปัจจุบัน: ฝึก core stability, single-leg stance โดยรักษาแนวลำตัว")
        lines.append("  • การป้องกัน: พิจารณาโปรแกรม trunk control เป็นประจำ")
    elif score == 1:
        lines.append("  [สรุป] มี trunk tilt ปานกลาง (5–10°) – อาจเป็น compensation กับ pelvic drop")
        lines.append("")
        lines.append("  ■ การตีความเชิงคลินิก")
        lines.append(f"    ลำตัวเอียง{tilt_dir} สัมพันธ์กับ lateral trunk flexion อาจเป็น compensation")
        lines.append("    เพื่อรักษา center of mass เหนือฐานรองรับ (compensatory strategy)")
        lines.append("    หรือมาจาก core/oblique อ่อนแรง")
        lines.append("")
        lines.append("  ■ การประเมินเพิ่มเติมที่แนะนำ")
        lines.append("    • ตรวจ pelvic stability (PA) ร่วม – มักมี pelvic drop ก่อน trunk lean")
        lines.append("    • วัด trunk lateral flexor, oblique strength")
        lines.append("    • ประเมิน hip abductor strength (มักเป็นสาเหตุต้นทาง)")
        lines.append("")
        lines.append("  ■ โปรแกรมการฝึกที่แนะนำ")
        lines.append("    • แก้ต้นทางก่อน: hip abductor strengthening (ลด pelvic drop)")
        lines.append("    • Core: side plank, bird dog, dead bug, anti-lateral flexion")
        lines.append("    • ฝึก single-leg stance พร้อม cue \"อกยก หลังตรง ไม่เอียงตัว\"")
        lines.append("")
        lines.append("  ■ ความถี่: 2–4 ครั้ง/สัปดาห์ ร่วมกับ hip strengthening")
    else:
        lines.append("  [สรุป] Trunk tilt สูง (>10°) – compensation มาก เสี่ยงต่อการบาดเจ็บหลัง")
        lines.append("")
        lines.append("  ■ การส่งต่อและการประเมินลึก")
        lines.append("    • ประเมิน pelvic stability, hip strength ร่วม (มักเป็น chain)")
        lines.append("    • ตรวจ core endurance, lumbar stability")
        lines.append("    • พิจารณา scoliosis, leg length discrepancy")
        lines.append("")
        lines.append("  ■ โปรแกรมรักษา (ต้องดูแลโดยผู้เชี่ยวชาญ)")
        lines.append("    • แก้ hip abductor ก่อน แล้วจึง trunk control")
        lines.append("    • Trunk lateral control training, movement pattern retraining")
        lines.append("    • หลีกเลี่ยงกิจกรรมที่ต้องทรงตัวบนขาเดียวจนกว่าจะได้คะแนน ≥ 1/2")


def format_trunk_stability_report(
    result: TrunkStabilityResult,
    data: Optional[C3DData] = None,
) -> str:
    """Format TrunkStabilityResult as readable report."""
    lines = [
        "Trunk Stability (TA)",
        "",
        f"Stance leg: {result.side}   │   Frame: {result.frame_idx}",
        "",
    ]
    if data is not None:
        idx = _get_trunk_indices(data)
        lsho_i = idx.get("LSHOULDER")
        rsho_i = idx.get("RSHOULDER")
        la_i = idx.get("LASIS")
        ra_i = idx.get("RASIS")
        if lsho_i is not None and rsho_i is not None and la_i is not None and ra_i is not None:
            lines.append(f"Markers: L_shoulder={lsho_i}  R_shoulder={rsho_i}  LASIS={la_i}  RASIS={ra_i}")
            lines.append("")
    if result.ta_deg is not None:
        result_lines = [
            f"TA:            {result.ta_deg:+.4f}\u00B0",
            f"Score:         {result.score}/2  {result.score_note}",
        ]
        lines.extend(_box_results(54, "■ ผลการวิเคราะห์ TA", result_lines))
        lines.append("")
        ta = result.ta_deg
        score = result.score or 0
        tilt_dir = "ขวา" if ta > 0 else "ซ้าย"
        lines.append("─── สรุปความหมายผล ───")
        summary_parts = [f"TA = {ta:+.1f}°", f"ลำตัวเอียง{tilt_dir}"]
        if score == 2:
            summary_parts.append("อยู่ในเกณฑ์")
        elif score == 1:
            summary_parts.append("ควรปรับ trunk control")
        lines.append("  " + "  ·  ".join(summary_parts))
        lines.append("")
        lines.append("─── นิยามมุม ───")
        lines.append("  TA = มุมแนว trunk (clavicular → midline pelvis) กับแนวตั้ง (vertical)")
        lines.append("  Clavicular = จุดกึ่งกลางไหล่ซ้าย–ขวา  ·  TA+ = เอียงขวา  TA- = เอียงซ้าย")
        if getattr(result, "used_world_vertical", False):
            lines.append("  Vertical จาก force plate (gravity)  ·  วัดในระนาบ frontal")
        lines.append("")
        lines.append("─── คำแนะนำ ───")
        _append_ts_recommendations(lines, result)
    else:
        lines.append("TA: N/A (required markers not found)")
    return "\n".join(lines)
