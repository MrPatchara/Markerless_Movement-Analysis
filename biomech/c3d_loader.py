"""
C3D File Loader for Markerless Motion Capture
Supports standard C3D format and extracts skeleton data for visualization.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


# Common skeleton connections for markerless systems
# Format: (start_idx, end_idx) for each bone
# MediaPipe Pose 33 landmarks
MEDIAPIPE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),   # Face
    (0, 4), (4, 5), (5, 6), (6, 8),   # Face
    (9, 10),  # shoulders
    (11, 12), # hips
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Left arm/leg
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Right arm/leg
    (11, 23), (12, 24), (23, 24),     # Torso
    (23, 25), (25, 27), (27, 29), (27, 31),  # Left leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # Right leg
]

# BTS Brain / HALPE 26 keypoints (TensorRT SimCC model)
HALPE_26_SKELETON = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (18, 0),
    (18, 6), (18, 5), (6, 8), (8, 10), (5, 7), (7, 9),
    (18, 19), (19, 12), (12, 14), (14, 16), (16, 25), (25, 21), (21, 23), (23, 25),
    (19, 11), (11, 13), (13, 15), (15, 24), (24, 20), (20, 22), (22, 24),
]

# OpenPose BODY_25 - simplified
BODY_25_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),   # Head
    (0, 15), (0, 16),  # Ears
    (1, 8),  # Neck to mid-hip
    (8, 9), (9, 10), (10, 11), (11, 22), (11, 24),  # Right arm
    (8, 12), (12, 13), (13, 14), (14, 19), (14, 21),  # Left arm
    (8, 1),  # Torso
    (1, 2), (2, 3), (3, 4),  # Right leg
    (1, 5), (5, 6), (6, 7),  # Left leg
]

# 2GRF.XMF skeleton: segment pairs by normalized label name (lowercase, hyphen).
# Resolved to indices at load time so 2D/3D view work regardless of C3D point order.
_2GRF_CONNECTION_NAME_PAIRS = [
    ("right-hip", "left-hip"),
    ("right-hip", "right-knee"),
    ("right-knee", "right-ankle"),
    ("left-hip", "left-knee"),
    ("left-knee", "left-ankle"),
    ("right-shoulder", "left-shoulder"),
    ("left-shoulder", "left-hip"),
    ("right-shoulder", "right-hip"),
    ("right-ankle", "right-heel"),
    ("right-heel", "right-small-toe"),
    ("right-small-toe", "right-big-toe"),
    ("right-big-toe", "right-heel"),
    ("left-heel", "left-big-toe"),
    ("left-big-toe", "left-small-toe"),
    ("left-small-toe", "left-heel"),
    ("left-ankle", "left-heel"),
    ("right-shoulder", "right-elbow"),
    ("right-elbow", "right-wrist"),
    ("left-elbow", "left-wrist"),
    ("left-elbow", "left-shoulder"),
    ("head", "left-ear"),
    ("head", "right-ear"),
    ("right-ear", "neck"),
    ("neck", "left-ear"),
    ("head", "neck"),
    ("right-ear", "left-ear"),
    ("right-eye", "left-eye"),
    ("left-eye", "nose"),
    ("nose", "right-eye"),
    ("hip", "right-hip"),
    ("hip", "left-hip"),
]


def _normalize_label(lbl: str) -> str:
    s = (lbl or "").strip().lower()
    for ch in ("_", " "):
        s = s.replace(ch, "-")
    return s


def _infer_connections_by_labels(labels: List[str]) -> Optional[List[Tuple[int, int]]]:
    """
    Build skeleton connections from label names (e.g. 2GRF.XMF export).
    Returns list of (start_idx, end_idx) or None if not enough name matches.
    """
    if not labels:
        return None
    norm_to_idx: Dict[str, int] = {}
    for i, lbl in enumerate(labels):
        n = _normalize_label(str(lbl))
        if n and n not in norm_to_idx:
            norm_to_idx[n] = i
    # Prefer 2GRF if we have key markers
    key_2grf = {"right-shoulder", "left-shoulder", "right-hip", "left-hip", "hip", "neck"}
    if key_2grf.intersection(norm_to_idx.keys()):
        conns = []
        for a, b in _2GRF_CONNECTION_NAME_PAIRS:
            ia = norm_to_idx.get(a)
            ib = norm_to_idx.get(b)
            if ia is not None and ib is not None:
                conns.append((ia, ib))
        if len(conns) >= 10:
            return conns
    return None


@dataclass
class C3DData:
    """Loaded C3D motion capture data."""
    points: np.ndarray          # (n_frames, n_points, 3) - x,y,z
    labels: List[str]           # Point/marker names
    frame_rate: float           # Hz
    residuals: Optional[np.ndarray] = None  # (n_frames, n_points)
    connections: List[Tuple[int, int]] = None  # Skeleton bone connections
    source: str = "c3d"         # "c3d" or "json" - for coord system handling
    force_plates: Optional[List[np.ndarray]] = None  # List of (4,3) corner arrays per plate
    force_forces: Optional[np.ndarray] = None  # (n_frames, n_plates, 3) - GRF vector per frame per plate
    force_cops: Optional[np.ndarray] = None  # (n_frames, n_plates, 3) - COP position per frame per plate
    
    @property
    def n_frames(self) -> int:
        return self.points.shape[0]
    
    @property
    def n_points(self) -> int:
        return self.points.shape[1]
    
    def get_frame(self, frame_idx: int) -> np.ndarray:
        """Get 3D points for a single frame (n_points, 3)."""
        return self.points[frame_idx]
    
    def get_valid_points(self, frame_idx: int, max_residual: float = 10.0) -> np.ndarray:
        """Get points with valid (low residual) data. Returns mask."""
        if self.residuals is None:
            return np.ones(self.n_points, dtype=bool)
        return self.residuals[frame_idx] < max_residual


def load_motion_file(filepath: str | Path) -> C3DData:
    """
    Load motion capture file (C3D or BTS Brain JSON).
    
    Args:
        filepath: Path to .c3d or .json file
        
    Returns:
        C3DData with points, labels, frame rate, and skeleton connections
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    suffix = filepath.suffix.lower()
    if suffix == '.json':
        return _load_bts_brain_json(filepath)
    
    # C3D
    try:
        return _load_ezc3d(filepath)
    except ImportError:
        pass
    except Exception:
        pass
    return _load_pyc3d(filepath)


def load_c3d(filepath: str | Path) -> C3DData:
    """Load C3D file. Alias for load_motion_file for .c3d files."""
    return load_motion_file(filepath)


def _load_bts_brain_json(filepath: Path) -> C3DData:
    """
    Load BTS Brain JSON format (frames_3d, keypoints_3d).
    Native output from BTS Brain AI markerless pipeline.
    """
    import json
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    frames_data = data.get('frames_3d', data.get('frames', []))
    if not frames_data:
        raise ValueError("JSON must contain 'frames_3d' or 'frames'")
    
    # Handle per-frame format: [{"frame_id": i, "keypoints_3d": [[x,y,z,conf],...]}, ...]
    points_list = []
    conf_list = []
    for fr in frames_data:
        kpts = fr.get('keypoints_3d', fr.get('keypoints', []))
        xyz = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in kpts], dtype=np.float64)
        conf = np.array([float(p[3]) if len(p) > 3 else 1.0 for p in kpts], dtype=np.float64)
        # Mark invalid (conf <= 0) as nan
        xyz[conf <= 0] = np.nan
        points_list.append(xyz)
        conf_list.append(conf)
    
    points = np.array(points_list)
    residuals = np.array(conf_list)
    n_points = points.shape[1]
    
    labels = [f"Kpt_{i}" for i in range(n_points)]
    frame_rate = float(data.get('frame_rate', data.get('fps', 60.0)))
    connections = _infer_skeleton_connections(n_points, labels)
    
    return C3DData(
        points=points,
        labels=labels,
        frame_rate=frame_rate,
        residuals=residuals,
        connections=connections,
        source="json"
    )


def _load_ezc3d(filepath: Path) -> C3DData:
    """Load using ezc3d (C++ backend)."""
    import ezc3d  # type: ignore[import-untyped]
    c3d = ezc3d.c3d(str(filepath))
    
    raw_points = c3d['data']['points']
    n_points = raw_points.shape[1]
    n_frames = raw_points.shape[2]
    
    points = np.transpose(raw_points[:3, :, :], (2, 1, 0))
    residuals = np.transpose(raw_points[3, :, :], (1, 0))
    
    params = c3d.get('parameters', {})
    point_params = params.get('POINT', {})
    labels_val = point_params.get('LABELS', {}).get('value', [])
    
    if isinstance(labels_val, str):
        labels = [labels_val]
    else:
        labels = [str(l).strip() for l in labels_val] if labels_val else [f"Point_{i}" for i in range(n_points)]
    while len(labels) < n_points:
        labels.append(f"Point_{len(labels)}")
    labels = labels[:n_points]
    
    rate_val = point_params.get('RATE', {}).get('value', [100])
    frame_rate = float(rate_val[0]) if rate_val else 100.0
    connections = _infer_skeleton_connections(n_points, labels)
    force_plates = _read_force_plates_ezc3d(c3d)
    force_forces, force_cops = _extract_force_cop(filepath)

    return C3DData(
        points=points, labels=labels, frame_rate=frame_rate,
        residuals=residuals, connections=connections, source="c3d",
        force_plates=force_plates, force_forces=force_forces, force_cops=force_cops
    )


def _read_force_plates_ezc3d(c3d) -> Optional[List[np.ndarray]]:
    """Extract force plate corners from ezc3d C3D data."""
    try:
        params = c3d.get("parameters", {})
        fp = params.get("FORCE_PLATFORM", {})
        used = fp.get("USED", {}).get("value", [0])
        n = int(used[0]) if np.isscalar(used) or len(used) == 0 else int(used[0])
        if n <= 0:
            return None
        corners_val = fp.get("CORNERS", {}).get("value")
        if corners_val is None:
            return None
        arr = np.asarray(corners_val)
        if arr.ndim == 2:
            arr = arr.reshape(3, 4, n)
        plates = [arr[:, :, i].T for i in range(n)]
        return plates
    except Exception:
        return None


def _load_pyc3d(filepath: Path) -> C3DData:
    """Load using py-c3d (pure Python)."""
    import warnings
    try:
        import c3d as pyc3d
    except ImportError:
        raise ImportError("Please install c3d: pip install c3d")
    
    frames_list = []
    labels = []
    frame_rate = 100.0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with open(filepath, 'rb') as f:
            reader = pyc3d.Reader(f)
            
            # Get point labels from header
            pl = getattr(reader, 'point_labels', None)
            if pl is not None and len(pl) > 0:
                labels = [str(l).strip() for l in pl]
            if hasattr(reader, 'point_rate') and reader.point_rate:
                frame_rate = float(reader.point_rate)
            
            for _frame_num, points, analog in reader.read_frames():
                # points: (n_points, 5) - x, y, z, error, camera_count
                # Invalid points have -1 in cols 4,5
                xyz = points[:, :3].copy()
                residuals = points[:, 3].copy()
                invalid = (points[:, 4] < 0) | (points[:, 3] < 0)
                xyz[invalid] = np.nan
                residuals[invalid] = np.inf  # use inf for invalid
                frames_list.append((xyz, residuals))
    
    if not frames_list:
        raise ValueError("No frames found in C3D file")
    
    points = np.array([f[0] for f in frames_list])
    residuals = np.array([f[1] for f in frames_list])
    n_points = points.shape[1]
    
    while len(labels) < n_points:
        labels.append(f"Point_{len(labels)}")
    labels = labels[:n_points]
    
    connections = _infer_skeleton_connections(n_points, labels)
    force_plates = _read_force_plates_pyc3d(filepath)
    force_forces, force_cops = _extract_force_cop(filepath)

    return C3DData(
        points=points,
        labels=labels,
        frame_rate=frame_rate,
        residuals=residuals,
        connections=connections,
        source="c3d",
        force_plates=force_plates,
        force_forces=force_forces,
        force_cops=force_cops,
    )


def _extract_force_cop(filepath: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract GRF forces and COP from C3D analog data."""
    try:
        from biomech.force_platform import extract_force_cop
        result = extract_force_cop(filepath)
        if result is not None:
            return result
    except Exception:
        pass
    return None, None


def _read_force_plates_pyc3d(filepath: Path) -> Optional[List[np.ndarray]]:
    """Extract force plate corners from py-c3d C3D file."""
    try:
        import c3d as c3d_lib
        with open(filepath, "rb") as f:
            reader = c3d_lib.Reader(f)
            for gname, g in reader.group_items():
                if gname == "FORCE_PLATFORM":
                    used = g.get_int16("USED")
                    if used <= 0:
                        return None
                    for pname, p in g.param_items():
                        if pname == "CORNERS":
                            arr = p.float_array
                            n = arr.shape[0]
                            plates = [arr[i] for i in range(n)]
                            return plates
                    return None
        return None
    except Exception:
        return None


def _infer_skeleton_connections(n_points: int, labels: List[str]) -> List[Tuple[int, int]]:
    """
    Infer skeleton bone connections from point count or labels.
    Uses name-based lookup for 2GRF-style exports first, then presets by point count.
    """
    # Prefer label-based connections (2GRF.XMF and similar) so 2D/3D view match model topology
    if labels:
        by_name = _infer_connections_by_labels(labels)
        if by_name:
            return [(a, b) for a, b in by_name if a < n_points and b < n_points]

    # MediaPipe 33 points
    if n_points == 33:
        return MEDIAPIPE_CONNECTIONS

    # BTS Brain / HALPE 26 (index order matches HALPE, not 2GRF)
    if n_points == 26:
        return HALPE_26_SKELETON

    # OpenPose BODY_25
    if n_points == 25:
        return BODY_25_CONNECTIONS

    # Generic: connect sequential points (simple chain)
    if n_points <= 15:
        return [(i, i + 1) for i in range(n_points - 1)]

    connections = [(i, i + 1) for i in range(n_points - 1)]
    if not connections:
        connections = [(i, i + 1) for i in range(min(n_points - 1, 30))]
    return [(a, b) for a, b in connections if a < n_points and b < n_points]
