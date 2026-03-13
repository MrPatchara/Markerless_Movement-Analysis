"""
Force plate signal filtering - standard biomechanics.
Butterworth 4th-order, zero-phase (filtfilt).
Common: Low-pass 10–20 Hz, High-pass 0.5–1 Hz (drift removal), Bandpass 0.5–10 Hz.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

# Common cutoffs (Hz) from biomechanics literature
LP_FORCE_HZ = 10.0   # Low-pass for force/moments
HP_DRIFT_HZ = 0.5    # High-pass to remove baseline drift
BP_LOW_HZ = 0.5
BP_HIGH_HZ = 10.0


def _get_butter_filtfilt():
    try:
        from scipy.signal import butter, filtfilt
        return butter, filtfilt
    except ImportError:
        return None, None


def lowpass_butterworth(
    data: np.ndarray,
    cutoff_hz: float,
    sample_rate_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase low-pass Butterworth filter.
    
    Args:
        data: (n_samples,) or (n_samples, n_channels)
        cutoff_hz: Cutoff frequency (Hz)
        sample_rate_hz: Sampling rate (Hz)
        order: Filter order (default 4)
    
    Returns:
        Filtered array, same shape as data.
    """
    butter, filtfilt = _get_butter_filtfilt()
    if butter is None:
        return data.copy()
    if sample_rate_hz <= 0 or cutoff_hz <= 0:
        return data.copy()
    nyq = sample_rate_hz / 2
    if cutoff_hz >= nyq * 0.99:
        return data.copy()
    
    b, a = butter(order, cutoff_hz / nyq, btype="low", analog=False)
    data = np.asarray(data, dtype=np.float64)
    
    if data.ndim == 1:
        return filtfilt(b, a, data)
    
    out = np.empty_like(data)
    for c in range(data.shape[1]):
        out[:, c] = filtfilt(b, a, data[:, c])
    return out


def highpass_butterworth(
    data: np.ndarray,
    cutoff_hz: float,
    sample_rate_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase high-pass Butterworth. Typical: 0.5 Hz to remove drift."""
    butter, filtfilt = _get_butter_filtfilt()
    if butter is None:
        return data.copy()
    nyq = sample_rate_hz / 2
    if cutoff_hz <= 0 or cutoff_hz >= nyq * 0.99:
        return data.copy()
    b, a = butter(order, cutoff_hz / nyq, btype="high", analog=False)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        return filtfilt(b, a, data)
    out = np.empty_like(data)
    for c in range(data.shape[1]):
        out[:, c] = filtfilt(b, a, data[:, c])
    return out


def bandpass_butterworth(
    data: np.ndarray,
    low_hz: float,
    high_hz: float,
    sample_rate_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase bandpass (high-pass + low-pass). Typical: 0.5–10 Hz for force/COP."""
    butter, filtfilt = _get_butter_filtfilt()
    if butter is None:
        return data.copy()
    nyq = sample_rate_hz / 2
    if low_hz <= 0 or high_hz >= nyq or low_hz >= high_hz:
        return data.copy()
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band", analog=False)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        return filtfilt(b, a, data)
    out = np.empty_like(data)
    for c in range(data.shape[1]):
        out[:, c] = filtfilt(b, a, data[:, c])
    return out


def filter_force_plate_data(
    forces: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float = 10.0,
    order: int = 4,
) -> np.ndarray:
    """
    Filter force plate forces (n_frames, n_plates, 3).
    
    Args:
        forces: (n_frames, n_plates, 3) - Fx, Fy, Fz
        sample_rate_hz: Point/marker rate (frames per second)
        cutoff_hz: Low-pass cutoff (default 10 Hz, common for SLS/gait)
        order: Butterworth order (default 4)
    
    Returns:
        Filtered forces, same shape.
    """
    n_f, n_p, _ = forces.shape
    out = np.empty_like(forces)
    for p in range(n_p):
        out[:, p, :] = lowpass_butterworth(
            forces[:, p, :],
            cutoff_hz=cutoff_hz,
            sample_rate_hz=sample_rate_hz,
            order=order,
        )
    return out


def filter_cop_data(
    cops: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float = 10.0,
    order: int = 4,
) -> np.ndarray:
    """Filter COP trajectories (n_frames, n_plates, 3)."""
    return filter_force_plate_data(cops, sample_rate_hz, cutoff_hz, order)


def filter_force_bandpass(
    forces: np.ndarray,
    sample_rate_hz: float,
    low_hz: float = 0.5,
    high_hz: float = 10.0,
    order: int = 4,
) -> np.ndarray:
    """Bandpass filter force (0.5–10 Hz typical). Removes drift + noise."""
    n_f, n_p, _ = forces.shape
    out = np.empty_like(forces)
    for p in range(n_p):
        out[:, p, :] = bandpass_butterworth(
            forces[:, p, :], low_hz, high_hz, sample_rate_hz, order
        )
    return out


def filter_cop_bandpass(
    cops: np.ndarray,
    sample_rate_hz: float,
    low_hz: float = 0.5,
    high_hz: float = 10.0,
    order: int = 4,
) -> np.ndarray:
    """Bandpass filter COP (0.5–10 Hz typical)."""
    return filter_force_bandpass(cops, sample_rate_hz, low_hz, high_hz, order)
