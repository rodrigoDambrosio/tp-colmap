"""Localization accuracy metrics."""
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np


def quaternion_angular_error(q1, q2) -> float:
    """Angular difference in degrees between two unit quaternions."""
    q1 = np.array(q1, dtype=float)
    q2 = np.array(q2, dtype=float)
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return float(np.degrees(2 * np.arccos(dot)))


def _qvec2rotmat(qvec) -> np.ndarray:
    qw, qx, qy, qz = qvec / np.linalg.norm(qvec)
    return np.array([
        [1-2*qy**2-2*qz**2,  2*qx*qy-2*qw*qz,   2*qx*qz+2*qw*qy],
        [2*qx*qy+2*qw*qz,    1-2*qx**2-2*qz**2,  2*qy*qz-2*qw*qx],
        [2*qx*qz-2*qw*qy,    2*qy*qz+2*qw*qx,    1-2*qx**2-2*qy**2],
    ])


def parse_results_file(results_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Parses hloc localize_sfm output.
    Format per line: name qw qx qy qz tx ty tz
    Returns {name: (qvec, tvec)}.
    """
    poses = {}
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            name = parts[0]
            qvec = np.array([float(v) for v in parts[1:5]])
            tvec = np.array([float(v) for v in parts[5:8]])
            poses[name] = (qvec, tvec)
    return poses


def compute_localization_errors(
    results_path: Path,
    ground_truth: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Returns {image_name: (translation_error_m, rotation_error_deg)}.
    ground_truth values are (camera_center_world, quaternion_w2c).
    """
    if not results_path.exists():
        return None

    estimated = parse_results_file(results_path)
    errors = {}

    for name, (qvec_est, tvec_est) in estimated.items():
        if name not in ground_truth:
            continue
        center_gt, qvec_gt = ground_truth[name]

        # Estimated camera center: c = -R^T @ tvec
        R_est = _qvec2rotmat(qvec_est)
        center_est = -R_est.T @ tvec_est

        t_err = float(np.linalg.norm(center_est - center_gt))
        r_err = quaternion_angular_error(qvec_est, qvec_gt)
        errors[name] = (t_err, r_err)

    return errors


def recall_at_thresholds(
    errors: Dict[str, Tuple[float, float]],
    t_thresholds=(0.25, 0.5, 5.0),
    r_thresholds=(2.0, 5.0, 10.0),
) -> Dict[str, float]:
    """
    Returns recall (%) at each (t_threshold, r_threshold) pair.
    Standard benchmarks use (0.25m, 2°), (0.5m, 5°), (5m, 10°).
    """
    results = {}
    total = len(errors)
    if total == 0:
        return {}
    for t_thr, r_thr in zip(t_thresholds, r_thresholds):
        key = f"({t_thr}m, {r_thr}°)"
        count = sum(1 for t, r in errors.values() if t <= t_thr and r <= r_thr)
        results[key] = 100.0 * count / total
    return results
