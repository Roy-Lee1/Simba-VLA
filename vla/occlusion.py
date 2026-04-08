"""
Occlusion simulation utilities for robustness evaluation.

Provides controlled occlusion patterns to evaluate how well the
completion-augmented VLA pipeline handles missing geometry. This is
the key ablation axis: demonstrating that Simba completion improves
action prediction under increasing occlusion severity.

Occlusion types:
    - Viewpoint: simulate single-view depth observation (realistic)
    - Planar: remove points on one side of a cutting plane
    - Random dropout: randomly remove a fraction of points
    - Distance-based: remove points far from a reference point
    - Sector: remove points in an angular sector (partial view)
"""

import numpy as np


def viewpoint_occlusion(points, viewpoint=None, keep_ratio=0.5):
    """Simulate single-view observation by keeping only points visible
    from a specified viewpoint.

    Approximates a depth camera by projecting points and keeping only
    the front-facing (closest) points per view ray bucket.

    Args:
        points: (N, 3) numpy array
        viewpoint: (3,) camera position. If None, uses random viewpoint.
        keep_ratio: fraction of points to keep (controls occlusion severity)
    Returns:
        occluded: (M, 3) surviving points
        mask: (N,) boolean mask of kept points
    """
    N = points.shape[0]
    if viewpoint is None:
        viewpoint = np.random.randn(3).astype(np.float32)
        viewpoint = viewpoint / (np.linalg.norm(viewpoint) + 1e-8) * 2.0

    # Direction from viewpoint to each point
    directions = points - viewpoint[None, :]
    distances = np.linalg.norm(directions, axis=1)

    # Simple visibility: keep the closest points
    # (real visibility would require ray-casting, this approximates it)
    n_keep = max(1, int(N * keep_ratio))
    sorted_idx = np.argsort(distances)
    keep_idx = sorted_idx[:n_keep]

    mask = np.zeros(N, dtype=bool)
    mask[keep_idx] = True

    return points[mask], mask


def planar_occlusion(points, normal=None, offset=None, keep_ratio=0.5):
    """Remove points on one side of a cutting plane.

    Simulates occlusion from a wall, table surface, or other planar obstacle.

    Args:
        points: (N, 3)
        normal: (3,) plane normal. Random if None.
        offset: scalar plane offset. Auto-computed if None.
        keep_ratio: target fraction of points to keep
    Returns:
        occluded: (M, 3)
        mask: (N,) boolean
    """
    N = points.shape[0]
    if normal is None:
        normal = np.random.randn(3).astype(np.float32)
        normal /= np.linalg.norm(normal) + 1e-8

    projections = points @ normal
    if offset is None:
        offset = np.quantile(projections, 1.0 - keep_ratio)

    mask = projections <= offset
    if mask.sum() == 0:
        mask[np.argmin(projections)] = True

    return points[mask], mask


def random_dropout(points, keep_ratio=0.5):
    """Randomly drop a fraction of points.

    Simulates sensor noise, sparse LiDAR returns, or network packet loss.

    Args:
        points: (N, 3)
        keep_ratio: fraction to keep
    Returns:
        occluded: (M, 3)
        mask: (N,) boolean
    """
    N = points.shape[0]
    n_keep = max(1, int(N * keep_ratio))
    indices = np.random.choice(N, size=n_keep, replace=False)

    mask = np.zeros(N, dtype=bool)
    mask[indices] = True

    return points[mask], mask


def distance_occlusion(points, center=None, keep_ratio=0.5, keep_near=True):
    """Remove points based on distance from a reference point.

    Simulates range-limited sensors or distance-based degradation.

    Args:
        points: (N, 3)
        center: (3,) reference point. Uses centroid if None.
        keep_ratio: fraction to keep
        keep_near: if True, keep nearest points; if False, keep farthest
    Returns:
        occluded: (M, 3)
        mask: (N,) boolean
    """
    N = points.shape[0]
    if center is None:
        center = points.mean(axis=0)

    distances = np.linalg.norm(points - center[None, :], axis=1)
    n_keep = max(1, int(N * keep_ratio))
    sorted_idx = np.argsort(distances)

    if keep_near:
        keep_idx = sorted_idx[:n_keep]
    else:
        keep_idx = sorted_idx[-n_keep:]

    mask = np.zeros(N, dtype=bool)
    mask[keep_idx] = True

    return points[mask], mask


def sector_occlusion(points, axis=2, angle_range=None, keep_ratio=0.5):
    """Remove points in an angular sector around a specified axis.

    Simulates partial-view observation where only a portion of the
    object is visible due to viewing angle constraints.

    Args:
        points: (N, 3)
        axis: which axis to compute angle around (0=x, 1=y, 2=z)
        angle_range: (min_angle, max_angle) in radians to REMOVE. Auto if None.
        keep_ratio: target keep ratio (used to auto-compute angle_range)
    Returns:
        occluded: (M, 3)
        mask: (N,) boolean
    """
    N = points.shape[0]
    # Project to the plane perpendicular to the chosen axis
    axes = [0, 1, 2]
    axes.remove(axis)
    a, b = axes

    angles = np.arctan2(points[:, b], points[:, a])  # range [-pi, pi]

    if angle_range is None:
        remove_fraction = 1.0 - keep_ratio
        span = remove_fraction * 2 * np.pi
        start = np.random.uniform(-np.pi, np.pi - span)
        angle_range = (start, start + span)

    min_angle, max_angle = angle_range
    if min_angle < max_angle:
        in_sector = (angles >= min_angle) & (angles <= max_angle)
    else:
        in_sector = (angles >= min_angle) | (angles <= max_angle)

    mask = ~in_sector
    if mask.sum() == 0:
        mask[0] = True

    return points[mask], mask


class OcclusionSimulator:
    """Configurable occlusion pipeline for data augmentation and evaluation.

    Applies a sequence of occlusion operations with controllable severity.
    Used during training for data augmentation and during evaluation for
    controlled robustness benchmarking.

    Args:
        occlusion_types: list of occlusion type names to apply
        severity: overall occlusion severity in [0, 1] (0 = no occlusion)
        random_severity: whether to randomize severity per sample
    """

    METHODS = {
        "viewpoint": viewpoint_occlusion,
        "planar": planar_occlusion,
        "random": random_dropout,
        "distance": distance_occlusion,
        "sector": sector_occlusion,
    }

    def __init__(self, occlusion_types=None, severity=0.3, random_severity=False):
        self.occlusion_types = occlusion_types or ["random"]
        self.severity = severity
        self.random_severity = random_severity

        for otype in self.occlusion_types:
            if otype not in self.METHODS:
                raise ValueError(f"Unknown occlusion type: {otype}. Available: {list(self.METHODS.keys())}")

    def __call__(self, points):
        """Apply occlusion to a point cloud.

        Args:
            points: (N, 3) numpy array
        Returns:
            occluded_points: (M, 3) with M < N
            occlusion_info: dict with applied method and parameters
        """
        severity = self.severity
        if self.random_severity:
            severity = np.random.uniform(0.0, self.severity)

        keep_ratio = 1.0 - severity
        keep_ratio = max(keep_ratio, 0.05)  # keep at least 5%

        method_name = np.random.choice(self.occlusion_types)
        method = self.METHODS[method_name]

        occluded, mask = method(points, keep_ratio=keep_ratio)

        info = {
            "method": method_name,
            "severity": severity,
            "original_points": points.shape[0],
            "remaining_points": occluded.shape[0],
            "actual_keep_ratio": occluded.shape[0] / max(points.shape[0], 1),
        }

        return occluded, info


def create_occlusion_sweep(points, method="random", severities=None):
    """Generate multiple occlusion levels for systematic evaluation.

    Creates a sweep of increasing occlusion severity for plotting
    performance degradation curves.

    Args:
        points: (N, 3)
        method: occlusion method name
        severities: list of severity values. Defaults to [0, 0.1, ..., 0.9].
    Returns:
        results: list of (occluded_points, severity, keep_ratio) tuples
    """
    if severities is None:
        severities = [i * 0.1 for i in range(10)]

    fn = OcclusionSimulator.METHODS[method]
    results = []

    for severity in severities:
        keep_ratio = max(1.0 - severity, 0.05)
        occluded, mask = fn(points, keep_ratio=keep_ratio)
        results.append({
            "points": occluded,
            "severity": severity,
            "keep_ratio": occluded.shape[0] / max(points.shape[0], 1),
        })

    return results
