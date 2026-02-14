"""Shared geometry helper functions."""
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from svgpathtools import parse_path, CubicBezier, QuadraticBezier, Arc
from typing import Tuple, Optional


def path_to_bbox(d: str) -> Tuple[float, float, float, float]:
    """
    Compute accurate bounding box from SVG path d string
    using svgpathtools.
    Returns (xmin, ymin, xmax, ymax).
    """
    try:
        path = parse_path(d)
        if len(path) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        xmin, xmax, ymin, ymax = path.bbox()
        return (xmin, ymin, xmax, ymax)
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)


def bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    """Calculate area of a bounding box."""
    xmin, ymin, xmax, ymax = bbox
    return max(0.0, (xmax - xmin) * (ymax - ymin))


def bbox_centroid(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Calculate centroid of a bounding box."""
    xmin, ymin, xmax, ymax = bbox
    return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)


def bbox_aspect_ratio(bbox: Tuple[float, float, float, float]) -> float:
    """Calculate aspect ratio (width / height). Returns 1.0 if degenerate."""
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    if h < 1e-6:
        return 1.0
    return w / h


def bboxes_overlap(a: Tuple, b: Tuple) -> bool:
    """Check if two bounding boxes overlap."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def bbox_distance(a: Tuple, b: Tuple) -> float:
    """Minimum distance between two bounding boxes."""
    if bboxes_overlap(a, b):
        return 0.0

    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return np.sqrt(dx**2 + dy**2)


def path_to_polygon(d: str, num_samples: int = 64) -> Optional[Polygon]:
    """
    Convert SVG path to a Shapely polygon by sampling points along the path.
    Used for accurate containment checking.
    """
    try:
        path = parse_path(d)
        if len(path) == 0:
            return None

        points = []
        total_length = path.length()
        if total_length < 1e-6:
            return None

        for i in range(num_samples):
            t = i / num_samples
            try:
                pt = path.point(t)
                points.append((pt.real, pt.imag))
            except Exception:
                continue

        if len(points) < 3:
            return None

        poly = Polygon(points)
        if not poly.is_valid:
            poly = make_valid(poly)
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda g: g.area)
            if not isinstance(poly, Polygon):
                return None

        return poly
    except Exception:
        return None


def path_complexity(d: str) -> int:
    """
    Count the number of segments in a path.
    Higher = more complex shape.
    """
    try:
        path = parse_path(d)
        return len(path)
    except Exception:
        return 0


def path_curve_ratio(d: str) -> float:
    """
    Ratio of curved segments (Bezier, Arc) to total segments.
    0.0 = all straight lines, 1.0 = all curves.
    """
    try:
        path = parse_path(d)
        if len(path) == 0:
            return 0.0

        curved = sum(
            1
            for seg in path
            if isinstance(seg, (CubicBezier, QuadraticBezier, Arc))
        )
        return curved / len(path)
    except Exception:
        return 0.0
