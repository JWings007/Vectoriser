"""Color analysis utilities for grouping decisions."""
import numpy as np
from collections import Counter
from typing import List, Dict
from ..parsing.svg_parser import PathInfo
from ..utils.color import delta_e
from ..utils.logger import get_logger

logger = get_logger(__name__)


def dominant_colors(paths: List[PathInfo], top_n: int = 10) -> List[str]:
    """Find the most common fill colors in a set of paths."""
    counter = Counter(p.fill for p in paths)
    return [color for color, _ in counter.most_common(top_n)]


def group_color_stats(paths: List[PathInfo]) -> Dict:
    """
    Compute color statistics for a group of paths.
    Returns dict with mean LAB, color variance, dominant color, etc.
    """
    lab_values = [p.lab_color for p in paths if p.lab_color is not None]

    if not lab_values:
        return {
            "mean_lab": (50.0, 0.0, 0.0),
            "variance": 0.0,
            "dominant_fill": paths[0].fill if paths else "#000000",
            "unique_colors": 0,
        }

    lab_array = np.array(lab_values)
    mean_lab = tuple(lab_array.mean(axis=0))

    # Color variance: average Delta-E from mean
    variance = np.mean([delta_e(mean_lab, lab) for lab in lab_values])

    fills = [p.fill for p in paths]
    dominant = Counter(fills).most_common(1)[0][0]

    return {
        "mean_lab": mean_lab,
        "variance": variance,
        "dominant_fill": dominant,
        "unique_colors": len(set(fills)),
    }
