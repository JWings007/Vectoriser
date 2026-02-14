"""Build rich feature vectors for path clustering."""
import numpy as np
from typing import List
from ..parsing.svg_parser import PathInfo
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """Extract multi-dimensional feature vectors from paths for clustering."""

    def __init__(self, config: dict):
        cluster_cfg = config.get("clustering", {})
        self.weights = cluster_cfg.get("feature_weights", {})

    def extract(
        self,
        paths: List[PathInfo],
        image_width: float = 1.0,
        image_height: float = 1.0,
    ) -> np.ndarray:
        """
        Build normalized feature matrix.

        Features per path:
          [0] centroid_x (normalized)
          [1] centroid_y (normalized)
          [2] L  (CIELAB lightness, normalized 0-1)
          [3] a  (CIELAB, normalized)
          [4] b  (CIELAB, normalized)
          [5] log_area (normalized)
          [6] aspect_ratio (clamped and normalized)
          [7] complexity (log-scaled)
          [8] curve_ratio (already 0-1)

        All features are scaled to roughly 0-1 range, then multiplied
        by configured weights.
        """
        if not paths:
            return np.empty((0, 9))

        # Compute normalization bounds
        max_area = max(p.area for p in paths) or 1.0
        max_complexity = max(p.complexity for p in paths) or 1

        # Ensure we have reasonable image dimensions for normalization
        if image_width < 1.0:
            image_width = max(p.bbox[2] for p in paths) or 1.0
        if image_height < 1.0:
            image_height = max(p.bbox[3] for p in paths) or 1.0

        features = []
        for p in paths:
            cx, cy = p.centroid

            # Position features (normalized to image dimensions)
            norm_cx = cx / image_width
            norm_cy = cy / image_height

            # Color features (CIELAB)
            if p.lab_color is not None:
                l_norm = p.lab_color[0] / 100.0
                a_norm = (p.lab_color[1] + 128) / 256.0
                b_norm = (p.lab_color[2] + 128) / 256.0
            else:
                l_norm, a_norm, b_norm = 0.0, 0.5, 0.5

            # Area feature (log-scaled)
            log_area = np.log1p(p.area) / np.log1p(max_area)

            # Aspect ratio (clamped to 0-1 range)
            ar = min(p.aspect_ratio, 10.0) / 10.0

            # Complexity (log-scaled)
            comp = np.log1p(p.complexity) / np.log1p(max_complexity)

            # Curve ratio (already 0-1)
            cr = p.curve_ratio

            features.append(
                [
                    norm_cx,
                    norm_cy,
                    l_norm,
                    a_norm,
                    b_norm,
                    log_area,
                    ar,
                    comp,
                    cr,
                ]
            )

        features = np.array(features, dtype=np.float64)

        # Apply weights
        w_pos = self.weights.get("position", 1.0)
        w_color = self.weights.get("color", 2.0)
        w_area = self.weights.get("area", 0.5)
        w_complexity = self.weights.get("complexity", 0.3)
        w_ar = self.weights.get("aspect_ratio", 0.3)

        weight_vector = np.array(
            [
                w_pos,
                w_pos,
                w_color,
                w_color,
                w_color,
                w_area,
                w_ar,
                w_complexity,
                w_complexity,
            ]
        )

        features *= weight_vector

        logger.info(
            f"Extracted {features.shape[1]} features for {features.shape[0]} paths"
        )
        return features
