"""DBSCAN-based path clustering."""
import numpy as np
from collections import defaultdict
from typing import List
from sklearn.cluster import DBSCAN

from ..parsing.svg_parser import PathInfo
from ..analysis.feature_extractor import FeatureExtractor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PathClusterer:
    """Group paths into clusters using DBSCAN on feature vectors."""

    def __init__(self, config: dict):
        self.config = config
        cluster_cfg = config.get("clustering", {})
        self.eps = cluster_cfg.get("eps", 0.5)
        self.min_samples = cluster_cfg.get("min_samples", 1)
        self.feature_extractor = FeatureExtractor(config)

    def cluster(
        self,
        paths: List[PathInfo],
        image_width: float = 1.0,
        image_height: float = 1.0,
    ) -> List[List[PathInfo]]:
        """
        Cluster paths using DBSCAN.

        Returns:
            List of groups, where each group is a list of PathInfo.
        """
        if len(paths) == 0:
            return []
        if len(paths) == 1:
            return [paths]

        # Extract features
        features = self.feature_extractor.extract(paths, image_width, image_height)

        # Run DBSCAN
        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="euclidean",
        )
        labels = clustering.fit_predict(features)

        # Organize into groups
        groups_dict = defaultdict(list)
        for path, label in zip(paths, labels):
            path.group_id = int(label)
            groups_dict[label].append(path)

        groups = list(groups_dict.values())

        # Sort groups by average position (top-left first) for consistent ordering
        groups.sort(
            key=lambda g: (
                np.mean([p.centroid[1] for p in g]),
                np.mean([p.centroid[0] for p in g]),
            )
        )

        logger.info(
            f"DBSCAN produced {len(groups)} clusters "
            f"from {len(paths)} paths (eps={self.eps})"
        )

        return groups
