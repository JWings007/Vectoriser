"""
Graph-based path grouping using spatial adjacency.
Alternative to pure DBSCAN — builds an adjacency graph and uses
community detection.
"""
import numpy as np
import networkx as nx
from typing import List

from ..parsing.svg_parser import PathInfo
from ..utils.geometry import bbox_distance
from ..utils.color import delta_e
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GraphGrouper:
    """
    Build a spatial adjacency graph of paths and detect communities.
    Edges connect paths that are spatially close AND chromatically similar.
    """

    def __init__(self, config: dict):
        grouping_cfg = config.get("grouping", {})
        self.proximity_threshold = grouping_cfg.get("proximity_threshold", 50.0)
        self.color_tolerance = grouping_cfg.get("color_tolerance", 25.0)

    def group(self, paths: List[PathInfo]) -> List[List[PathInfo]]:
        """
        Build adjacency graph and detect communities.

        Returns:
            List of groups (list of PathInfo each).
        """
        if len(paths) == 0:
            return []
        if len(paths) <= 2:
            return [paths]

        G = nx.Graph()

        # Add all paths as nodes
        for i, _ in enumerate(paths):
            G.add_node(i)

        # Add edges based on spatial proximity + color similarity
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                dist = bbox_distance(paths[i].bbox, paths[j].bbox)

                if dist > self.proximity_threshold:
                    continue

                # Check color similarity
                color_dist = float("inf")
                if paths[i].lab_color and paths[j].lab_color:
                    color_dist = delta_e(paths[i].lab_color, paths[j].lab_color)
                elif paths[i].fill == paths[j].fill:
                    color_dist = 0.0

                if color_dist > self.color_tolerance:
                    continue

                # Edge weight: inverse of combined distance (higher = more similar)
                spatial_weight = 1.0 - (dist / self.proximity_threshold)
                color_weight = 1.0 - (color_dist / self.color_tolerance)
                weight = (spatial_weight + color_weight) / 2.0

                G.add_edge(i, j, weight=weight)

        # Detect communities using Louvain method
        try:
            communities = nx.community.louvain_communities(
                G, weight="weight", resolution=1.0, seed=42
            )
        except Exception:
            communities = list(nx.connected_components(G))

        # Convert to path groups
        groups = []
        for community in communities:
            group = [paths[i] for i in sorted(community)]
            groups.append(group)

        # Sort groups by position
        groups.sort(
            key=lambda g: (
                np.mean([p.centroid[1] for p in g]),
                np.mean([p.centroid[0] for p in g]),
            )
        )

        logger.info(
            f"Graph grouping produced {len(groups)} groups "
            f"from {len(paths)} paths"
        )

        return groups
