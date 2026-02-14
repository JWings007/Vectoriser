"""Merge groups that are very similar based on group-level statistics."""
from typing import List
from ..parsing.svg_parser import PathInfo
from ..analysis.color_analysis import group_color_stats
from ..utils.color import delta_e
from ..utils.geometry import bbox_distance
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GroupMerger:
    """Merge groups that have similar color and spatial proximity."""

    def __init__(self, config: dict):
        grouping_cfg = config.get("grouping", {})
        self.proximity_threshold = grouping_cfg.get("proximity_threshold", 50.0)
        self.color_tolerance = grouping_cfg.get("color_tolerance", 25.0)

    def merge(self, groups: List[List[PathInfo]]) -> List[List[PathInfo]]:
        """
        Iteratively merge groups that are similar.
        Uses group-level statistics (not just first path).
        """
        if len(groups) <= 1:
            return groups

        merged = True
        current_groups = [list(g) for g in groups]

        while merged:
            merged = False
            new_groups = []
            used = set()

            # Pre-compute group stats and bounding boxes
            stats = [group_color_stats(g) for g in current_groups]
            bboxes = [self._group_bbox(g) for g in current_groups]

            for i in range(len(current_groups)):
                if i in used:
                    continue

                combined = list(current_groups[i])

                for j in range(i + 1, len(current_groups)):
                    if j in used:
                        continue

                    if self._should_merge(stats[i], stats[j], bboxes[i], bboxes[j]):
                        combined.extend(current_groups[j])
                        used.add(j)
                        merged = True

                new_groups.append(combined)
                used.add(i)

            current_groups = new_groups

        logger.info(
            f"Merged {len(groups)} groups down to {len(current_groups)} groups"
        )
        return current_groups

    def _should_merge(
        self, stats1: dict, stats2: dict, bbox1: tuple, bbox2: tuple
    ) -> bool:
        """Decide if two groups should be merged based on group-level stats."""
        # Color similarity (using mean CIELAB color)
        color_dist = delta_e(stats1["mean_lab"], stats2["mean_lab"])
        if color_dist > self.color_tolerance:
            return False

        # Spatial proximity (minimum distance between group bounding boxes)
        dist = bbox_distance(bbox1, bbox2)
        if dist > self.proximity_threshold:
            return False

        return True

    @staticmethod
    def _group_bbox(group: List[PathInfo]) -> tuple:
        """Compute bounding box for entire group."""
        xmin = min(p.bbox[0] for p in group)
        ymin = min(p.bbox[1] for p in group)
        xmax = max(p.bbox[2] for p in group)
        ymax = max(p.bbox[3] for p in group)
        return (xmin, ymin, xmax, ymax)
