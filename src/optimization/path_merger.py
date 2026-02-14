"""Merge adjacent paths that share the same fill color."""
from typing import List
from ..parsing.svg_parser import PathInfo
from ..utils.geometry import bboxes_overlap
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PathMerger:
    """
    Merge paths within a group that have the same fill and are adjacent.
    This reduces SVG element count without visual change.
    """

    def __init__(self, config: dict):
        opt_cfg = config.get("optimization", {})
        self.merge_adjacent = opt_cfg.get("merge_adjacent", True)

    def merge_within_groups(self, groups: List[List[PathInfo]]) -> List[List[PathInfo]]:
        """Merge same-fill adjacent paths within each group."""
        if not self.merge_adjacent:
            return groups

        total_before = sum(len(g) for g in groups)
        merged_groups = []

        for group in groups:
            merged_group = self._merge_group(group)
            merged_groups.append(merged_group)

        total_after = sum(len(g) for g in merged_groups)
        logger.info(
            f"Path merging: {total_before} paths → {total_after} paths "
            f"({total_before - total_after} merged)"
        )
        return merged_groups

    def _merge_group(self, group: List[PathInfo]) -> List[PathInfo]:
        """Merge same-fill adjacent paths in a single group."""
        if len(group) <= 1:
            return group

        # Group by fill color + transform (avoid breaking transformed paths)
        by_fill = {}
        for path in group:
            key = (path.fill, path.transform)
            if key not in by_fill:
                by_fill[key] = []
            by_fill[key].append(path)

        result = []
        for (fill, _transform), same_color_paths in by_fill.items():
            if len(same_color_paths) == 1:
                result.append(same_color_paths[0])
                continue

            merged = self._merge_overlapping(same_color_paths)
            result.extend(merged)

        return result

    def _merge_overlapping(self, paths: List[PathInfo]) -> List[PathInfo]:
        """
        Merge paths that overlap or are very close.
        Combines their d strings into a single compound path.
        """
        if len(paths) <= 1:
            return paths

        used = set()
        result = []

        for i, p1 in enumerate(paths):
            if i in used:
                continue

            to_merge = [p1]
            used.add(i)

            for j, p2 in enumerate(paths):
                if j in used:
                    continue
                if bboxes_overlap(p1.bbox, p2.bbox):
                    to_merge.append(p2)
                    used.add(j)

            if len(to_merge) == 1:
                result.append(to_merge[0])
            else:
                combined_d = " ".join(p.d for p in to_merge)
                merged_path = PathInfo(
                    index=to_merge[0].index,
                    d=combined_d,
                    fill=to_merge[0].fill,
                    stroke=to_merge[0].stroke,
                    opacity=to_merge[0].opacity,
                    original_attribs=to_merge[0].original_attribs,
                    transform=to_merge[0].transform,
                )
                merged_path.translate = to_merge[0].translate
                if merged_path.translate != (0.0, 0.0):
                    tx, ty = merged_path.translate
                    xmin, ymin, xmax, ymax = merged_path.bbox
                    merged_path.bbox = (
                        xmin + tx,
                        ymin + ty,
                        xmax + tx,
                        ymax + ty,
                    )
                    cx, cy = merged_path.centroid
                    merged_path.centroid = (cx + tx, cy + ty)
                result.append(merged_path)

        return result
