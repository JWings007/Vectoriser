"""Detect parent-child containment relationships between groups."""
from typing import List, Dict
from collections import defaultdict

from ..parsing.svg_parser import PathInfo
from ..utils.logger import get_logger

logger = get_logger(__name__)


class HierarchyDetector:
    """
    Detect which groups visually contain other groups.
    Uses actual geometric containment via Shapely when polygons are available,
    falling back to bounding box checks.
    """

    def __init__(self, config: dict):
        hier_cfg = config.get("hierarchy", {})
        self.method = hier_cfg.get("containment_method", "geometric")
        self.area_ratio_threshold = hier_cfg.get("area_ratio_threshold", 1.1)

    def detect(self, groups: List[List[PathInfo]]) -> Dict:
        """
        Build hierarchy tree of groups.

        Returns:
            {
                "roots": [group_index, ...],
                "children": {parent_index: [child_index, ...]},
                "parent": {child_index: parent_index},
                "depth": {group_index: depth_int}
            }
        """
        n = len(groups)
        if n <= 1:
            return {
                "roots": list(range(n)),
                "children": {},
                "parent": {},
                "depth": {i: 0 for i in range(n)},
            }

        # Compute group bounding boxes and union polygons
        group_bboxes = []
        group_polygons = []
        group_areas = []

        for group in groups:
            xmin = min(p.bbox[0] for p in group)
            ymin = min(p.bbox[1] for p in group)
            xmax = max(p.bbox[2] for p in group)
            ymax = max(p.bbox[3] for p in group)
            group_bboxes.append((xmin, ymin, xmax, ymax))
            group_areas.append((xmax - xmin) * (ymax - ymin))

            # Try to build a union polygon from the group's paths
            if self.method == "geometric":
                union_poly = None
                for path in group:
                    if path.polygon is not None:
                        if union_poly is None:
                            union_poly = path.polygon
                        else:
                            try:
                                union_poly = union_poly.union(path.polygon)
                            except Exception:
                                pass
                group_polygons.append(union_poly)
            else:
                group_polygons.append(None)

        # Find containment relationships
        contains_map: Dict[int, List[int]] = defaultdict(list)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if group_areas[i] <= group_areas[j] * self.area_ratio_threshold:
                    continue

                if self._group_contains(
                    group_bboxes[i],
                    group_polygons[i],
                    group_bboxes[j],
                    group_polygons[j],
                ):
                    contains_map[i].append(j)

        # Resolve to find direct parents (remove transitive containment)
        parent = {}
        for child_idx in range(n):
            potential_parents = [
                p for p in range(n) if child_idx in contains_map.get(p, [])
            ]

            if not potential_parents:
                continue

            direct_parent = min(potential_parents, key=lambda p: group_areas[p])
            parent[child_idx] = direct_parent

        # Build children map from parent map
        children = defaultdict(list)
        for child, par in parent.items():
            children[par].append(child)

        # Roots are groups with no parent
        roots = [i for i in range(n) if i not in parent]

        # Compute depths
        depth = {}

        def compute_depth(node, d):
            depth[node] = d
            for child in children.get(node, []):
                compute_depth(child, d + 1)

        for root in roots:
            compute_depth(root, 0)

        for i in range(n):
            if i not in depth:
                depth[i] = 0
                if i not in roots:
                    roots.append(i)

        logger.info(
            f"Hierarchy: {len(roots)} root groups, "
            f"max depth {max(depth.values()) if depth else 0}"
        )

        return {
            "roots": roots,
            "children": dict(children),
            "parent": parent,
            "depth": depth,
        }

    def _group_contains(self, bbox_outer, poly_outer, bbox_inner, poly_inner) -> bool:
        """Check if outer group geometrically contains inner group."""
        o = bbox_outer
        i = bbox_inner
        if not (o[0] <= i[0] and o[1] <= i[1] and o[2] >= i[2] and o[3] >= i[3]):
            return False

        if poly_outer is not None and poly_inner is not None:
            try:
                return poly_outer.contains(poly_inner)
            except Exception:
                pass

        return True
