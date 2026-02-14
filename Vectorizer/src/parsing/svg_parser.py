"""SVG parsing and PathInfo data extraction."""
from dataclasses import dataclass, field
import re
from typing import List, Optional, Tuple
from lxml import etree
from shapely.geometry import Polygon
from shapely import affinity

from ..utils.geometry import (
    path_to_bbox,
    bbox_area,
    bbox_centroid,
    bbox_aspect_ratio,
    path_to_polygon,
    path_complexity,
    path_curve_ratio,
)
from ..utils.color import hex_to_lab
from ..utils.logger import get_logger

logger = get_logger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"
NSMAP = {"svg": SVG_NS}


@dataclass
class PathInfo:
    """Complete information about a single SVG path element."""

    index: int
    d: str
    fill: str = "#000000"
    stroke: str = "none"
    opacity: float = 1.0
    original_attribs: dict = field(default_factory=dict)
    transform: Optional[str] = None
    translate: Tuple[float, float] = (0.0, 0.0)

    # Computed fields (set in __post_init__)
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    centroid: Tuple[float, float] = (0.0, 0.0)
    area: float = 0.0
    aspect_ratio: float = 1.0
    complexity: int = 0
    curve_ratio: float = 0.0
    lab_color: Optional[Tuple[float, float, float]] = None
    polygon: Optional[Polygon] = None

    # Assigned during grouping
    group_id: int = -1

    def __post_init__(self):
        self.bbox = path_to_bbox(self.d)
        self.area = bbox_area(self.bbox)
        self.centroid = bbox_centroid(self.bbox)
        self.aspect_ratio = bbox_aspect_ratio(self.bbox)
        self.complexity = path_complexity(self.d)
        self.curve_ratio = path_curve_ratio(self.d)
        self.lab_color = hex_to_lab(self.fill)


class SVGParser:
    """Parse raw SVG string into structured PathInfo objects."""

    def __init__(self, config: dict):
        self.min_area = config.get("analysis", {}).get("min_path_area", 2.0)

    def parse(self, svg_string: str) -> Tuple[dict, List[PathInfo]]:
        """
        Parse SVG string and extract paths.

        Returns:
            Tuple of (svg_metadata dict, list of PathInfo objects)
        """
        root = etree.fromstring(svg_string.encode("utf-8"))

        # Extract SVG-level metadata
        metadata = {
            "width": root.get("width", ""),
            "height": root.get("height", ""),
            "viewBox": root.get("viewBox", ""),
            "xmlns": root.get("xmlns", SVG_NS),
        }

        # Find all path elements
        paths = []
        path_elements = root.xpath("//svg:path", namespaces=NSMAP)

        # Fallback: try without namespace
        if not path_elements:
            path_elements = root.xpath("//*[local-name()='path']")

        logger.info(f"Found {len(path_elements)} raw path elements")

        for idx, elem in enumerate(path_elements):
            d = elem.get("d", "").strip()
            if not d:
                continue

            fill = elem.get("fill", "#000000")
            stroke = elem.get("stroke", "none")
            opacity = float(elem.get("opacity", "1.0"))
            transform = elem.get("transform")

            # Collect all original attributes
            attribs = dict(elem.attrib)

            path_info = PathInfo(
                index=idx,
                d=d,
                fill=fill,
                stroke=stroke,
                opacity=opacity,
                original_attribs=attribs,
                transform=transform,
            )

            if transform:
                tx, ty = self._parse_translate(transform)
                if tx or ty:
                    path_info.translate = (tx, ty)
                    xmin, ymin, xmax, ymax = path_info.bbox
                    path_info.bbox = (xmin + tx, ymin + ty, xmax + tx, ymax + ty)
                    cx, cy = path_info.centroid
                    path_info.centroid = (cx + tx, cy + ty)

            # Filter by minimum area
            if path_info.area < self.min_area:
                continue

            paths.append(path_info)

        logger.info(
            f"Extracted {len(paths)} valid paths (min area: {self.min_area})"
        )
        return metadata, paths

    def compute_polygons(self, paths: List[PathInfo], num_samples: int = 64) -> None:
        """
        Pre-compute shapely polygons for paths that need containment testing.
        This is expensive, so it's a separate step.
        """
        logger.info("Computing path polygons for containment analysis...")
        computed = 0
        for path in paths:
            poly = path_to_polygon(path.d, num_samples)
            if poly is not None:
                tx, ty = path.translate
                if tx or ty:
                    poly = affinity.translate(poly, xoff=tx, yoff=ty)
                path.polygon = poly
                computed += 1
        logger.info(f"Computed {computed}/{len(paths)} polygons successfully")

    @staticmethod
    def _parse_translate(transform: str) -> Tuple[float, float]:
        """Parse translate(x[,y]) from an SVG transform string."""
        match = re.search(
            r"translate\(\s*([-\d.+eE]+)(?:[,\s]+([-\d.+eE]+))?\s*\)",
            transform,
        )
        if not match:
            return (0.0, 0.0)
        try:
            tx = float(match.group(1))
            ty = float(match.group(2)) if match.group(2) is not None else 0.0
            return (tx, ty)
        except ValueError:
            return (0.0, 0.0)
