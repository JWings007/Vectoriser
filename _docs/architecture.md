Complete Implementation Plan: Clean SVG Vectorizer
Project Overview
A Python-based tool that converts raster images to clean, well-organized SVGs with intelligent grouping, hierarchy detection, and path optimization. Designed to serve a web visualizer page.

Folder Structure
svg-vectorizer/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│ └── defaults.yaml
├── src/
│ ├── **init**.py
│ ├── main.py # Entry point & CLI
│ ├── vectorizer.py # Main orchestrator class
│ ├── tracing/
│ │ ├── **init**.py
│ │ └── tracer.py # vtracer integration
│ ├── parsing/
│ │ ├── **init**.py
│ │ └── svg_parser.py # SVG parsing & PathInfo
│ ├── preprocessing/
│ │ ├── **init**.py
│ │ ├── color_quantizer.py # Color reduction
│ │ └── image_cleaner.py # Noise removal, contrast
│ ├── analysis/
│ │ ├── **init**.py
│ │ ├── feature_extractor.py # Build feature vectors
│ │ └── color_analysis.py # CIELAB color utilities
│ ├── grouping/
│ │ ├── **init**.py
│ │ ├── clusterer.py # DBSCAN clustering
│ │ ├── graph_grouper.py # Adjacency graph grouping
│ │ └── merger.py # Group merging logic
│ ├── hierarchy/
│ │ ├── **init**.py
│ │ └── containment.py # Geometric containment
│ ├── optimization/
│ │ ├── **init**.py
│ │ ├── path_simplifier.py # Reduce points, precision
│ │ ├── path_merger.py # Merge adjacent same-fill
│ │ └── svg_optimizer.py # Final SVG cleanup
│ ├── output/
│ │ ├── **init**.py
│ │ └── svg_generator.py # Build final SVG
│ └── utils/
│ ├── **init**.py
│ ├── geometry.py # Shared geometry helpers
│ ├── color.py # Color conversion helpers
│ └── logger.py # Logging setup
├── api/
│ ├── **init**.py
│ └── server.py # Flask/FastAPI web server
├── tests/
│ ├── **init**.py
│ ├── test_parser.py
│ ├── test_clustering.py
│ ├── test_hierarchy.py
│ ├── test_optimization.py
│ └── fixtures/
│ ├── simple_logo.png
│ ├── icon.png
│ └── illustration.png
└── examples/
├── basic_usage.py
└── batch_convert.py

requirements.txt
vtracer==0.6.3
opencv-python==4.9.0.80
numpy==1.26.4
scikit-learn==1.4.0
scipy==1.12.0
lxml==5.1.0
svgpathtools==1.6.1
shapely==2.0.2
Pillow==10.2.0
scikit-image==0.22.0
pyyaml==6.0.1
networkx==3.2.1
fastapi==0.109.0
uvicorn==0.27.0
python-multipart==0.0.6

config/defaults.yaml
yamlDownloadCopy codetracing:
colormode: "color"
hierarchical: "stacked"
mode: "spline"
filter_speckle: 4
color_precision: 6
layer_difference: 16
corner_threshold: 60
length_threshold: 10.0
splice_threshold: 45
path_precision: 8

preprocessing:
enabled: true
denoise: true
denoise_strength: 10
sharpen: false
quantize_colors: false
quantize_n_colors: 16

analysis:
min_path_area: 2.0
color_space: "cielab"

clustering:
method: "dbscan" # "dbscan" or "graph"
eps: 0.5
min_samples: 1
feature_weights:
position: 1.0
color: 2.0
area: 0.5
complexity: 0.3
aspect_ratio: 0.3

grouping:
merge_enabled: true
proximity_threshold: 50.0
color_tolerance: 25.0 # CIELAB Delta-E
min_group_size: 1

hierarchy:
enabled: true
containment_method: "geometric" # "geometric" or "bbox"
area_ratio_threshold: 1.1

optimization:
coordinate_precision: 2
remove_degenerate: true
merge_adjacent: true
simplify_paths: true
simplify_tolerance: 0.5

output:
indent: true
indent_spaces: 2
add_metadata: true
add_group_ids: true
promote_common_attributes: true

presets:
logo:
tracing:
color_precision: 4
filter_speckle: 8
clustering:
eps: 0.4
feature_weights:
color: 3.0
grouping:
color_tolerance: 15.0

illustration:
tracing:
color_precision: 6
clustering:
eps: 0.5
grouping:
color_tolerance: 30.0

photograph:
preprocessing:
quantize_colors: true
quantize_n_colors: 24
tracing:
color_precision: 8
clustering:
eps: 0.6
grouping:
color_tolerance: 40.0

Full Implementation
src/**init**.py
pythonDownloadCopy code"""SVG Vectorizer - Clean SVG generation from raster images."""
**version** = "0.1.0"
src/utils/logger.py
pythonDownloadCopy code"""Centralized logging configuration."""
import logging
import sys

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger

src/utils/color.py
pythonDownloadCopy code"""Color conversion and comparison utilities using CIELAB color space."""
import numpy as np
from typing import Tuple, Optional

def hex_to_rgb(hex_color: str) -> Optional[Tuple[int, int, int]]:
"""Convert hex color string to RGB tuple."""
hex_color = hex_color.strip()

    if not hex_color.startswith("#"):
        return None

    hex_color = hex_color[1:]

    # Handle shorthand like #FFF
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)

    if len(hex_color) != 6:
        return None

    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    except ValueError:
        return None

def rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
"""
Convert RGB to CIELAB color space.
Uses D65 illuminant reference white.
""" # Normalize to 0-1
r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0

    # Linearize (inverse sRGB companding)
    def linearize(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r_lin = linearize(r_n)
    g_lin = linearize(g_n)
    b_lin = linearize(b_n)

    # RGB to XYZ (sRGB D65)
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    # D65 reference white
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    # XYZ to Lab
    def f(t):
        delta = 6.0 / 29.0
        if t > delta**3:
            return t ** (1.0 / 3.0)
        else:
            return t / (3.0 * delta**2) + 4.0 / 29.0

    l = 116.0 * f(y) - 16.0
    a = 500.0 * (f(x) - f(y))
    b_val = 200.0 * (f(y) - f(z))

    return (l, a, b_val)

def hex_to_lab(hex_color: str) -> Optional[Tuple[float, float, float]]:
"""Convert hex color directly to CIELAB."""
rgb = hex_to_rgb(hex_color)
if rgb is None:
return None
return rgb_to_lab(\*rgb)

def delta_e(lab1: Tuple[float, float, float],
lab2: Tuple[float, float, float]) -> float:
"""
Calculate CIE76 Delta-E color difference.
Values roughly mean:
< 1.0 : imperceptible
1-2 : barely perceptible
2-10 : perceptible at close look
11-49 : colors are more similar than different
100 : exact opposite colors
"""
return np.sqrt(
(lab2[0] - lab1[0]) ** 2 + (lab2[1] - lab1[1]) ** 2 + (lab2[2] - lab1[2]) \*\* 2
)

def colors_similar(color1: str, color2: str, tolerance: float = 25.0) -> bool:
"""Check if two hex colors are perceptually similar using CIELAB Delta-E."""
if color1 == color2:
return True

    lab1 = hex_to_lab(color1)
    lab2 = hex_to_lab(color2)

    if lab1 is None or lab2 is None:
        return color1 == color2

    return delta_e(lab1, lab2) < tolerance

src/utils/geometry.py
pythonDownloadCopy code"""Shared geometry helper functions."""
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from svgpathtools import parse_path, Line, CubicBezier, QuadraticBezier, Arc
from typing import Tuple, Optional, List

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
return max(0.0, (xmax - xmin) \* (ymax - ymin))

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
"""Minimum distance between two bounding boxes.""" # If overlapping, distance is 0
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
                # Take the largest polygon
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
            1 for seg in path
            if isinstance(seg, (CubicBezier, QuadraticBezier, Arc))
        )
        return curved / len(path)
    except Exception:
        return 0.0

src/tracing/tracer.py
pythonDownloadCopy code"""vtracer integration for initial image-to-SVG conversion."""
import vtracer
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger(**name**)

class Tracer:
"""Wraps vtracer for raster-to-SVG conversion."""

    def __init__(self, config: dict):
        self.config = config

    def trace(self, image_path: str) -> str:
        """
        Trace a raster image and return raw SVG string.

        Args:
            image_path: Path to input PNG/JPG image.

        Returns:
            SVG content as a string.
        """
        image_path = str(Path(image_path).resolve())
        logger.info(f"Tracing image: {image_path}")

        cfg = self.config.get("tracing", {})

        svg_string = vtracer.convert_image_to_svg_py(
            image_path,
            colormode=cfg.get("colormode", "color"),
            hierarchical=cfg.get("hierarchical", "stacked"),
            mode=cfg.get("mode", "spline"),
            filter_speckle=cfg.get("filter_speckle", 4),
            color_precision=cfg.get("color_precision", 6),
            layer_difference=cfg.get("layer_difference", 16),
            corner_threshold=cfg.get("corner_threshold", 60),
            length_threshold=cfg.get("length_threshold", 10.0),
            splice_threshold=cfg.get("splice_threshold", 45),
            path_precision=cfg.get("path_precision", 8),
        )

        logger.info(f"Tracing complete. SVG length: {len(svg_string)} chars")
        return svg_string

src/parsing/svg_parser.py
pythonDownloadCopy code"""SVG parsing and PathInfo data extraction."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from lxml import etree
from shapely.geometry import Polygon

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

logger = get_logger(**name**)

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

            # Collect all original attributes
            attribs = dict(elem.attrib)

            path_info = PathInfo(
                index=idx,
                d=d,
                fill=fill,
                stroke=stroke,
                opacity=opacity,
                original_attribs=attribs,
            )

            # Filter by minimum area
            if path_info.area < self.min_area:
                continue

            paths.append(path_info)

        logger.info(f"Extracted {len(paths)} valid paths (min area: {self.min_area})")
        return metadata, paths

    def compute_polygons(self, paths: List[PathInfo],
                         num_samples: int = 64) -> None:
        """
        Pre-compute shapely polygons for paths that need containment testing.
        This is expensive, so it's a separate step.
        """
        logger.info("Computing path polygons for containment analysis...")
        computed = 0
        for path in paths:
            poly = path_to_polygon(path.d, num_samples)
            if poly is not None:
                path.polygon = poly
                computed += 1
        logger.info(f"Computed {computed}/{len(paths)} polygons successfully")

src/preprocessing/color_quantizer.py
pythonDownloadCopy code"""Color quantization for images with gradients or many colors."""
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from ..utils.logger import get_logger

logger = get_logger(**name**)

class ColorQuantizer:
"""Reduce number of colors in an image before vectorization."""

    def __init__(self, config: dict):
        pre_cfg = config.get("preprocessing", {})
        self.enabled = pre_cfg.get("quantize_colors", False)
        self.n_colors = pre_cfg.get("quantize_n_colors", 16)

    def quantize(self, image_path: str, output_path: str) -> str:
        """
        Quantize image colors using K-means clustering.

        Args:
            image_path: Path to input image.
            output_path: Path to save quantized image.

        Returns:
            Path to the quantized image (or original if quantization disabled).
        """
        if not self.enabled:
            return image_path

        logger.info(f"Quantizing colors to {self.n_colors} clusters...")

        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return image_path

        h, w, c = img.shape
        pixels = img.reshape(-1, 3).astype(np.float32)

        kmeans = MiniBatchKMeans(
            n_clusters=self.n_colors,
            random_state=42,
            batch_size=1024,
            n_init=3,
        )
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_.astype(np.uint8)

        quantized = centers[labels].reshape(h, w, c)
        cv2.imwrite(output_path, quantized)

        logger.info(f"Quantized image saved to: {output_path}")
        return output_path

src/preprocessing/image_cleaner.py
pythonDownloadCopy code"""Image preprocessing: denoising, sharpening, contrast."""
import cv2
import numpy as np
from ..utils.logger import get_logger

logger = get_logger(**name**)

class ImageCleaner:
"""Clean up input image before vectorization."""

    def __init__(self, config: dict):
        pre_cfg = config.get("preprocessing", {})
        self.enabled = pre_cfg.get("enabled", True)
        self.denoise = pre_cfg.get("denoise", True)
        self.denoise_strength = pre_cfg.get("denoise_strength", 10)
        self.sharpen = pre_cfg.get("sharpen", False)

    def clean(self, image_path: str, output_path: str) -> str:
        """
        Apply preprocessing to input image.

        Returns:
            Path to cleaned image (or original if preprocessing disabled).
        """
        if not self.enabled:
            return image_path

        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return image_path

        if self.denoise:
            logger.info("Applying denoising...")
            img = cv2.fastNlMeansDenoisingColored(
                img, None, self.denoise_strength, self.denoise_strength, 7, 21
            )

        if self.sharpen:
            logger.info("Applying sharpening...")
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])
            img = cv2.filter2D(img, -1, kernel)

        cv2.imwrite(output_path, img)
        logger.info(f"Cleaned image saved to: {output_path}")
        return output_path

src/analysis/feature_extractor.py
pythonDownloadCopy code"""Build rich feature vectors for path clustering."""
import numpy as np
from typing import List
from ..parsing.svg_parser import PathInfo
from ..utils.logger import get_logger

logger = get_logger(**name**)

class FeatureExtractor:
"""Extract multi-dimensional feature vectors from paths for clustering."""

    def __init__(self, config: dict):
        cluster_cfg = config.get("clustering", {})
        self.weights = cluster_cfg.get("feature_weights", {})

    def extract(self, paths: List[PathInfo],
                image_width: float = 1.0,
                image_height: float = 1.0) -> np.ndarray:
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
                l_norm = p.lab_color[0] / 100.0          # L is 0-100
                a_norm = (p.lab_color[1] + 128) / 256.0  # a is roughly -128 to 128
                b_norm = (p.lab_color[2] + 128) / 256.0  # b is roughly -128 to 128
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

            features.append([
                norm_cx, norm_cy,
                l_norm, a_norm, b_norm,
                log_area, ar, comp, cr,
            ])

        features = np.array(features, dtype=np.float64)

        # Apply weights
        w_pos = self.weights.get("position", 1.0)
        w_color = self.weights.get("color", 2.0)
        w_area = self.weights.get("area", 0.5)
        w_complexity = self.weights.get("complexity", 0.3)
        w_ar = self.weights.get("aspect_ratio", 0.3)

        weight_vector = np.array([
            w_pos, w_pos,                    # cx, cy
            w_color, w_color, w_color,       # L, a, b
            w_area,                          # log_area
            w_ar,                            # aspect_ratio
            w_complexity, w_complexity,      # complexity, curve_ratio
        ])

        features *= weight_vector

        logger.info(
            f"Extracted {features.shape[1]} features for {features.shape[0]} paths"
        )
        return features

src/analysis/color_analysis.py
pythonDownloadCopy code"""Color analysis utilities for grouping decisions."""
import numpy as np
from collections import Counter
from typing import List, Dict
from ..parsing.svg_parser import PathInfo
from ..utils.color import hex_to_lab, delta_e
from ..utils.logger import get_logger

logger = get_logger(**name**)

def dominant*colors(paths: List[PathInfo], top_n: int = 10) -> List[str]:
"""Find the most common fill colors in a set of paths."""
counter = Counter(p.fill for p in paths)
return [color for color, * in counter.most_common(top_n)]

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
    variance = np.mean([
        delta_e(mean_lab, lab) for lab in lab_values
    ])

    fills = [p.fill for p in paths]
    dominant = Counter(fills).most_common(1)[0][0]

    return {
        "mean_lab": mean_lab,
        "variance": variance,
        "dominant_fill": dominant,
        "unique_colors": len(set(fills)),
    }

src/grouping/clusterer.py
pythonDownloadCopy code"""DBSCAN-based path clustering."""
import numpy as np
from collections import defaultdict
from typing import List
from sklearn.cluster import DBSCAN

from ..parsing.svg_parser import PathInfo
from ..analysis.feature_extractor import FeatureExtractor
from ..utils.logger import get_logger

logger = get_logger(**name**)

class PathClusterer:
"""Group paths into clusters using DBSCAN on feature vectors."""

    def __init__(self, config: dict):
        self.config = config
        cluster_cfg = config.get("clustering", {})
        self.eps = cluster_cfg.get("eps", 0.5)
        self.min_samples = cluster_cfg.get("min_samples", 1)
        self.feature_extractor = FeatureExtractor(config)

    def cluster(self, paths: List[PathInfo],
                image_width: float = 1.0,
                image_height: float = 1.0) -> List[List[PathInfo]]:
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
        features = self.feature_extractor.extract(
            paths, image_width, image_height
        )

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
        groups.sort(key=lambda g: (
            np.mean([p.centroid[1] for p in g]),  # Y first (top to bottom)
            np.mean([p.centroid[0] for p in g]),  # Then X (left to right)
        ))

        logger.info(
            f"DBSCAN produced {len(groups)} clusters "
            f"from {len(paths)} paths (eps={self.eps})"
        )

        return groups

src/grouping/graph_grouper.py
pythonDownloadCopy code"""
Graph-based path grouping using spatial adjacency.
Alternative to pure DBSCAN — builds an adjacency graph and uses
community detection.
"""
import numpy as np
import networkx as nx
from typing import List
from collections import defaultdict

from ..parsing.svg_parser import PathInfo
from ..utils.geometry import bbox_distance
from ..utils.color import delta_e
from ..utils.logger import get_logger

logger = get_logger(**name**)

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
        for i, path in enumerate(paths):
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
            # Fallback: use connected components
            communities = list(nx.connected_components(G))

        # Convert to path groups
        groups = []
        for community in communities:
            group = [paths[i] for i in sorted(community)]
            groups.append(group)

        # Sort groups by position
        groups.sort(key=lambda g: (
            np.mean([p.centroid[1] for p in g]),
            np.mean([p.centroid[0] for p in g]),
        ))

        logger.info(
            f"Graph grouping produced {len(groups)} groups "
            f"from {len(paths)} paths"
        )

        return groups

src/grouping/merger.py
pythonDownloadCopy code"""Merge groups that are very similar based on group-level statistics."""
import numpy as np
from typing import List
from ..parsing.svg_parser import PathInfo
from ..analysis.color_analysis import group_color_stats
from ..utils.color import delta_e
from ..utils.geometry import bbox_distance
from ..utils.logger import get_logger

logger = get_logger(**name**)

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

                    if self._should_merge(
                        stats[i], stats[j], bboxes[i], bboxes[j]
                    ):
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

    def _should_merge(self, stats1: dict, stats2: dict,
                      bbox1: tuple, bbox2: tuple) -> bool:
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

src/hierarchy/containment.py
pythonDownloadCopy code"""Detect parent-child containment relationships between groups."""
from typing import List, Dict, Set
from collections import defaultdict

from ..parsing.svg_parser import PathInfo
from ..utils.geometry import path_to_polygon
from ..utils.logger import get_logger

logger = get_logger(**name**)

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
        # For each pair, check if the larger group contains the smaller
        contains_map: Dict[int, List[int]] = defaultdict(list)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Only check if i is larger than j
                if group_areas[i] <= group_areas[j] * self.area_ratio_threshold:
                    continue

                if self._group_contains(
                    group_bboxes[i], group_polygons[i],
                    group_bboxes[j], group_polygons[j],
                ):
                    contains_map[i].append(j)

        # Resolve to find direct parents (remove transitive containment)
        # If A contains B and B contains C, then A should NOT be direct parent of C
        parent = {}
        for child_idx in range(n):
            # Find all groups that contain this child
            potential_parents = [
                p for p in range(n)
                if child_idx in contains_map.get(p, [])
            ]

            if not potential_parents:
                continue

            # Direct parent is the smallest container
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

        # Any ungrouped nodes
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

    def _group_contains(self, bbox_outer, poly_outer,
                        bbox_inner, poly_inner) -> bool:
        """Check if outer group geometrically contains inner group."""
        # Quick reject: bounding box check first
        o = bbox_outer
        i = bbox_inner
        if not (o[0] <= i[0] and o[1] <= i[1] and
                o[2] >= i[2] and o[3] >= i[3]):
            return False

        # If we have polygons, use geometric containment
        if poly_outer is not None and poly_inner is not None:
            try:
                return poly_outer.contains(poly_inner)
            except Exception:
                pass

        # Fallback: bounding box containment is enough
        return True

src/optimization/path_simplifier.py
pythonDownloadCopy code"""Simplify and clean individual SVG paths."""
from svgpathtools import parse_path, Path, Line, CubicBezier, QuadraticBezier
from typing import List
from ..parsing.svg_parser import PathInfo
from ..utils.logger import get_logger

logger = get_logger(**name**)

class PathSimplifier:
"""Reduce path complexity while maintaining visual fidelity."""

    def __init__(self, config: dict):
        opt_cfg = config.get("optimization", {})
        self.coordinate_precision = opt_cfg.get("coordinate_precision", 2)
        self.remove_degenerate = opt_cfg.get("remove_degenerate", True)
        self.simplify_paths = opt_cfg.get("simplify_paths", True)
        self.simplify_tolerance = opt_cfg.get("simplify_tolerance", 0.5)

    def simplify_all(self, paths: List[PathInfo]) -> List[PathInfo]:
        """Apply simplification to all paths."""
        simplified_count = 0
        for path_info in paths:
            original_d = path_info.d
            new_d = self._simplify_path_d(original_d)
            if new_d != original_d:
                path_info.d = new_d
                simplified_count += 1

        logger.info(f"Simplified {simplified_count}/{len(paths)} paths")
        return paths

    def _simplify_path_d(self, d: str) -> str:
        """Simplify a single path d string."""
        try:
            path = parse_path(d)
        except Exception:
            return d

        if len(path) == 0:
            return d

        new_segments = []
        for seg in path:
            # Remove degenerate segments (near-zero length)
            if self.remove_degenerate:
                length = seg.length()
                if length < self.simplify_tolerance:
                    continue

            new_segments.append(seg)

        if not new_segments:
            return d

        new_path = Path(*new_segments)

        # Round coordinates to desired precision
        return self._round_path_d(new_path.d(), self.coordinate_precision)

    def _round_path_d(self, d: str, precision: int) -> str:
        """
        Round all numeric values in a path d string to given decimal places.
        This is a string-level operation for maximum control.
        """
        import re

        def round_match(match):
            value = float(match.group())
            rounded = round(value, precision)
            # Remove trailing zeros
            if precision == 0:
                return str(int(rounded))
            formatted = f"{rounded:.{precision}f}"
            # Strip unnecessary trailing zeros but keep at least one decimal
            formatted = formatted.rstrip("0").rstrip(".")
            return formatted

        return re.sub(r"-?\d+\.?\d*", round_match, d)

src/optimization/path_merger.py
pythonDownloadCopy code"""Merge adjacent paths that share the same fill color."""
from typing import List
from ..parsing.svg_parser import PathInfo
from ..utils.geometry import bboxes_overlap
from ..utils.logger import get_logger

logger = get_logger(**name**)

class PathMerger:
"""
Merge paths within a group that have the same fill and are adjacent.
This reduces SVG element count without visual change.
"""

    def __init__(self, config: dict):
        opt_cfg = config.get("optimization", {})
        self.merge_adjacent = opt_cfg.get("merge_adjacent", True)

    def merge_within_groups(
        self, groups: List[List[PathInfo]]
    ) -> List[List[PathInfo]]:
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

        # Group by fill color
        by_fill = {}
        for path in group:
            key = path.fill
            if key not in by_fill:
                by_fill[key] = []
            by_fill[key].append(path)

        result = []
        for fill, same_color_paths in by_fill.items():
            if len(same_color_paths) == 1:
                result.append(same_color_paths[0])
                continue

            # Among same-color paths, merge overlapping/adjacent ones
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

            # Collect all paths that should merge with p1
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
                # Combine d strings
                combined_d = " ".join(p.d for p in to_merge)
                merged_path = PathInfo(
                    index=to_merge[0].index,
                    d=combined_d,
                    fill=to_merge[0].fill,
                    stroke=to_merge[0].stroke,
                    opacity=to_merge[0].opacity,
                    original_attribs=to_merge[0].original_attribs,
                )
                result.append(merged_path)

        return result

src/optimization/svg_optimizer.py
pythonDownloadCopy code"""Final SVG-level optimizations."""
from ..utils.logger import get_logger

logger = get_logger(**name**)

class SVGOptimizer:
"""Apply final optimizations to the SVG output string."""

    def __init__(self, config: dict):
        self.config = config

    def optimize(self, svg_string: str) -> str:
        """Apply text-level SVG optimizations."""
        # Remove unnecessary whitespace in path data
        svg_string = self._compact_whitespace(svg_string)

        # Remove empty groups
        svg_string = self._remove_empty_groups(svg_string)

        return svg_string

    def _compact_whitespace(self, svg: str) -> str:
        """Reduce excessive whitespace in path d attributes."""
        import re
        # Collapse multiple spaces in d attributes
        svg = re.sub(r'd="([^"]*)"', lambda m: f'd="{" ".join(m.group(1).split())}"', svg)
        return svg

    def _remove_empty_groups(self, svg: str) -> str:
        """Remove <g> elements that contain no children."""
        import re
        # Remove empty <g></g> or <g/> tags
        svg = re.sub(r"<g[^>]*/>\s*", "", svg)
        svg = re.sub(r"<g[^>]*>\s*</g>\s*", "", svg)
        return svg

src/output/svg_generator.py
pythonDownloadCopy code"""Generate the final clean, grouped SVG output."""
from lxml import etree
from typing import List, Dict, Optional
from collections import Counter

from ..parsing.svg_parser import PathInfo
from ..utils.logger import get_logger

logger = get_logger(**name**)

SVG_NS = "http://www.w3.org/2000/svg"

class SVGGenerator:
"""Build a clean, hierarchically organized SVG document."""

    def __init__(self, config: dict):
        out_cfg = config.get("output", {})
        self.indent = out_cfg.get("indent", True)
        self.indent_spaces = out_cfg.get("indent_spaces", 2)
        self.add_metadata = out_cfg.get("add_metadata", True)
        self.add_group_ids = out_cfg.get("add_group_ids", True)
        self.promote_common_attrs = out_cfg.get("promote_common_attributes", True)

    def generate(
        self,
        groups: List[List[PathInfo]],
        hierarchy: Dict,
        svg_metadata: dict,
    ) -> str:
        """
        Generate final SVG string.

        Args:
            groups: List of path groups.
            hierarchy: Hierarchy dict from HierarchyDetector.
            svg_metadata: Original SVG attributes (viewBox, width, height).

        Returns:
            Clean SVG string.
        """
        # Create root SVG element
        nsmap = {None: SVG_NS}
        root = etree.Element("{%s}svg" % SVG_NS, nsmap=nsmap)

        # Set SVG attributes
        if svg_metadata.get("viewBox"):
            root.set("viewBox", svg_metadata["viewBox"])
        if svg_metadata.get("width"):
            root.set("width", svg_metadata["width"])
        if svg_metadata.get("height"):
            root.set("height", svg_metadata["height"])

        # Add metadata comment
        if self.add_metadata:
            comment = etree.Comment(
                " Generated by SVG Vectorizer | "
                f"{sum(len(g) for g in groups)} paths in "
                f"{len(groups)} groups "
            )
            root.append(comment)

        # Build SVG tree following hierarchy
        roots = hierarchy.get("roots", list(range(len(groups))))
        children_map = hierarchy.get("children", {})

        for root_idx in roots:
            self._add_group_recursive(
                root, groups, root_idx, children_map
            )

        # Serialize
        tree = etree.ElementTree(root)

        if self.indent:
            etree.indent(tree, space=" " * self.indent_spaces)

        svg_string = etree.tostring(
            root,
            pretty_print=self.indent,
            xml_declaration=True,
            encoding="unicode",
        )

        logger.info(f"Generated SVG: {len(svg_string)} characters")
        return svg_string

    def _add_group_recursive(
        self,
        parent_elem: etree._Element,
        groups: List[List[PathInfo]],
        group_idx: int,
        children_map: Dict,
    ):
        """Recursively add a group and its children to the SVG tree."""
        if group_idx >= len(groups):
            return

        group = groups[group_idx]
        children = children_map.get(group_idx, [])

        if len(group) == 1 and not children:
            # Single path, no children — add directly
            self._add_path_element(parent_elem, group[0])
            return

        # Create <g> element
        g_elem = etree.SubElement(parent_elem, "{%s}g" % SVG_NS)

        if self.add_group_ids:
            g_elem.set("id", f"group_{group_idx}")

        # Promote common fill to group level
        if self.promote_common_attrs and group:
            common_fill = self._get_common_attribute(group, "fill")
            if common_fill:
                g_elem.set("fill", common_fill)

            common_opacity = self._get_common_attribute(group, "opacity")
            if common_opacity and common_opacity != "1.0":
                g_elem.set("opacity", common_opacity)

        # Add paths
        for path in group:
            self._add_path_element(g_elem, path, promoted_fill=common_fill if self.promote_common_attrs else None)

        # Add child groups
        for child_idx in children:
            self._add_group_recursive(g_elem, groups, child_idx, children_map)

    def _add_path_element(
        self,
        parent: etree._Element,
        path_info: PathInfo,
        promoted_fill: Optional[str] = None,
    ):
        """Add a single path element."""
        elem = etree.SubElement(parent, "{%s}path" % SVG_NS)
        elem.set("d", path_info.d)

        # Only set fill if not promoted to parent group
        if promoted_fill is None or path_info.fill != promoted_fill:
            elem.set("fill", path_info.fill)

        if path_info.stroke != "none":
            elem.set("stroke", path_info.stroke)

        if path_info.opacity != 1.0:
            elem.set("opacity", str(path_info.opacity))

    @staticmethod
    def _get_common_attribute(group: List[PathInfo], attr: str) -> Optional[str]:
        """
        Check if all paths in a group share the same value for an attribute.
        Returns the common value, or None if values differ.
        """
        values = []
        for p in group:
            if attr == "fill":
                values.append(p.fill)
            elif attr == "opacity":
                values.append(str(p.opacity))
            elif attr == "stroke":
                values.append(p.stroke)

        if not values:
            return None

        counter = Counter(values)
        most_common_val, most_common_count = counter.most_common(1)[0]

        # If ALL paths share the value, promote it
        if most_common_count == len(values):
            return most_common_val

        return None

src/vectorizer.py
pythonDownloadCopy code"""Main orchestrator that ties the full pipeline together."""
import tempfile
import os
from pathlib import Path
import yaml

from .tracing.tracer import Tracer
from .parsing.svg_parser import SVGParser
from .preprocessing.color_quantizer import ColorQuantizer
from .preprocessing.image_cleaner import ImageCleaner
from .grouping.clusterer import PathClusterer
from .grouping.graph_grouper import GraphGrouper
from .grouping.merger import GroupMerger
from .hierarchy.containment import HierarchyDetector
from .optimization.path_simplifier import PathSimplifier
from .optimization.path_merger import PathMerger
from .optimization.svg_optimizer import SVGOptimizer
from .output.svg_generator import SVGGenerator
from .utils.logger import get_logger

logger = get_logger(**name**)

class SVGVectorizer:
"""
Complete SVG vectorization pipeline.

    Pipeline:
      1. Preprocess image (optional denoise, quantize)
      2. Trace image to raw SVG (vtracer)
      3. Parse SVG into structured PathInfo objects
      4. Cluster/group paths intelligently
      5. Merge over-fragmented groups
      6. Detect containment hierarchy
      7. Optimize paths (simplify, merge, round)
      8. Generate clean SVG output
    """

    def __init__(self, config: dict = None, config_path: str = None,
                 preset: str = None):
        """
        Initialize with config dict, YAML path, or preset name.

        Args:
            config: Direct config dictionary.
            config_path: Path to YAML config file.
            preset: Preset name ("logo", "illustration", "photograph").
        """
        self.config = self._load_config(config, config_path, preset)

        # Initialize pipeline components
        self.cleaner = ImageCleaner(self.config)
        self.quantizer = ColorQuantizer(self.config)
        self.tracer = Tracer(self.config)
        self.parser = SVGParser(self.config)
        self.simplifier = PathSimplifier(self.config)
        self.path_merger = PathMerger(self.config)
        self.group_merger = GroupMerger(self.config)
        self.hierarchy_detector = HierarchyDetector(self.config)
        self.svg_optimizer = SVGOptimizer(self.config)
        self.svg_generator = SVGGenerator(self.config)

        # Choose clustering method
        method = self.config.get("clustering", {}).get("method", "dbscan")
        if method == "graph":
            self.grouper = GraphGrouper(self.config)
        else:
            self.grouper = PathClusterer(self.config)

    def _load_config(self, config, config_path, preset) -> dict:
        """Load and merge configuration."""
        # Start with defaults
        default_path = Path(__file__).parent.parent / "config" / "defaults.yaml"
        if default_path.exists():
            with open(default_path) as f:
                base_config = yaml.safe_load(f)
        else:
            base_config = {}

        # Apply preset if specified
        if preset and "presets" in base_config:
            preset_config = base_config.get("presets", {}).get(preset, {})
            base_config = self._deep_merge(base_config, preset_config)

        # Load from YAML file if provided
        if config_path:
            with open(config_path) as f:
                file_config = yaml.safe_load(f)
            base_config = self._deep_merge(base_config, file_config)

        # Apply direct config overrides
        if config:
            base_config = self._deep_merge(base_config, config)

        return base_config

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Deep merge two dicts. Override takes precedence."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = SVGVectorizer._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def vectorize(self, input_path: str, output_path: str = None) -> str:
        """
        Full vectorization pipeline.

        Args:
            input_path: Path to input image (PNG, JPG).
            output_path: Path to save SVG. If None, returns SVG string only.

        Returns:
            SVG string.
        """
        logger.info(f"Starting vectorization: {input_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            # --- Step 1: Preprocess ---
            logger.info("Step 1/8: Preprocessing image...")
            cleaned_path = os.path.join(tmpdir, "cleaned.png")
            cleaned_path = self.cleaner.clean(input_path, cleaned_path)

            quantized_path = os.path.join(tmpdir, "quantized.png")
            quantized_path = self.quantizer.quantize(cleaned_path, quantized_path)

            # --- Step 2: Trace to raw SVG ---
            logger.info("Step 2/8: Tracing image to SVG...")
            raw_svg = self.tracer.trace(quantized_path)

            # --- Step 3: Parse SVG ---
            logger.info("Step 3/8: Parsing SVG paths...")
            svg_metadata, paths = self.parser.parse(raw_svg)

            if not paths:
                logger.warning("No paths extracted. Returning raw SVG.")
                if output_path:
                    with open(output_path, "w") as f:
                        f.write(raw_svg)
                return raw_svg

            logger.info(f"  Extracted {len(paths)} paths")

            # Parse image dimensions for feature normalization
            img_w, img_h = self._parse_dimensions(svg_metadata)

            # --- Step 4: Compute polygons for hierarchy ---
            if self.config.get("hierarchy", {}).get("enabled", True):
                logger.info("Step 4/8: Computing path polygons...")
                self.parser.compute_polygons(paths)
            else:
                logger.info("Step 4/8: Skipping polygon computation (hierarchy disabled)")

            # --- Step 5: Group paths ---
            logger.info("Step 5/8: Clustering paths into groups...")
            if isinstance(self.grouper, PathClusterer):
                groups = self.grouper.cluster(paths, img_w, img_h)
            else:
                groups = self.grouper.group(paths)

            logger.info(f"  Created {len(groups)} initial groups")

            # --- Step 6: Merge groups ---
            if self.config.get("grouping", {}).get("merge_enabled", True):
                logger.info("Step 6/8: Merging similar groups...")
                groups = self.group_merger.merge(groups)
                logger.info(f"  {len(groups)} groups after merging")
            else:
                logger.info("Step 6/8: Skipping group merging")

            # --- Step 7: Detect hierarchy ---
            logger.info("Step 7/8: Detecting containment hierarchy...")
            if self.config.get("hierarchy", {}).get("enabled", True):
                hierarchy = self.hierarchy_detector.detect(groups)
            else:
                hierarchy = {
                    "roots": list(range(len(groups))),
                    "children": {},
                    "parent": {},
                    "depth": {i: 0 for i in range(len(groups))},
                }

            # --- Step 7b: Optimize paths ---
            logger.info("  Optimizing paths...")
            for group in groups:
                self.simplifier.simplify_all(group)

            groups = self.path_merger.merge_within_groups(groups)

            # --- Step 8: Generate clean SVG ---
            logger.info("Step 8/8: Generating clean SVG...")
            svg_string = self.svg_generator.generate(
                groups, hierarchy, svg_metadata
            )

            # Final SVG-level optimization
            svg_string = self.svg_optimizer.optimize(svg_string)

            # Save to file if path provided
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(svg_string)
                logger.info(f"Saved to: {output_path}")

            logger.info("Vectorization complete!")
            return svg_string

    @staticmethod
    def _parse_dimensions(metadata: dict) -> tuple:
        """Extract image width and height from SVG metadata."""
        w, h = 1.0, 1.0

        viewbox = metadata.get("viewBox", "")
        if viewbox:
            parts = viewbox.split()
            if len(parts) == 4:
                try:
                    w = float(parts[2])
                    h = float(parts[3])
                except ValueError:
                    pass

        if w <= 1.0:
            width_str = metadata.get("width", "")
            try:
                w = float(width_str.replace("px", "").replace("pt", ""))
            except (ValueError, AttributeError):
                pass

        if h <= 1.0:
            height_str = metadata.get("height", "")
            try:
                h = float(height_str.replace("px", "").replace("pt", ""))
            except (ValueError, AttributeError):
                pass

        return w, h

src/main.py
pythonDownloadCopy code"""CLI entry point."""
import argparse
import sys
from .vectorizer import SVGVectorizer
from .utils.logger import get_logger

logger = get_logger("cli")

def main():
parser = argparse.ArgumentParser(
description="Clean SVG Vectorizer - Convert raster images to organized SVGs"
)

    parser.add_argument("input", help="Input image path (PNG, JPG)")
    parser.add_argument("output", help="Output SVG path")

    parser.add_argument(
        "--preset",
        choices=["logo", "illustration", "photograph"],
        default=None,
        help="Use a preset configuration",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to custom YAML config file",
    )
    parser.add_argument(
        "--method",
        choices=["dbscan", "graph"],
        default=None,
        help="Clustering method",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="DBSCAN epsilon parameter",
    )
    parser.add_argument(
        "--color-tolerance",
        type=float,
        default=None,
        help="Color similarity tolerance (CIELAB Delta-E)",
    )
    parser.add_argument(
        "--no-hierarchy",
        action="store_true",
        help="Disable hierarchy detection",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable group merging",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        default=None,
        help="Quantize colors to N colors before tracing",
    )

    args = parser.parse_args()

    # Build config overrides from CLI args
    overrides = {}

    if args.method:
        overrides.setdefault("clustering", {})["method"] = args.method
    if args.eps:
        overrides.setdefault("clustering", {})["eps"] = args.eps
    if args.color_tolerance:
        overrides.setdefault("grouping", {})["color_tolerance"] = args.color_tolerance
    if args.no_hierarchy:
        overrides.setdefault("hierarchy", {})["enabled"] = False
    if args.no_merge:
        overrides.setdefault("grouping", {})["merge_enabled"] = False
    if args.quantize:
        overrides.setdefault("preprocessing", {})["quantize_colors"] = True
        overrides["preprocessing"]["quantize_n_colors"] = args.quantize

    try:
        vectorizer = SVGVectorizer(
            config=overrides if overrides else None,
            config_path=args.config,
            preset=args.preset,
        )
        svg = vectorizer.vectorize(args.input, args.output)
        logger.info(f"Success! Output: {args.output} ({len(svg)} bytes)")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Vectorization failed: {e}", exc_info=True)
        sys.exit(1)

if **name** == "**main**":
main()
api/server.py
pythonDownloadCopy code"""FastAPI web server for the vectorizer."""
import tempfile
import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from typing import Optional

from src.vectorizer import SVGVectorizer

app = FastAPI(
title="SVG Vectorizer API",
description="Convert raster images to clean, organized SVGs",
version="0.1.0",
)

@app.post("/vectorize")
async def vectorize(
image: UploadFile = File(...),
preset: Optional[str] = Form(None),
method: Optional[str] = Form(None),
eps: Optional[float] = Form(None),
color_tolerance: Optional[float] = Form(None),
quantize_colors: Optional[int] = Form(None),
):
"""
Vectorize an uploaded image and return clean SVG.
""" # Validate file type
allowed = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
if image.content_type not in allowed:
raise HTTPException(400, f"Unsupported image type: {image.content_type}")

    # Build config overrides
    overrides = {}
    if method:
        overrides.setdefault("clustering", {})["method"] = method
    if eps:
        overrides.setdefault("clustering", {})["eps"] = eps
    if color_tolerance:
        overrides.setdefault("grouping", {})["color_tolerance"] = color_tolerance
    if quantize_colors:
        overrides.setdefault("preprocessing", {})["quantize_colors"] = True
        overrides["preprocessing"]["quantize_n_colors"] = quantize_colors

    # Use temp directory for thread safety
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save uploaded file
        file_id = uuid.uuid4().hex[:8]
        ext = os.path.splitext(image.filename)[1] or ".png"
        input_path = os.path.join(tmpdir, f"input_{file_id}{ext}")

        content = await image.read()
        with open(input_path, "wb") as f:
            f.write(content)

        # Vectorize
        try:
            vectorizer = SVGVectorizer(
                config=overrides if overrides else None,
                preset=preset,
            )
            svg_string = vectorizer.vectorize(input_path)
        except Exception as e:
            raise HTTPException(500, f"Vectorization failed: {str(e)}")

    return Response(
        content=svg_string,
        media_type="image/svg+xml",
        headers={
            "Content-Disposition": f'attachment; filename="{file_id}.svg"'
        },
    )

@app.get("/health")
async def health():
return {"status": "ok"}
setup.py
pythonDownloadCopy codefrom setuptools import setup, find_packages

setup(
name="svg-vectorizer",
version="0.1.0",
packages=find_packages(),
install_requires=[
"vtracer",
"opencv-python",
"numpy",
"scikit-learn",
"scipy",
"lxml",
"svgpathtools",
"shapely",
"Pillow",
"scikit-image",
"pyyaml",
"networkx",
],
extras_require={
"api": ["fastapi", "uvicorn", "python-multipart"],
},
entry_points={
"console_scripts": [
"svg-vectorizer=src.main:main",
],
},
)
examples/basic_usage.py
pythonDownloadCopy code"""Basic usage examples."""
from src.vectorizer import SVGVectorizer

# --- Example 1: Simple logo ---

vectorizer = SVGVectorizer(preset="logo")
vectorizer.vectorize("examples/logo.png", "examples/logo_output.svg")

# --- Example 2: Illustration with custom settings ---

vectorizer = SVGVectorizer(config={
"clustering": {
"method": "graph", # Use graph-based grouping
"eps": 0.5,
},
"grouping": {
"color_tolerance": 20.0,
},
})
vectorizer.vectorize("examples/illustration.png", "examples/illustration_output.svg")

# --- Example 3: Photo with color quantization ---

vectorizer = SVGVectorizer(preset="photograph")
vectorizer.vectorize("examples/photo.jpg", "examples/photo_output.svg")

# --- Example 4: Get SVG string without saving to file ---

vectorizer = SVGVectorizer(preset="logo")
svg_string = vectorizer.vectorize("examples/icon.png")
print(f"SVG length: {len(svg_string)} characters")

Development Timeline
Weeks 1–2: Foundation. Set up the project structure, install dependencies, get vtracer working, implement the SVG parser with proper svgpathtools bounding boxes, and validate that you can round-trip an image through trace → parse → regenerate SVG without data loss. This is your "hello world" milestone — raw vtracer output parsed into PathInfo objects and written back out as a valid SVG.
Weeks 3–4: Grouping. Implement the feature extractor, DBSCAN clustering, and graph-based grouping. Test both methods on 5–10 sample images of varying complexity. This is where you'll spend the most time tuning. Start with simple logos (few colors, clear shapes) before moving to complex images.
Weeks 5–6: Hierarchy and Optimization. Implement polygon computation, geometric containment detection, path simplification, coordinate rounding, and same-fill path merging. Test hierarchy detection on images with obvious nesting (icons inside circles, text on backgrounds).
Weeks 7–8: Integration and API. Wire up the full pipeline in SVGVectorizer, implement the CLI and FastAPI server, write tests, tune parameters across your actual use-case images, and handle edge cases (empty images, single-color images, extremely complex images with thousands of paths).

Key Things to Watch For
Performance on large images. The polygon computation step (path_to_polygon with Shapely) and the O(n²) pairwise comparisons in graph grouping and hierarchy detection will be slow if you have thousands of paths. Consider adding early-exit conditions and spatial indexing (Shapely's STRtree) if you hit performance issues.
DBSCAN epsilon sensitivity. The eps parameter is the single most impactful tuning knob in the whole system. Too small and everything becomes its own group. Too large and unrelated paths clump together. You'll likely need different values for different image types — that's what the presets are for.
vtracer's own grouping. vtracer's --hierarchical stacked mode already does some layering. Your post-processing should work with this structure rather than fighting it. If you find vtracer's own grouping is reasonable for your images, you might need less aggressive re-clustering than you expect.
Test with YOUR actual images early. Don't build the whole pipeline before testing on the real images from your visualizer. The specific characteristics of your images (complexity, color count, typical sizes) should drive your parameter choices from day one.
