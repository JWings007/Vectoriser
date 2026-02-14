"""Main orchestrator that ties the full pipeline together."""
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

logger = get_logger(__name__)


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

    def __init__(self, config: dict = None, config_path: str = None, preset: str = None):
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
        default_path = Path(__file__).parent.parent / "config" / "defaults.yaml"
        if default_path.exists():
            with open(default_path) as f:
                base_config = yaml.safe_load(f)
        else:
            base_config = {}

        if preset and "presets" in base_config:
            preset_config = base_config.get("presets", {}).get(preset, {})
            base_config = self._deep_merge(base_config, preset_config)

        if config_path:
            with open(config_path) as f:
                file_config = yaml.safe_load(f)
            base_config = self._deep_merge(base_config, file_config)

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
            svg_string = self.svg_generator.generate(groups, hierarchy, svg_metadata)

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
