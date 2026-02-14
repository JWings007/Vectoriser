"""vtracer integration for initial image-to-SVG conversion."""
import vtracer
import tempfile
import os
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger(__name__)


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

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
            out_path = tmp.name

        try:
            vtracer.convert_image_to_svg_py(
                image_path,
                out_path,
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

            with open(out_path, "r", encoding="utf-8") as f:
                svg_string = f.read()
        finally:
            try:
                os.remove(out_path)
            except OSError:
                pass

        logger.info(f"Tracing complete. SVG length: {len(svg_string)} chars")
        return svg_string
