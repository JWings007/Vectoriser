"""Final SVG-level optimizations."""
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SVGOptimizer:
    """Apply final optimizations to the SVG output string."""

    def __init__(self, config: dict):
        self.config = config

    def optimize(self, svg_string: str) -> str:
        """Apply text-level SVG optimizations."""
        svg_string = self._compact_whitespace(svg_string)
        svg_string = self._remove_empty_groups(svg_string)
        return svg_string

    def _compact_whitespace(self, svg: str) -> str:
        """Reduce excessive whitespace in path d attributes."""
        import re

        svg = re.sub(
            r'd="([^"]*)"', lambda m: f'd="{" ".join(m.group(1).split())}"', svg
        )
        return svg

    def _remove_empty_groups(self, svg: str) -> str:
        """Remove <g> elements that contain no children."""
        import re

        svg = re.sub(r"<g[^>]*/>\s*", "", svg)
        svg = re.sub(r"<g[^>]*>\s*</g>\s*", "", svg)
        return svg
