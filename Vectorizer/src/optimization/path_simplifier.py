"""Simplify and clean individual SVG paths."""
from svgpathtools import parse_path, Path
from typing import List
from ..parsing.svg_parser import PathInfo
from ..utils.logger import get_logger

logger = get_logger(__name__)


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
        if not self.simplify_paths:
            return paths

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
            if precision == 0:
                return str(int(rounded))
            formatted = f"{rounded:.{precision}f}"
            formatted = formatted.rstrip("0").rstrip(".")
            return formatted

        return re.sub(r"-?\d+\.?\d*", round_match, d)
