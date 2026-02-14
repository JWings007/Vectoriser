"""CLI entry point."""
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
        choices=["logo", "illustration", "photograph", "lineart"],
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
        "--no-path-merge",
        action="store_true",
        help="Disable merging adjacent same-fill paths",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable path simplification/rounding",
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
    if args.no_path_merge:
        overrides.setdefault("optimization", {})["merge_adjacent"] = False
    if args.no_simplify:
        overrides.setdefault("optimization", {})["simplify_paths"] = False
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


if __name__ == "__main__":
    main()
