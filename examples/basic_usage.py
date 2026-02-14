"""Basic usage examples."""
from src.vectorizer import SVGVectorizer

# --- Example 1: Simple logo ---
vectorizer = SVGVectorizer(preset="logo")
vectorizer.vectorize("examples/logo.png", "examples/logo_output.svg")

# --- Example 2: Illustration with custom settings ---
vectorizer = SVGVectorizer(
    config={
        "clustering": {
            "method": "graph",
            "eps": 0.5,
        },
        "grouping": {
            "color_tolerance": 20.0,
        },
    }
)
vectorizer.vectorize(
    "examples/illustration.png", "examples/illustration_output.svg"
)

# --- Example 3: Photo with color quantization ---
vectorizer = SVGVectorizer(preset="photograph")
vectorizer.vectorize("examples/photo.jpg", "examples/photo_output.svg")

# --- Example 4: Get SVG string without saving to file ---
vectorizer = SVGVectorizer(preset="logo")
svg_string = vectorizer.vectorize("examples/icon.png")
print(f"SVG length: {len(svg_string)} characters")
