"""Batch conversion example."""
from pathlib import Path
from src.vectorizer import SVGVectorizer

INPUT_DIR = Path("examples/batch_inputs")
OUTPUT_DIR = Path("examples/batch_outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

vectorizer = SVGVectorizer(preset="illustration")

for image_path in INPUT_DIR.glob("*.*"):
    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        continue
    output_path = OUTPUT_DIR / f"{image_path.stem}.svg"
    vectorizer.vectorize(str(image_path), str(output_path))
    print(f"Saved: {output_path}")
