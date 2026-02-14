"""FastAPI web server for the vectorizer."""
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
    """
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

    with tempfile.TemporaryDirectory() as tmpdir:
        file_id = uuid.uuid4().hex[:8]
        ext = os.path.splitext(image.filename)[1] or ".png"
        input_path = os.path.join(tmpdir, f"input_{file_id}{ext}")

        content = await image.read()
        with open(input_path, "wb") as f:
            f.write(content)

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
        headers={"Content-Disposition": f'attachment; filename="{file_id}.svg"'},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
