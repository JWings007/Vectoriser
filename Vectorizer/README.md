SVG Vectorizer
==============

Clean SVG generation from raster images with grouping, hierarchy detection,
and path optimization. Includes a CLI and a FastAPI server.

Quick Start
-----------
1) Install dependencies:
   `pip install -r requirements.txt`

2) Run the CLI:
   `python -m src.main input.png output.svg`

3) Run the API:
   `uvicorn api.server:app --reload`

Configuration
-------------
Default settings live in `config/defaults.yaml`. You can override with a
custom YAML file or CLI flags.
