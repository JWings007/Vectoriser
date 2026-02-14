from setuptools import setup, find_packages

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
