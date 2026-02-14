"""Color conversion and comparison utilities using CIELAB color space."""
import numpy as np
from typing import Tuple, Optional


def hex_to_rgb(hex_color: str) -> Optional[Tuple[int, int, int]]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.strip()

    if not hex_color.startswith("#"):
        return None

    hex_color = hex_color[1:]

    # Handle shorthand like #FFF
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)

    if len(hex_color) != 6:
        return None

    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    except ValueError:
        return None


def rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """
    Convert RGB to CIELAB color space.
    Uses D65 illuminant reference white.
    """
    # Normalize to 0-1
    r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0

    # Linearize (inverse sRGB companding)
    def linearize(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r_lin = linearize(r_n)
    g_lin = linearize(g_n)
    b_lin = linearize(b_n)

    # RGB to XYZ (sRGB D65)
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    # D65 reference white
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    # XYZ to Lab
    def f(t):
        delta = 6.0 / 29.0
        if t > delta**3:
            return t ** (1.0 / 3.0)
        else:
            return t / (3.0 * delta**2) + 4.0 / 29.0

    l = 116.0 * f(y) - 16.0
    a = 500.0 * (f(x) - f(y))
    b_val = 200.0 * (f(y) - f(z))

    return (l, a, b_val)


def hex_to_lab(hex_color: str) -> Optional[Tuple[float, float, float]]:
    """Convert hex color directly to CIELAB."""
    rgb = hex_to_rgb(hex_color)
    if rgb is None:
        return None
    return rgb_to_lab(*rgb)


def delta_e(
    lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]
) -> float:
    """
    Calculate CIE76 Delta-E color difference.
    Values roughly mean:
    < 1.0 : imperceptible
    1-2 : barely perceptible
    2-10 : perceptible at close look
    11-49 : colors are more similar than different
    100 : exact opposite colors
    """
    return np.sqrt(
        (lab2[0] - lab1[0]) ** 2
        + (lab2[1] - lab1[1]) ** 2
        + (lab2[2] - lab1[2]) ** 2
    )


def colors_similar(color1: str, color2: str, tolerance: float = 25.0) -> bool:
    """Check if two hex colors are perceptually similar using CIELAB Delta-E."""
    if color1 == color2:
        return True

    lab1 = hex_to_lab(color1)
    lab2 = hex_to_lab(color2)

    if lab1 is None or lab2 is None:
        return color1 == color2

    return delta_e(lab1, lab2) < tolerance
