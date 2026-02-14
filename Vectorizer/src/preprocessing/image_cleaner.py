"""Image preprocessing: denoising, sharpening, contrast."""
import cv2
import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ImageCleaner:
    """Clean up input image before vectorization."""

    def __init__(self, config: dict):
        pre_cfg = config.get("preprocessing", {})
        self.enabled = pre_cfg.get("enabled", True)
        self.denoise = pre_cfg.get("denoise", True)
        self.denoise_strength = pre_cfg.get("denoise_strength", 10)
        self.sharpen = pre_cfg.get("sharpen", False)

    def clean(self, image_path: str, output_path: str) -> str:
        """
        Apply preprocessing to input image.

        Returns:
            Path to cleaned image (or original if preprocessing disabled).
        """
        if not self.enabled:
            return image_path

        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return image_path

        if self.denoise:
            logger.info("Applying denoising...")
            img = cv2.fastNlMeansDenoisingColored(
                img, None, self.denoise_strength, self.denoise_strength, 7, 21
            )

        if self.sharpen:
            logger.info("Applying sharpening...")
            kernel = np.array(
                [
                    [0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0],
                ]
            )
            img = cv2.filter2D(img, -1, kernel)

        cv2.imwrite(output_path, img)
        logger.info(f"Cleaned image saved to: {output_path}")
        return output_path
