"""Color quantization for images with gradients or many colors."""
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ColorQuantizer:
    """Reduce number of colors in an image before vectorization."""

    def __init__(self, config: dict):
        pre_cfg = config.get("preprocessing", {})
        self.enabled = pre_cfg.get("quantize_colors", False)
        self.n_colors = pre_cfg.get("quantize_n_colors", 16)

    def quantize(self, image_path: str, output_path: str) -> str:
        """
        Quantize image colors using K-means clustering.

        Args:
            image_path: Path to input image.
            output_path: Path to save quantized image.

        Returns:
            Path to the quantized image (or original if quantization disabled).
        """
        if not self.enabled:
            return image_path

        logger.info(f"Quantizing colors to {self.n_colors} clusters...")

        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return image_path

        h, w, c = img.shape
        pixels = img.reshape(-1, 3).astype(np.float32)

        kmeans = MiniBatchKMeans(
            n_clusters=self.n_colors,
            random_state=42,
            batch_size=1024,
            n_init=3,
        )
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_.astype(np.uint8)

        quantized = centers[labels].reshape(h, w, c)
        cv2.imwrite(output_path, quantized)

        logger.info(f"Quantized image saved to: {output_path}")
        return output_path
