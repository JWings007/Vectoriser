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
        self.mode = pre_cfg.get("quantize_mode", "fixed")
        self.min_colors = pre_cfg.get("quantize_n_colors_min", 8)
        self.max_colors = pre_cfg.get("quantize_n_colors_max", 64)
        self.color_space = pre_cfg.get("quantize_color_space", "lab")
        self.sample_size = pre_cfg.get("quantize_sample_size", 50000)

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

        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return image_path

        h, w, c = img.shape

        if self.color_space.lower() == "lab":
            work = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        else:
            work = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pixels = work.reshape(-1, 3).astype(np.float32)

        if self.mode == "adaptive":
            n_colors = self._choose_k(pixels)
            if n_colors <= 0:
                logger.info("Adaptive quantization skipped (already low color count).")
                return image_path
        else:
            n_colors = self.n_colors

        logger.info(f"Quantizing colors to {n_colors} clusters ({self.mode})...")

        kmeans = MiniBatchKMeans(
            n_clusters=n_colors,
            random_state=42,
            batch_size=1024,
            n_init=3,
        )
        labels = kmeans.fit_predict(self._sample_pixels(pixels))
        centers = kmeans.cluster_centers_

        # Re-assign for all pixels
        labels_full = kmeans.predict(pixels)
        centers = np.clip(centers, 0, 255).astype(np.uint8)

        quantized = centers[labels_full].reshape(h, w, c)

        if self.color_space.lower() == "lab":
            quantized = cv2.cvtColor(quantized, cv2.COLOR_LAB2BGR)
        else:
            quantized = cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_path, quantized)

        logger.info(f"Quantized image saved to: {output_path}")
        return output_path

    def _sample_pixels(self, pixels: np.ndarray) -> np.ndarray:
        if len(pixels) <= self.sample_size:
            return pixels
        idx = np.random.choice(len(pixels), self.sample_size, replace=False)
        return pixels[idx]

    def _choose_k(self, pixels: np.ndarray) -> int:
        sample = self._sample_pixels(pixels)
        unique = len(np.unique(sample.astype(np.uint8), axis=0))
        if unique <= self.n_colors:
            return 0

        std = float(np.std(sample, axis=0).mean())
        n = self.n_colors
        if std > 30 or unique > 2048:
            n = max(n, self.max_colors)
        elif std > 20 or unique > 1024:
            n = max(n, min(self.max_colors, int(n * 2)))
        elif std > 12 or unique > 512:
            n = max(n, min(self.max_colors, int(n * 1.5)))

        n = max(self.min_colors, min(self.max_colors, int(n)))
        return n
