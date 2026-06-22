import numpy as np
from medpreprocess.steps.base import Step


class Normalize(Step):
    def __init__(self, method: str = "min_max"):
        if method not in ("min_max", "z_score"):
            raise ValueError(f"method must be 'min_max' or 'z_score', got {method!r}")
        self.method = method

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        if self.method == "min_max":
            mn, mx = img.min(), img.max()
            if mx == mn:
                return np.zeros_like(image, dtype=np.uint8)
            return ((img - mn) / (mx - mn) * 255).astype(np.uint8)
        mean, std = img.mean(), img.std()
        if std == 0:
            return np.zeros_like(image, dtype=np.uint8)
        normalized = np.clip((img - mean) / std, -3, 3)
        return ((normalized + 3) / 6 * 255).astype(np.uint8)
