import cv2
import numpy as np
from medpreprocess.steps.base import Step


class Denoise(Step):
    def __init__(self, method: str = "gaussian", kernel_size: int = 3):
        if method not in ("gaussian", "median"):
            raise ValueError(f"method must be 'gaussian' or 'median', got {method!r}")
        self.method = method
        # kernel_size must be odd and positive
        k = int(kernel_size)
        self.kernel_size = k if k % 2 == 1 else k + 1

    def apply(self, image: np.ndarray) -> np.ndarray:
        if self.method == "gaussian":
            return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)
        return cv2.medianBlur(image, self.kernel_size)
