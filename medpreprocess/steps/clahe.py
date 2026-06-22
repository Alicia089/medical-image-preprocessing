import cv2
import numpy as np
from medpreprocess.steps.base import Step


class CLAHE(Step):
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: list = None):
        self.clip_limit = clip_limit
        self.tile_grid_size = tuple(tile_grid_size or [8, 8])

    def apply(self, image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        if image.ndim == 3:
            channels = [clahe.apply(image[:, :, i]) for i in range(image.shape[2])]
            return np.stack(channels, axis=2)
        return clahe.apply(image)
