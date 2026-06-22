import cv2
import numpy as np
from medpreprocess.steps.base import Step


class Resize(Step):
    def __init__(self, width: int, height: int, interpolation=cv2.INTER_LINEAR):
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.width, self.height), interpolation=self.interpolation)
