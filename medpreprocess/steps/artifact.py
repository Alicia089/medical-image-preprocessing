import warnings
import numpy as np
from medpreprocess.steps.base import Step


class ArtifactWarning(UserWarning):
    pass


class ArtifactDetector(Step):
    def __init__(
        self,
        clip_threshold: float = 0.01,
        overexposure_threshold: float = 0.95,
        underexposure_threshold: float = 0.05,
    ):
        self.clip_threshold = clip_threshold
        self.overexposure_threshold = overexposure_threshold
        self.underexposure_threshold = underexposure_threshold

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        total = img.size

        overexposed_ratio = np.sum(img >= 255 * self.overexposure_threshold) / total
        if overexposed_ratio > self.clip_threshold:
            warnings.warn(
                f"Overexposure detected: {overexposed_ratio:.1%} of pixels near max intensity",
                ArtifactWarning,
                stacklevel=2,
            )

        underexposed_ratio = np.sum(img <= 255 * self.underexposure_threshold) / total
        if underexposed_ratio > self.clip_threshold:
            warnings.warn(
                f"Underexposure detected: {underexposed_ratio:.1%} of pixels near min intensity",
                ArtifactWarning,
                stacklevel=2,
            )

        return image
