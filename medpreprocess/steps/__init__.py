from medpreprocess.steps.clahe import CLAHE
from medpreprocess.steps.normalize import Normalize
from medpreprocess.steps.resize import Resize
from medpreprocess.steps.denoise import Denoise
from medpreprocess.steps.artifact import ArtifactDetector

STEP_REGISTRY = {
    "clahe": CLAHE,
    "normalize": Normalize,
    "resize": Resize,
    "denoise": Denoise,
    "artifact": ArtifactDetector,
}

__all__ = ["CLAHE", "Normalize", "Resize", "Denoise", "ArtifactDetector", "STEP_REGISTRY"]
