from abc import ABC, abstractmethod
import numpy as np


class Step(ABC):
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        pass
