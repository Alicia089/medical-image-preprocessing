import numpy as np
import pytest
import cv2


@pytest.fixture
def gray_image():
    rng = np.random.default_rng(42)
    return (rng.normal(128, 30, (128, 128)).clip(0, 255)).astype(np.uint8)


@pytest.fixture
def rgb_image():
    rng = np.random.default_rng(42)
    return (rng.normal(128, 30, (128, 128, 3)).clip(0, 255)).astype(np.uint8)


@pytest.fixture
def overexposed_image():
    return np.full((128, 128), 252, dtype=np.uint8)


@pytest.fixture
def underexposed_image():
    return np.full((128, 128), 3, dtype=np.uint8)


@pytest.fixture
def tmp_image_file(tmp_path, gray_image):
    path = tmp_path / "test_scan.png"
    cv2.imwrite(str(path), gray_image)
    return path
