import warnings

import numpy as np
import pytest
from medpreprocess.steps.base import Step


def test_step_is_abstract():
    with pytest.raises(TypeError):
        Step()


def test_step_subclass_must_implement_apply():
    class BadStep(Step):
        pass

    with pytest.raises(TypeError):
        BadStep()


def test_step_subclass_apply_called(gray_image):
    class DummyStep(Step):
        def apply(self, image: np.ndarray) -> np.ndarray:
            return image

    step = DummyStep()
    result = step.apply(gray_image)
    assert result is gray_image


# ---------------------------------------------------------------------------
# Task 3: CLAHE
# ---------------------------------------------------------------------------
from medpreprocess.steps.clahe import CLAHE


def test_clahe_output_shape_matches_input(gray_image):
    step = CLAHE(clip_limit=2.0, tile_grid_size=[8, 8])
    result = step.apply(gray_image)
    assert result.shape == gray_image.shape


def test_clahe_output_dtype_is_uint8(gray_image):
    step = CLAHE(clip_limit=2.0)
    result = step.apply(gray_image)
    assert result.dtype == np.uint8


def test_clahe_works_on_rgb_image(rgb_image):
    step = CLAHE(clip_limit=2.0)
    result = step.apply(rgb_image)
    assert result.shape == rgb_image.shape
    assert result.dtype == np.uint8


def test_clahe_default_params():
    step = CLAHE()
    assert step.clip_limit == 2.0
    assert step.tile_grid_size == (8, 8)


# ---------------------------------------------------------------------------
# Task 4: Normalize
# ---------------------------------------------------------------------------
from medpreprocess.steps.normalize import Normalize


def test_normalize_min_max_output_range(gray_image):
    step = Normalize(method="min_max")
    result = step.apply(gray_image)
    assert result.dtype == np.uint8
    assert int(result.min()) >= 0
    assert int(result.max()) <= 255


def test_normalize_z_score_output_range(gray_image):
    step = Normalize(method="z_score")
    result = step.apply(gray_image)
    assert result.dtype == np.uint8


def test_normalize_output_shape_unchanged(gray_image):
    step = Normalize(method="min_max")
    result = step.apply(gray_image)
    assert result.shape == gray_image.shape


def test_normalize_invalid_method_raises():
    with pytest.raises(ValueError, match="method must be"):
        Normalize(method="unknown")


# ---------------------------------------------------------------------------
# Task 5: Resize
# ---------------------------------------------------------------------------
from medpreprocess.steps.resize import Resize


def test_resize_output_dimensions(gray_image):
    step = Resize(width=64, height=64)
    result = step.apply(gray_image)
    assert result.shape == (64, 64)


def test_resize_rgb_output_dimensions(rgb_image):
    step = Resize(width=64, height=32)
    result = step.apply(rgb_image)
    assert result.shape == (32, 64, 3)


def test_resize_dtype_preserved(gray_image):
    step = Resize(width=256, height=256)
    result = step.apply(gray_image)
    assert result.dtype == np.uint8


def test_resize_default_interpolation():
    step = Resize(width=64, height=64)
    assert step.interpolation is not None


# ---------------------------------------------------------------------------
# Task 6: Denoise
# ---------------------------------------------------------------------------
from medpreprocess.steps.denoise import Denoise


def test_denoise_gaussian_shape_preserved(gray_image):
    step = Denoise(method="gaussian", kernel_size=3)
    result = step.apply(gray_image)
    assert result.shape == gray_image.shape
    assert result.dtype == np.uint8


def test_denoise_median_shape_preserved(gray_image):
    step = Denoise(method="median", kernel_size=3)
    result = step.apply(gray_image)
    assert result.shape == gray_image.shape
    assert result.dtype == np.uint8


def test_denoise_even_kernel_rounded_up():
    step = Denoise(kernel_size=4)
    assert step.kernel_size % 2 == 1


def test_denoise_invalid_method_raises():
    with pytest.raises(ValueError, match="method must be"):
        Denoise(method="unknown")


# ---------------------------------------------------------------------------
# Task 7: ArtifactDetector
# ---------------------------------------------------------------------------
from medpreprocess.steps.artifact import ArtifactDetector, ArtifactWarning


def test_artifact_detector_returns_image_unchanged(gray_image):
    step = ArtifactDetector()
    result = step.apply(gray_image)
    np.testing.assert_array_equal(result, gray_image)


def test_artifact_detector_warns_on_overexposed(overexposed_image):
    step = ArtifactDetector(overexposure_threshold=0.9)
    with pytest.warns(ArtifactWarning, match="Overexposure"):
        step.apply(overexposed_image)


def test_artifact_detector_warns_on_underexposed(underexposed_image):
    step = ArtifactDetector(underexposure_threshold=0.1)
    with pytest.warns(ArtifactWarning, match="Underexposure"):
        step.apply(underexposed_image)


def test_artifact_detector_no_warning_on_normal_image(gray_image):
    step = ArtifactDetector()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ArtifactWarning)
        step.apply(gray_image)
    artifact_warnings = [w for w in caught if issubclass(w.category, ArtifactWarning)]
    assert len(artifact_warnings) == 0
