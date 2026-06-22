import hashlib
import json
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import yaml

import medpreprocess
from medpreprocess.steps import STEP_REGISTRY
from medpreprocess.steps.artifact import ArtifactWarning

_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


def _validate_config(steps_config: list) -> None:
    """Fail fast: check all step names and required params before touching any image."""
    errors = []
    for i, spec in enumerate(steps_config):
        name = spec.get("name")
        if not name:
            errors.append(f"Step {i}: missing required 'name' field")
            continue
        if name not in STEP_REGISTRY:
            errors.append(
                f"Step {i}: unknown step '{name}'. "
                f"Available steps: {sorted(STEP_REGISTRY.keys())}"
            )
    if errors:
        raise ValueError("Invalid pipeline config:\n" + "\n".join(f"  - {e}" for e in errors))


def _validate_input(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
        )


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f"Could not decode image: {path}")
    return image


def _save_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_hash(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def run_pipeline(config_path: str, input_path: str, output_dir: str) -> dict:
    config_path = Path(config_path)
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    _validate_input(input_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    steps_config = config.get("steps", [])
    _validate_config(steps_config)

    steps = []
    for spec in steps_config:
        name = spec["name"]
        params = {k: v for k, v in spec.items() if k != "name"}
        steps.append((name, params, STEP_REGISTRY[name](**params)))

    input_checksum = _file_checksum(input_path)
    image = _load_image(input_path)
    artifact_warnings = []
    applied = []
    t_start = time.time()

    for step_name, params, step in steps:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ArtifactWarning)
            image = step.apply(image)
        caught = [str(warning.message) for warning in w if issubclass(warning.category, ArtifactWarning)]
        artifact_warnings.extend(caught)
        applied.append({"step": step_name, "params": params})

    elapsed = round(time.time() - t_start, 3)
    stem = input_path.stem
    out_image_path = output_dir / f"{stem}_preprocessed{input_path.suffix}"
    _save_image(image, out_image_path)

    metadata = {
        "medpreprocess_version": medpreprocess.__version__,
        "input": str(input_path),
        "input_checksum_sha256": input_checksum,
        "config": str(config_path),
        "config_hash_sha256": _config_hash(config_path),
        "steps_applied": applied,
        "artifact_warnings": artifact_warnings,
        "output": str(out_image_path),
        "output_shape": list(image.shape),
        "elapsed_seconds": elapsed,
    }
    meta_path = output_dir / f"{stem}_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {"output_image": out_image_path, "metadata": metadata, "preprocessed_image": image}
