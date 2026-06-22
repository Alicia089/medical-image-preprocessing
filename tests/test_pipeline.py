import json
from pathlib import Path

import numpy as np
import pytest

from medpreprocess.pipeline import run_pipeline


def test_pipeline_creates_output_image(tmp_image_file, tmp_path):
    config = Path(__file__).parent.parent / "configs" / "xray_standard.yaml"
    result = run_pipeline(str(config), str(tmp_image_file), str(tmp_path))
    assert Path(result["output_image"]).exists()


def test_pipeline_writes_metadata_sidecar(tmp_image_file, tmp_path):
    config = Path(__file__).parent.parent / "configs" / "xray_standard.yaml"
    run_pipeline(str(config), str(tmp_image_file), str(tmp_path))
    meta_path = tmp_path / f"{tmp_image_file.stem}_meta.json"
    assert meta_path.exists()
    with open(meta_path) as f:
        meta = json.load(f)
    assert "steps_applied" in meta
    assert "artifact_warnings" in meta
    assert "output" in meta


def test_pipeline_metadata_contains_version_and_checksums(tmp_image_file, tmp_path):
    config = Path(__file__).parent.parent / "configs" / "xray_standard.yaml"
    result = run_pipeline(str(config), str(tmp_image_file), str(tmp_path))
    meta = result["metadata"]
    assert "medpreprocess_version" in meta
    assert "input_checksum_sha256" in meta
    assert "config_hash_sha256" in meta
    assert "elapsed_seconds" in meta
    assert "output_shape" in meta
    assert len(meta["input_checksum_sha256"]) == 64  # sha256 hex


def test_pipeline_metadata_contains_expected_steps(tmp_image_file, tmp_path):
    config = Path(__file__).parent.parent / "configs" / "xray_standard.yaml"
    result = run_pipeline(str(config), str(tmp_image_file), str(tmp_path))
    step_names = [s["step"] for s in result["metadata"]["steps_applied"]]
    assert "clahe" in step_names
    assert "normalize" in step_names
    assert "resize" in step_names
    assert "denoise" in step_names


def test_pipeline_output_image_is_correct_size(tmp_image_file, tmp_path):
    config = Path(__file__).parent.parent / "configs" / "xray_standard.yaml"
    result = run_pipeline(str(config), str(tmp_image_file), str(tmp_path))
    assert result["preprocessed_image"].shape[:2] == (512, 512)


def test_pipeline_histology_config(tmp_image_file, tmp_path):
    config = Path(__file__).parent.parent / "configs" / "histology_standard.yaml"
    result = run_pipeline(str(config), str(tmp_image_file), str(tmp_path))
    assert result["preprocessed_image"].shape[:2] == (256, 256)


def test_pipeline_unknown_step_raises_before_processing(tmp_image_file, tmp_path, tmp_path_factory):
    config_dir = tmp_path_factory.mktemp("cfg")
    bad_config = config_dir / "bad.yaml"
    bad_config.write_text("steps:\n  - name: nonexistent_step\n")
    with pytest.raises(ValueError, match="nonexistent_step"):
        run_pipeline(str(bad_config), str(tmp_image_file), str(tmp_path))


def test_pipeline_missing_step_name_raises(tmp_image_file, tmp_path, tmp_path_factory):
    config_dir = tmp_path_factory.mktemp("cfg")
    bad_config = config_dir / "bad.yaml"
    bad_config.write_text("steps:\n  - clip_limit: 2.0\n")
    with pytest.raises(ValueError, match="missing required 'name'"):
        run_pipeline(str(bad_config), str(tmp_image_file), str(tmp_path))


def test_pipeline_unsupported_file_type_raises(tmp_path, tmp_path_factory):
    config = Path(__file__).parent.parent / "configs" / "xray_standard.yaml"
    fake_input = tmp_path / "scan.xyz"
    fake_input.write_bytes(b"fake")
    with pytest.raises(ValueError, match="Unsupported file type"):
        run_pipeline(str(config), str(fake_input), str(tmp_path))


def test_pipeline_missing_input_raises(tmp_path):
    config = Path(__file__).parent.parent / "configs" / "xray_standard.yaml"
    with pytest.raises(FileNotFoundError):
        run_pipeline(str(config), str(tmp_path / "nonexistent.png"), str(tmp_path))
