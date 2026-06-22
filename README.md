# medpreprocess

A reusable, config-driven medical image preprocessing package with standardized YAML pipelines and metadata sidecars.

---

## Install

```bash
git clone https://github.com/Alicia089/medical-image-preprocessing.git
cd medical-image-preprocessing
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

```bash
# Run the standard X-ray pipeline
medpreprocess run \
  --config configs/xray_standard.yaml \
  --input tests/fixtures/synthetic_xray.png \
  --output output/
```

Each run writes two files to the output directory:
- `<stem>_preprocessed.png` — processed image
- `<stem>_meta.json` — full provenance sidecar (version, config hash, input checksum, steps, timing)

---

## Pipeline Config

Pipelines are YAML files — an ordered list of named steps with parameters. The config is validated before any image processing begins; invalid step names or missing fields raise a clear error immediately.

```yaml
# configs/xray_standard.yaml
steps:
  - name: clahe
    clip_limit: 2.5
    tile_grid_size: [8, 8]
  - name: normalize
    method: min_max
  - name: resize
    width: 512
    height: 512
  - name: denoise
    method: gaussian
    kernel_size: 3
  - name: artifact
```

### Available Steps

| Step | Key Parameters | Description |
|------|---------------|-------------|
| `clahe` | `clip_limit`, `tile_grid_size` | CLAHE contrast enhancement |
| `normalize` | `method` (`min_max`, `z_score`) | Pixel value normalization |
| `resize` | `width`, `height`, `interpolation` | Resize to fixed dimensions |
| `denoise` | `method` (`gaussian`, `median`), `kernel_size` | Noise reduction |
| `artifact` | `clip_threshold`, `overexposure_threshold`, `underexposure_threshold` | Artifact detection (warning only — does not alter image) |

Two reference configs ship with the repo:
- `configs/xray_standard.yaml` — chest X-ray pipeline (512×512, Gaussian denoise)
- `configs/histology_standard.yaml` — histology slide pipeline (256×256, median denoise, tighter artifact thresholds)

> **Modality note:** Thresholds and step order matter per modality. The reference configs are starting points — validate outputs against your specific imaging context before use in a downstream pipeline.

---

## Metadata Sidecar

Every run produces a JSON sidecar for full reproducibility:

```json
{
  "medpreprocess_version": "0.1.0",
  "input": "tests/fixtures/synthetic_xray.png",
  "input_checksum_sha256": "a3f...",
  "config": "configs/xray_standard.yaml",
  "config_hash_sha256": "b7c...",
  "steps_applied": [
    {"step": "clahe", "params": {"clip_limit": 2.5, "tile_grid_size": [8, 8]}},
    ...
  ],
  "artifact_warnings": [],
  "output": "output/synthetic_xray_preprocessed.png",
  "output_shape": [512, 512],
  "elapsed_seconds": 0.041
}
```

---

## Safety

This repo contains no patient data. All test fixtures are synthetically generated (`tests/fixtures/generate_fixture.py`). Do not commit real medical images. If adapting this toolkit for real imaging workflows, ensure compliance with applicable data governance and de-identification requirements for your jurisdiction.

---

## Development

```bash
pip install -r requirements-dev.txt
pip install -e .
pytest tests/
flake8 medpreprocess tests
```

---

## Skills

![Python](https://img.shields.io/badge/Python-3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green)
![Docker](https://img.shields.io/badge/Docker-multi--stage-blue)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI-orange)
