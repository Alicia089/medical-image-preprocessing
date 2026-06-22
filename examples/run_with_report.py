"""
Run the X-ray pipeline and print the metadata sidecar.

Usage:
    python examples/run_with_report.py
"""
import json
from pathlib import Path

from medpreprocess.pipeline import run_pipeline

ROOT = Path(__file__).parent.parent
input_image = ROOT / "tests" / "fixtures" / "synthetic_xray.png"
config = ROOT / "configs" / "xray_standard.yaml"
output_dir = ROOT / "output"

result = run_pipeline(str(config), str(input_image), str(output_dir))
meta = result["metadata"]
print(f"Preprocessing done. Output: {meta['output']}")
print(f"Steps applied: {[s['step'] for s in meta['steps_applied']]}")
print(f"Elapsed: {meta['elapsed_seconds']}s")
