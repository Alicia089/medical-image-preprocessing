"""
Run the standard X-ray preprocessing pipeline on a sample image.

Usage:
    python examples/run_xray_pipeline.py
"""
from pathlib import Path

from medpreprocess.pipeline import run_pipeline

ROOT = Path(__file__).parent.parent
input_image = ROOT / "tests" / "fixtures" / "synthetic_xray.png"
config = ROOT / "configs" / "xray_standard.yaml"
output_dir = ROOT / "output"

result = run_pipeline(str(config), str(input_image), str(output_dir))
meta = result["metadata"]

print(f"Output image : {meta['output']}")
print(f"Steps applied: {[s['step'] for s in meta['steps_applied']]}")
if meta["artifact_warnings"]:
    print("Artifact warnings:")
    for w in meta["artifact_warnings"]:
        print(f"  - {w}")
else:
    print("No artifact warnings.")
