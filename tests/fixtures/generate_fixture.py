"""Run once to regenerate the synthetic test fixture: python tests/fixtures/generate_fixture.py"""
from pathlib import Path

import cv2
import numpy as np

rng = np.random.default_rng(42)
image = (rng.normal(128, 40, (256, 256)).clip(0, 255)).astype(np.uint8)
out = Path(__file__).parent / "synthetic_xray.png"
cv2.imwrite(str(out), image)
print(f"Wrote {out}")
