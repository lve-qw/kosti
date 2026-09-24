"""Small local previews from already decoded DICOM pixels."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def thumbnail_png(image, max_side=420):
    pixels = np.clip(image.pixels * 255, 0, 255).astype(np.uint8)
    rendered = Image.fromarray(pixels, mode='L')
    rendered.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    result = io.BytesIO()
    rendered.save(result, format='PNG')
    return result.getvalue()
