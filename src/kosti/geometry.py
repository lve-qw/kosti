"""Deterministic geometry signals grounded in the supplied scanner protocol."""
import numpy as np

# 4.md states that the challenge dataset uses one scanner with this geometry.
# DICOM PixelSpacing, when present and valid, always takes precedence.
DEFAULT_SCANNER_SPACING_MM = (1.05, 0.6)  # row (Y), column (X)


def spine_axis_probability(pixels, spacing_mm=None):
    """Return a bounded axis-risk score or None when geometry is unusable.

    This is a robust centerline measurement, not a learned segmentation.  The
    five-degree clinical boundary maps to 0.5 via angle / (angle + 5).
    """
    image = np.asarray(pixels, dtype=float)
    if image.ndim != 2 or min(image.shape) < 64 or not np.isfinite(image).all():
        return None
    h, w = image.shape
    xs, ys = [], []
    width = max(3, int(.12 * w))
    kernel = np.ones(width) / width
    for y in np.linspace(.12 * h, .72 * h, 20):
        lo_y, hi_y = max(0, int(y-.02*h)), min(h, int(y+.02*h)+1)
        band = image[lo_y:hi_y].mean(0)
        smooth = np.convolve(band, kernel, mode='same')
        left, right = int(.25*w), int(.75*w)
        peak = left + int(np.argmax(smooth[left:right]))
        lo, hi = max(left, peak-int(.12*w)), min(right, peak+int(.12*w)+1)
        weights = np.maximum(band[lo:hi] - np.quantile(band[lo:hi], .25), 0)
        if weights.sum() > 0:
            xs.append(float(np.average(np.arange(lo, hi), weights=weights)))
            ys.append(float(y))
    if len(xs) != 20:
        return None
    xs, ys = np.asarray(xs), np.asarray(ys)
    local = []
    for start in (0, 5, 10):
        slopes = [(xs[j]-xs[i])/(ys[j]-ys[i])
                  for i in range(start, start+10)
                  for j in range(i+3, start+10)]
        local.append(abs(float(np.median(slopes))))
    row_mm, column_mm = spacing_mm or DEFAULT_SCANNER_SPACING_MM
    if not all(np.isfinite(v) and v > 0 for v in (row_mm, column_mm)):
        return None
    angle = float(np.degrees(np.arctan(max(local) * column_mm / row_mm)))
    return angle / (angle + 5.)
