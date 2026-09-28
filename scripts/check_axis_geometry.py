"""Diagnostic only: intensity-based spinal axis, not a validated segmentation.

Uses scanner spacing explicitly supplied in 4.md, never writes it to DICOM.
No outcome-driven threshold fitting: fixed five degrees from the specification.
"""
import json
from pathlib import Path

import numpy as np

from kosti.dicom import read_dicom
from kosti.evaluation import binary_metrics
from kosti.preparation import write_json


def estimate_axis(pixels, spacing, local=False):
    h, w = pixels.shape
    xs, ys = [], []
    # Central column only; omit pelvis and upper peripheral ribs.
    for y in np.linspace(.12*h, .72*h, 20):
        band = pixels[max(0, int(y-.02*h)):min(h, int(y+.02*h)+1)].mean(0)
        smooth = np.convolve(band, np.ones(max(3, int(.12*w))) / max(3, int(.12*w)), mode='same')
        left, right = int(.25*w), int(.75*w)
        peak = left + int(np.argmax(smooth[left:right]))
        lo, hi = max(left, peak-int(.12*w)), min(right, peak+int(.12*w)+1)
        weights = np.maximum(band[lo:hi] - np.quantile(band[lo:hi], .25), 0)
        if weights.sum() > 0:
            xs.append(float(np.average(np.arange(lo, hi), weights=weights)))
            ys.append(float(y))
    if len(xs) < 5:
        raise ValueError('Insufficient central bone signal')
    xs, ys = np.asarray(xs), np.asarray(ys)
    slopes = [(xs[j]-xs[i])/(ys[j]-ys[i]) for i in range(len(xs)) for j in range(i+5,len(xs))]
    slope = float(np.median(slopes))
    if local:
        # Test curved as well as globally tilted axes; each local segment spans
        # half of the sampled column. This remains a diagnostic hypothesis.
        slope = max(abs(float(np.median([
            (xs[j]-xs[i])/(ys[j]-ys[i])
            for i in range(start, start+10) for j in range(i+3, start+10)
        ]))) for start in (0, 5, 10)) if len(xs) == 20 else abs(slope)
    angle = float(np.degrees(np.arctan(abs(slope) * spacing[1]/spacing[0])))
    return angle


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path('artifacts/balanced-baseline-20260927/cv-config.json').read_text())
    samples = json.loads(Path(config['manifest']).read_text())['samples']
    rows = []
    for s in samples:
        if s['region'] != 'spine':
            continue
        image = read_dicom(Path(config['data_root']) / s['path'])
        spacing = image.spacing_mm or (1.05, .6)
        rows.append({'pixel_hash': s['pixel_hash'], 'label': s['labels']['spine_axis'],
                     'angle': estimate_axis(image.pixels, spacing, local=args.local),
                     'spacing_source': 'DICOM' if image.spacing_mm else '4.md scanner profile'})
    y = [r['label'] for r in rows]
    # Monotone bounded score solely for ranking/AUC; NOT a calibrated probability.
    p = [r['angle']/(r['angle']+5) for r in rows]
    report = {'metrics': binary_metrics(y, p), 'samples': rows, 'production_ready': False}
    suffix = '-local' if args.local else ''
    write_json(Path(f'artifacts/axis-geometry-20260928{suffix}.json'), report)
    print(report['metrics'])
