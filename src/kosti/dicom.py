"""Deterministic, metadata-minimal DICOM decoding for single-frame DXA."""
from hashlib import sha256
from pathlib import Path
import warnings

import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from pydicom.uid import UID

from .contracts import DicomImage

# Bound decompression and subsequent float64 normalization allocations. File/ZIP
# byte limits alone do not constrain the decoded size of compressed DICOMs.
MAX_DECODED_PIXELS = 16_000_000


def _projection_from_metadata(ds):
    """Report only an explicit standard view, never infer it from pixels or names.

    ImageOrientationPatient describes the image plane, not the projection beam
    direction, and therefore cannot safely substitute for ViewPosition.
    """
    view = str(getattr(ds, 'ViewPosition', '')).strip().upper()
    if view in {'AP', 'PA', 'LL', 'RL', 'RLD', 'LLD', 'RLO', 'LLO'}:
        return view, 'DICOM.ViewPosition'
    return 'unknown', 'unavailable'


def read_dicom(path: str | Path) -> DicomImage:
    """Decode monochrome pixels to float32 [0, 1]; never manufacture UIDs/scale.

    Padding is excluded from normalization and mapped to black. Missing required
    UIDs, undecodable pixels, multiframe, color, or constant foreground fail.
    Invalid but present UIDs remain usable with an explicit warning.
    """
    path = Path(path)
    notes: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ds = pydicom.dcmread(path)
            study_uid = str(getattr(ds, 'StudyInstanceUID', '')).strip()
            image_uid = str(getattr(ds, 'SOPInstanceUID', '')).strip()
            if not study_uid or not image_uid:
                raise ValueError('Missing required StudyInstanceUID or SOPInstanceUID')
            for name, value in [('StudyInstanceUID', study_uid), ('SOPInstanceUID', image_uid)]:
                if not UID(value).is_valid:
                    notes.append(f'Invalid {name}; original value retained')
            if int(getattr(ds, 'NumberOfFrames', 1)) != 1:
                raise ValueError('Only single-frame DICOM is supported')
            photo = str(getattr(ds, 'PhotometricInterpretation', ''))
            if int(getattr(ds, 'SamplesPerPixel', 1)) != 1 or photo not in ('MONOCHROME1', 'MONOCHROME2'):
                raise ValueError('Only monochrome DICOM is supported')
            rows, columns = int(getattr(ds, 'Rows', 0)), int(getattr(ds, 'Columns', 0))
            if rows <= 0 or columns <= 0 or rows * columns > MAX_DECODED_PIXELS:
                raise ValueError(f'Decoded image exceeds pixel budget or has invalid dimensions (maximum {MAX_DECODED_PIXELS} pixels)')
            if int(getattr(ds, 'BitsAllocated', 0)) not in (1, 8, 16, 32):
                raise ValueError('Unsupported BitsAllocated; expected 1, 8, 16 or 32')
            raw = ds.pixel_array
            if raw.ndim == 3 and raw.shape[0] == 1:
                raw = raw[0]
            if raw.ndim != 2 or raw.shape != (rows, columns) or raw.size > MAX_DECODED_PIXELS:
                raise ValueError('Decoded pixel array does not match bounded image dimensions')
            # Hash decoded stored pixels, not display/windowed intensities.
            canonical = np.ascontiguousarray(raw.astype(raw.dtype.newbyteorder('<')))
            digest = sha256(str(raw.shape).encode() + canonical.dtype.str.encode() + canonical.tobytes()).hexdigest()
            foreground = np.ones(raw.shape, dtype=bool)
            if 'PixelPaddingValue' in ds:
                low = float(ds.PixelPaddingValue)
                high = float(getattr(ds, 'PixelPaddingRangeLimit', low))
                foreground &= ~((raw >= min(low, high)) & (raw <= max(low, high)))
            pixels = np.asarray(apply_voi_lut(apply_modality_lut(raw, ds), ds), dtype=np.float64)
            if not np.isfinite(pixels).all():
                raise ValueError('Non-finite pixel intensities')
            if not foreground.any():
                raise ValueError('Image contains padding only')
            lo, hi = pixels[foreground].min(), pixels[foreground].max()
            if hi <= lo:
                raise ValueError('Constant foreground pixel intensities')
            pixels = np.clip((pixels - lo) / (hi - lo), 0, 1)
            if photo == 'MONOCHROME1':
                pixels = 1 - pixels
            pixels[~foreground] = 0
            spacing = None
            if 'PixelSpacing' in ds:
                try:
                    values = tuple(float(x) for x in ds.PixelSpacing)
                    if len(values) != 2 or not all(np.isfinite(x) and x > 0 for x in values):
                        raise ValueError
                    spacing = values
                except (ValueError, TypeError):
                    notes.append('Invalid PixelSpacing; physical scale unavailable')
            else:
                notes.append('PixelSpacing absent; physical scale unavailable')
            # Do not propagate arbitrary metadata-containing warning messages.
            if caught:
                notes.append('DICOM decoder emitted warnings')
    except Exception as exc:
        raise ValueError(f'Cannot decode DICOM: {exc}') from exc
    projection, projection_source = _projection_from_metadata(ds)
    return DicomImage(path, pixels.astype(np.float32), study_uid, image_uid, digest, spacing, notes,
                      projection, projection_source)
