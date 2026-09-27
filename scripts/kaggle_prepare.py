"""Build a private Kaggle training input with no original DICOM metadata.

The original DICOM and the local reviewed annotations never leave this machine.
This command deliberately cannot run without a separate image redaction review.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import zipfile
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from kosti.dicom import read_dicom
from kosti.preparation import LABELS, write_json
from kosti.training import load_manifest, partitions, training_support


def checked_pixel_review(path: Path, hashes: set[str]) -> None:
    """Require a separate human check for text burned into every unique image."""
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['pixel_hash', 'reviewed']:
            raise ValueError('Pixel review needs columns pixel_hash,reviewed')
        seen = set()
        for row in reader:
            key = row['pixel_hash']
            if key in seen or key not in hashes or row['reviewed'] != '1':
                raise ValueError('Pixel review must mark every unique image exactly once')
            seen.add(key)
    if seen != hashes:
        raise ValueError('Pixel review does not cover every unique image')


def safe_image(source: Path, destination: Path, study_uid: str, image_uid: str) -> None:
    """Reconstruct a DICOM from an explicit numeric/pixel allowlist."""
    ds = pydicom.dcmread(source)
    if 'ModalityLUTSequence' in ds or 'VOILUTSequence' in ds:
        raise ValueError('LUT-sequence DICOM needs a reviewed conversion path')
    if int(getattr(ds, 'NumberOfFrames', 1)) != 1:
        raise ValueError('Multiframe DICOM is not supported')
    raw = ds.pixel_array
    if raw.ndim != 2 or raw.dtype.kind not in 'ui' or raw.dtype.itemsize not in (1, 2):
        raise ValueError('Only single-frame 8/16-bit integer pixels are supported')
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = image_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    out = FileDataset(str(destination), {}, file_meta=meta, preamble=b'\0' * 128)
    out.SOPClassUID = SecondaryCaptureImageStorage
    out.SOPInstanceUID = image_uid
    out.StudyInstanceUID = study_uid
    out.SeriesInstanceUID = generate_uid()
    out.Modality = 'OT'
    out.Rows, out.Columns = raw.shape
    out.SamplesPerPixel = 1
    out.PhotometricInterpretation = str(ds.PhotometricInterpretation)
    out.BitsAllocated = raw.dtype.itemsize * 8
    out.BitsStored = int(ds.BitsStored)
    out.HighBit = int(ds.HighBit)
    out.PixelRepresentation = int(ds.PixelRepresentation)
    out.PixelData = raw.astype(raw.dtype.newbyteorder('<'), copy=False).tobytes()
    for name in ('PixelPaddingValue', 'PixelPaddingRangeLimit', 'RescaleSlope',
                 'RescaleIntercept', 'WindowCenter', 'WindowWidth', 'PixelSpacing'):
        if name in ds:
            setattr(out, name, ds.get(name).value)
    if 'VOILUTFunction' in ds:
        value = str(ds.VOILUTFunction)
        if value not in ('LINEAR', 'LINEAR_EXACT', 'SIGMOID'):
            raise ValueError('Unsupported VOI LUT function')
        out.VOILUTFunction = value
    out.BurnedInAnnotation = 'NO'  # A separate per-image visual review is required.
    destination.parent.mkdir(parents=True, exist_ok=True)
    out.save_as(destination, enforce_file_format=True)


def prepare(manifest_path: Path, data_root: Path, pixel_review: Path,
            initial_weights: Path, output: Path, dataset_id: str) -> dict:
    manifest = load_manifest(manifest_path)
    for _, training, _ in partitions(manifest, 'cross_validation'):
        training_support(training)
    if not data_root.is_dir() or not initial_weights.is_file():
        raise ValueError('Original DICOM directory and ImageNet weights must exist')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+', dataset_id):
        raise ValueError('Dataset ID must be owner/slug')
    samples = manifest['samples']
    checked_pixel_review(pixel_review, {s['pixel_hash'] for s in samples})
    if output.exists():
        raise FileExistsError('Choose a new package directory')
    staging = output.with_name(output.name + '.partial')
    if staging.exists():
        raise FileExistsError('Remove or inspect an earlier partial package first')
    staging.mkdir(parents=True)
    try:
        root = data_root.resolve()
        study_map, image_map = {}, {}
        def pseudo_study(uid):
            return study_map.setdefault(uid, generate_uid())
        def pseudo_image(uid):
            return image_map.setdefault(uid, generate_uid())
        clean_samples = []
        for index, sample in enumerate(samples):
            source = (root / sample['path']).resolve()
            if not source.is_relative_to(root):
                raise ValueError('Source path escapes DICOM root')
            image = read_dicom(source)
            if any(getattr(image, key) != sample[key]
                   for key in ('study_uid', 'image_uid', 'pixel_hash')):
                raise ValueError('Original DICOM changed since reviewed manifest')
            relpath = f'images/{index:05d}.dcm'
            destination = staging / relpath
            safe_image(source, destination, pseudo_study(sample['study_uid']),
                       pseudo_image(sample['image_uid']))
            checked = read_dicom(destination)
            if checked.pixel_hash != sample['pixel_hash'] or not (checked.pixels == image.pixels).all():
                raise ValueError('De-identification changed training pixels')
            copies = []
            for copy_index, copy in enumerate(sample['copies']):
                copies.append({'path': relpath if copy['path'] == sample['path'] else
                               f'copies/{index:05d}_{copy_index:03d}.dcm',
                               'study_uid': pseudo_study(copy['study_uid']),
                               'image_uid': pseudo_image(copy['image_uid']),
                               'pixel_hash': copy['pixel_hash']})
            clean_samples.append({
                'path': relpath, 'study_uid': checked.study_uid,
                'image_uid': checked.image_uid, 'pixel_hash': checked.pixel_hash,
                'region': sample['region'], 'side': sample['side'],
                'label_source': 'reviewed local annotation', 'labels': sample['labels'],
                'conflict_reviewed': sample.get('conflict_reviewed', False),
                'fold': sample['fold'], 'copies': copies,
            })
        clean = {'schema_version': 1, 'seed': manifest['seed'],
                 'n_splits': manifest['n_splits'], 'label_names': list(LABELS),
                 'samples': clean_samples}
        write_json(staging / 'folds.json', clean)
        load_manifest(staging / 'folds.json')
        with zipfile.ZipFile(staging / 'images.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted((staging / 'images').glob('*.dcm')):
                archive.write(path, path.relative_to(staging).as_posix())
        shutil.rmtree(staging / 'images')
        source = Path(__file__).resolve().parents[1] / 'src' / 'kosti'
        with zipfile.ZipFile(staging / 'project.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(source.rglob('*.py')):
                archive.write(path, (Path('src/kosti') / path.relative_to(source)).as_posix())
        shutil.copyfile(initial_weights, staging / 'imagenet-resnet18.pth')
        write_json(staging / 'dataset-metadata.json', {
            'id': dataset_id, 'title': 'Private DXA training input',
            'licenses': [{'name': 'unknown'}],
        })
        output.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging)
        raise
    return {'samples': len(samples), 'dataset_id': dataset_id, 'package': str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pixel-review', type=Path, required=True)
    parser.add_argument('--initial-weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-id', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.folds, args.data_root, args.pixel_review,
                             args.initial_weights, args.output, args.dataset_id)))


if __name__ == '__main__':
    main()
