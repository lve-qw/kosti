"""Privacy and provenance gates for Kaggle packaging; no network calls."""
import csv
import json
import zipfile

import numpy as np
import pydicom
import pytest
from pydicom.tag import Tag

from kosti.dicom import read_dicom
from kosti.preparation import IDENTITY, LABELS
from scripts.kaggle_prepare import prepare
from scripts.kaggle_run import stage_kernel, validate_package
from test_dicom import write_dicom


def inputs(tmp_path):
    samples = []
    for fold in range(3):
        for region in ('spine', 'hip'):
            for positive in (0, 1):
                path = tmp_path / f'image-{len(samples)}.dcm'
                ds = write_dicom(path, np.arange(64, dtype=np.uint16).reshape(8, 8) + len(samples),
                                 PatientID='SECRET', InstitutionName='SECRET')
                ds.add_new(Tag(0x00111010), 'LO', 'SECRET')
                ds.save_as(path)
                image = read_dicom(path)
                sample = {k: getattr(image, k) for k in IDENTITY if k != 'path'}
                sample.update(path=path.name, region=region, side='' if region == 'spine' else 'left',
                              label_source='local-source-secret', fold=fold,
                              labels={k: positive if k.startswith(region + '_') else None for k in LABELS})
                sample['copies'] = [{k: sample[k] for k in IDENTITY}]
                samples.append(sample)
    manifest = {'schema_version': 1, 'seed': 42, 'label_names': list(LABELS),
                'samples': samples, 'n_splits': 3}
    folds = tmp_path / 'folds.json'
    folds.write_text(json.dumps(manifest), encoding='utf-8')
    review = tmp_path / 'pixel-review.csv'
    with review.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pixel_hash', 'reviewed'])
        writer.writerows((s['pixel_hash'], '1') for s in samples)
    weights = tmp_path / 'imagenet.pth'
    weights.write_bytes(b'synthetic-test-weights')
    return folds, review, weights, samples


def test_requires_separate_pixel_review_and_all_training_labels(tmp_path):
    folds, review, weights, samples = inputs(tmp_path)
    review.unlink()
    with pytest.raises(FileNotFoundError):
        prepare(folds, tmp_path, review, weights, tmp_path / 'package', 'tester/private-dxa')
    assert not (tmp_path / 'package').exists()
    with review.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pixel_hash', 'reviewed'])
        writer.writerows((s['pixel_hash'], '1') for s in samples)
    for sample in samples:
        if sample['region'] == 'spine':
            sample['labels']['spine_axis'] = None
    folds.write_text(json.dumps({'schema_version': 1, 'seed': 42, 'label_names': list(LABELS),
                                  'samples': samples, 'n_splits': 3}), encoding='utf-8')
    with pytest.raises(ValueError, match='both observed classes for spine_axis'):
        prepare(folds, tmp_path, review, weights, tmp_path / 'package', 'tester/private-dxa')


def test_package_removes_source_metadata_and_stages_private_kernel(tmp_path):
    folds, review, weights, samples = inputs(tmp_path)
    package = tmp_path / 'package'
    result = prepare(folds, tmp_path, review, weights, package, 'tester/private-dxa')
    assert result['samples'] == len(samples)
    assert validate_package(package) == ('tester/private-dxa', 'private-dxa')
    clean = json.loads((package / 'folds.json').read_text())
    assert 'local-source-secret' not in (package / 'folds.json').read_text()
    assert clean['samples'][0]['study_uid'] != samples[0]['study_uid']
    with zipfile.ZipFile(package / 'images.zip') as archive:
        raw = archive.read(clean['samples'][0]['path'])
    output = tmp_path / 'clean.dcm'
    output.write_bytes(raw)
    ds = pydicom.dcmread(output)
    assert 'PatientName' not in ds
    assert 'PatientID' not in ds and 'InstitutionName' not in ds
    assert all(not element.tag.is_private for element in ds.iterall())
    assert read_dicom(output).pixel_hash == samples[0]['pixel_hash']
    kernel = tmp_path / 'kernel'
    staged = stage_kernel(package, kernel, 'tester/dxa-baseline')
    assert staged['kernel_id'] == 'tester/dxa-baseline'
    metadata = json.loads((kernel / 'kernel-metadata.json').read_text())
    assert metadata['is_private'] is True and metadata['enable_gpu'] is True
    compile((kernel / 'runner.py').read_text(), 'runner.py', 'exec')
    assert not (kernel / 'kaggle.json').exists()


def test_package_rejects_unreviewed_labels(tmp_path):
    folds, review, weights, samples = inputs(tmp_path)
    samples[0]['label_source'] = ''
    folds.write_text(json.dumps({'schema_version': 1, 'seed': 42, 'label_names': list(LABELS),
                                  'samples': samples, 'n_splits': 3}), encoding='utf-8')
    with pytest.raises(ValueError, match='reviewed label source'):
        prepare(folds, tmp_path, review, weights, tmp_path / 'package', 'tester/private-dxa')
    assert not (tmp_path / 'package').exists()
