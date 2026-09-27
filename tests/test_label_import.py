import csv
import json

import pytest

from kosti.dicom import read_dicom
from kosti.label_import import import_expert_labels
from test_dicom import write_dicom


def fixture(tmp_path, region='hip', orientation=None):
    path = tmp_path / 'image.dcm'
    write_dicom(path, **({'PatientOrientation': orientation} if orientation else {}))
    image = read_dicom(path)
    record = dict(path=path.name, study_uid=image.study_uid, image_uid=image.image_uid,
                  pixel_hash=image.pixel_hash, label_study_folder='study')
    labels = dict(spine_position=0, spine_axis=0, spine_artifact=0, spine_quality=1,
                  left_hip_position=1, left_hip_roi=0, left_hip_quality=1,
                  right_hip_position=0, right_hip_roi=0, right_hip_quality=0)
    audit = {'manifest': [record], 'labels': {'records': [
        {'study_folder': 'study', 'excel_row': 3, 'labels': labels}]}}
    review = {'schema_version': 1, 'reviewer': 'test visual indexing', 'images': [
        {'pixel_hash': image.pixel_hash, 'region': region,
         'shaft_image_side': 'right' if region == 'hip' else '',
         'reviewed': True, 'pixel_reviewed': True}]}
    a, r = tmp_path / 'audit.json', tmp_path / 'review.json'
    a.write_text(json.dumps(audit)); r.write_text(json.dumps(review))
    return a, r


def result_row(tmp_path):
    with (tmp_path / 'output/annotations.csv').open() as stream:
        return next(csv.DictReader(stream))


def test_orientation_links_correct_excel_side(tmp_path):
    a, r = fixture(tmp_path, orientation=['L', 'F'])
    summary = import_expert_labels(a, r, tmp_path, tmp_path / 'output')
    row = result_row(tmp_path)
    assert row['side'] == 'left' and row['hip_position'] == '1'
    assert row['spine_quality'] == '' and summary['unknown_hip_side'] == 0


def test_unknown_side_keeps_only_matching_labels(tmp_path):
    a, r = fixture(tmp_path)
    summary = import_expert_labels(a, r, tmp_path, tmp_path / 'output')
    row = result_row(tmp_path)
    assert row['side'] == '' and row['hip_roi'] == '0'
    assert row['hip_position'] == '' and row['hip_quality'] == ''
    assert summary['unknown_hip_side'] == 1
    assert {d['reason'] for d in summary['decisions']} == {'unknown_laterality'}


def test_source_contradiction_is_masked_and_recorded(tmp_path):
    a, r = fixture(tmp_path, region='spine')
    summary = import_expert_labels(a, r, tmp_path, tmp_path / 'output')
    row = result_row(tmp_path)
    assert row['spine_quality'] == '' and row['spine_axis'] == '0'
    assert summary['decisions'][0]['reason'] == 'quality_conflicts_with_types'
    assert summary['decisions'][0]['original'] == 1


def test_unreviewed_pixels_rejected(tmp_path):
    a, r = fixture(tmp_path)
    review = json.loads(r.read_text()); review['images'][0]['pixel_reviewed'] = False
    r.write_text(json.dumps(review))
    with pytest.raises(ValueError, match='explicit region and pixel review'):
        import_expert_labels(a, r, tmp_path, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_changed_dicom_rejected(tmp_path):
    a, r = fixture(tmp_path)
    write_dicom(tmp_path / 'image.dcm')  # New UIDs; same pixels must not suffice.
    with pytest.raises(ValueError, match='changed since audit'):
        import_expert_labels(a, r, tmp_path, tmp_path / 'output')
