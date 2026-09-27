import json

import numpy as np
import pytest

from kosti.dataset import audit_dataset
from kosti.review import build_review_gallery
from test_dicom import write_dicom


def test_gallery_has_one_preview_per_unique_image_and_keeps_all_rows(tmp_path):
    root = tmp_path / 'dicom'
    root.mkdir()
    write_dicom(root / 'one.dcm')
    write_dicom(root / 'two.dcm')
    write_dicom(root / 'three.dcm', np.array([[5, 15], [25, 35]], dtype=np.uint16))
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps(audit_dataset(root), ensure_ascii=False))
    output = tmp_path / 'review'
    result = build_review_gallery(audit, root, output)
    assert result['unique_images'] == 2 and result['files'] == 3
    assert len(list((output / 'previews').glob('*.png'))) == 2
    page = (output / 'index.html').read_text()
    assert 'one.dcm' in page and 'three.dcm' in page
    assert 'MustNotBeExported' not in page
    assert 'fetch(' not in page and 'http://' not in page and 'https://' not in page
    assert 'annotations.csv' in page and 'pixel-review.csv' in page
    assert 'pixel_reviewed:false' in page
    assert 'conflict_reviewed:false' in page and 'conflict-reviewed' in page
    with pytest.raises(ValueError, match='must be new'):
        build_review_gallery(audit, root, output)


def test_gallery_rejects_stale_audit_and_cleans_partial_output(tmp_path):
    root = tmp_path / 'dicom'
    root.mkdir()
    write_dicom(root / 'one.dcm')
    report = audit_dataset(root)
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps(report, ensure_ascii=False))
    write_dicom(root / 'one.dcm', np.array([[0, 2], [4, 8]], dtype=np.uint16))
    output = tmp_path / 'review'
    with pytest.raises(ValueError, match='identity changed'):
        build_review_gallery(audit, root, output)
    assert not output.exists()


def test_gallery_rejects_path_escape(tmp_path):
    root = tmp_path / 'dicom'
    root.mkdir()
    write_dicom(root / 'one.dcm')
    report = audit_dataset(root)
    report['manifest'][0]['path'] = '../outside.dcm'
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps(report, ensure_ascii=False))
    with pytest.raises(ValueError, match='relative paths'):
        build_review_gallery(audit, root, tmp_path / 'review')
