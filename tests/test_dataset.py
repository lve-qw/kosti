import json
import pytest
from openpyxl import Workbook

from kosti.dataset import audit_dataset, audit_labels
from test_dicom import write_dicom


def write_labels(path):
    book = Workbook()
    sheet = book.active
    sheet.append(['№', 'study', 'Позвоночник', None, None, 'Правое бедро', None,
                  'Левое бедро', None, 'Итог', None, None])
    sheet.append([None, None, 'корректная укладка', 'ось позвоночника', 'предметы',
                  'позиционирование', 'области', 'позиционирование', 'области',
                  'Позвоночник', 'правого', 'левого'])
    sheet.append([1, 'folder-one', 0, 0, 0, None, None, 0, 1, 1, None, 1])
    sheet.append([2, 'folder-two', 0, 1, 0, 0, 0, None, None, 0, 0, None])
    book.save(path)


def test_missing_masks_and_conflicts(tmp_path):
    path = tmp_path / 'labels.xlsx'
    write_labels(path)
    report = audit_labels(path)
    assert report['row_count'] == 2
    assert report['records'][0]['labels']['right_hip_roi'] is None
    assert report['records'][0]['label_mask']['right_hip_roi'] is False
    assert report['counts']['right_hip_roi'] == {'observed': 1, 'positive': 0, 'missing': 1}
    assert [r['excel_row'] for r in report['contradictions']] == [3, 4]
    assert report['records'][0]['labels']['spine_quality'] == 1


def test_duplicates_and_folder_uid_mapping(tmp_path):
    folder = tmp_path / 'folder-one'
    folder.mkdir()
    first = write_dicom(folder / 'one.dcm')
    write_dicom(folder / 'two.dcm', StudyInstanceUID=first.StudyInstanceUID)
    (tmp_path / 'bad.dcm').write_bytes(b'broken')
    labels = tmp_path / 'labels.xlsx'
    write_labels(labels)
    report = audit_dataset(tmp_path, labels)
    assert report['file_count'] == 3
    assert report['decoded_count'] == 2
    assert report['failed_count'] == 1
    assert report['study_count'] == 1
    assert report['unique_pixel_count'] == 1
    assert report['duplicate_file_count'] == 1
    assert report['folder_to_study_uid'] == {'folder-one': [str(first.StudyInstanceUID)]}
    assert report['unmatched_label_folders'] == ['folder-two']
    assert report['image_labels_assigned'] is False
    assert 'MustNotBeExported' not in json.dumps(report)
    assert 'anatomical_region' not in report['manifest'][0]


def test_cross_study_duplicates_flagged(tmp_path):
    write_dicom(tmp_path / 'one.dcm')
    write_dicom(tmp_path / 'two.dcm')
    report = audit_dataset(tmp_path)
    assert len(report['cross_study_duplicate_groups']) == 1
    assert report['study_count'] == 2


def test_unknown_excel_schema_fails(tmp_path):
    path = tmp_path / 'unknown.xlsx'
    book = Workbook()
    book.active.append(['not', 'our', 'schema'])
    book.save(path)
    with pytest.raises(ValueError, match='schema'):
        audit_labels(path)
