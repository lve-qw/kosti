"""Projection metadata must remain explicit and must not leak arbitrary text."""
from types import SimpleNamespace

import pytest

from kosti.dicom import _projection_from_metadata, read_dicom
from kosti.pipeline import process_files
from kosti.contracts import Prediction, SPINE, VIOLATIONS
from test_dicom import write_dicom


@pytest.mark.parametrize('view', ['AP', 'PA', 'LL', 'RL', 'RLD', 'LLD', 'RLO', 'LLO'])
def test_known_projection(view):
    assert _projection_from_metadata(SimpleNamespace(ViewPosition=view)) == (view, 'DICOM.ViewPosition')


@pytest.mark.parametrize('view', ['', 'PRIVATE PATIENT TEXT', 'LATERAL', None])
def test_missing_or_unrecognized_projection_is_unknown(view):
    assert _projection_from_metadata(SimpleNamespace(ViewPosition=view)) == ('unknown', 'unavailable')


def test_projection_uses_explicit_view_not_orientation_or_filename(tmp_path):
    path = tmp_path / 'AP.dcm'
    write_dicom(path, ImageOrientationPatient=[1, 0, 0, 0, 1, 0])
    assert read_dicom(path).projection == 'unknown'
    write_dicom(path, ViewPosition='PA')
    image = read_dicom(path)
    assert (image.projection, image.projection_source) == ('PA', 'DICOM.ViewPosition')
    predictor = SimpleNamespace(predict=lambda image: Prediction(
        SPINE, 0.1, {label: 0.1 for label in VIOLATIONS[SPINE]}))
    rows, diagnostics = process_files([path], predictor)
    assert rows[0].processing_status == 'Success'
    assert diagnostics[0]['projection'] == 'PA'
    assert diagnostics[0]['projection_source'] == 'DICOM.ViewPosition'
    assert 'MustNotBeExported' not in str(diagnostics)
