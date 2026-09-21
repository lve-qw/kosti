import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from kosti.dicom import read_dicom


def write_dicom(path, pixels=None, **tags):
    if pixels is None:
        pixels = np.array([[0, 10], [20, 30]], dtype=np.uint16)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
    ds.StudyInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.PatientName = 'MustNotBeExported'
    ds.Rows, ds.Columns = pixels.shape[-2:]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = pixels.dtype.itemsize * 8
    ds.BitsStored = ds.BitsAllocated
    ds.HighBit = ds.BitsStored - 1
    ds.PixelRepresentation = int(pixels.dtype.kind == 'i')
    ds.PixelData = pixels.tobytes()
    for key, value in tags.items():
        if value is None:
            if key in ds:
                delattr(ds, key)
        else:
            setattr(ds, key, value)
    ds.save_as(path)
    return ds


def test_monochrome_and_spacing(tmp_path):
    path = tmp_path / 'image.dcm'
    ds = write_dicom(path, PixelSpacing=[0.5, 0.75])
    image = read_dicom(path)
    assert image.pixels.dtype == np.float32
    np.testing.assert_allclose(image.pixels, [[0, 1/3], [2/3, 1]])
    assert image.study_uid == ds.StudyInstanceUID
    assert image.spacing_mm == (0.5, 0.75)


def test_inversion_and_absent_spacing(tmp_path):
    path = tmp_path / 'image.dcm'
    write_dicom(path, PhotometricInterpretation='MONOCHROME1')
    image = read_dicom(path)
    np.testing.assert_allclose(image.pixels, [[1, 2/3], [1/3, 0]])
    assert image.spacing_mm is None


def test_modality_rescale_before_window(tmp_path):
    path = tmp_path / 'image.dcm'
    write_dicom(path, RescaleSlope=2, RescaleIntercept=-10, WindowCenter=20,
                WindowWidth=20, VOILUTFunction='LINEAR_EXACT')
    np.testing.assert_allclose(read_dicom(path).pixels, [[0, 0], [1, 1]])


def test_padding_does_not_determine_range_and_is_black(tmp_path):
    path = tmp_path / 'image.dcm'
    write_dicom(path, np.array([[0, 10], [15, 20]], dtype=np.uint16), PixelPaddingValue=0)
    np.testing.assert_allclose(read_dicom(path).pixels, [[0, 0], [0.5, 1]])


def test_hash_is_raw_and_shape_sensitive(tmp_path):
    a, b, c = [tmp_path / f'{name}.dcm' for name in 'abc']
    write_dicom(a)
    write_dicom(b, PhotometricInterpretation='MONOCHROME1', RescaleSlope=2)
    write_dicom(c, np.array([[0, 10, 20, 30]], dtype=np.uint16))
    assert read_dicom(a).pixel_hash == read_dicom(b).pixel_hash
    assert read_dicom(a).pixel_hash != read_dicom(c).pixel_hash


@pytest.mark.parametrize('tags,reason', [
    ({'StudyInstanceUID': None}, 'Missing required'),
    ({'SOPInstanceUID': None}, 'Missing required'),
    ({'NumberOfFrames': 2}, 'single-frame'),
    ({'PhotometricInterpretation': 'RGB', 'SamplesPerPixel': 3}, 'monochrome'),
    ({'PixelData': None}, 'Pixel Data'),
])
def test_rejected_inputs(tmp_path, tags, reason):
    path = tmp_path / 'image.dcm'
    write_dicom(path, **tags)
    with pytest.raises(ValueError, match=reason):
        read_dicom(path)


def test_constant_and_padding_only(tmp_path):
    path = tmp_path / 'image.dcm'
    write_dicom(path, np.ones((2, 2), dtype=np.uint16))
    with pytest.raises(ValueError, match='Constant'):
        read_dicom(path)
    write_dicom(path, np.ones((2, 2), dtype=np.uint16), PixelPaddingValue=1)
    with pytest.raises(ValueError, match='padding only'):
        read_dicom(path)


def test_damaged_file(tmp_path):
    path = tmp_path / 'bad.dcm'
    path.write_bytes(b'not a dicom')
    with pytest.raises(ValueError, match='Cannot decode DICOM'):
        read_dicom(path)


def test_invalid_uid_retained_with_warning(tmp_path):
    path = tmp_path / 'image.dcm'
    with pytest.warns(UserWarning):
        write_dicom(path, StudyInstanceUID='not-a-valid-uid')
    image = read_dicom(path)
    assert image.study_uid == 'not-a-valid-uid'
    assert any('Invalid StudyInstanceUID' in warning for warning in image.warnings)
