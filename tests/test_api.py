import csv
import io
import json
import zipfile
import numpy as np
from fastapi.testclient import TestClient
from kosti.api import create_app
from kosti.contracts import DicomImage, Prediction, HIP, VIOLATIONS
from kosti import pipeline


class FakePredictor:
    def predict(self, image):
        return Prediction(HIP, 0.1, dict.fromkeys(VIOLATIONS[HIP], 0.1))


def test_unconfigured_model():
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/v1/batch", files={"files": ("x.dcm", b"x")}).status_code == 503


def test_full_batch_success_and_error(monkeypatch):
    def reader(path):
        if path.name == "bad.dcm":
            raise ValueError("invalid DICOM")
        return DicomImage(path, np.zeros((2, 2)), "1.2", "1.3", "abc")
    monkeypatch.setattr(pipeline, "read_dicom", reader)
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w") as bundle:
        bundle.writestr("good.dcm", b"good")
        bundle.writestr("bad.dcm", b"bad")
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post("/v1/batch", files={"files": ("batch.zip", source.getvalue())})
        assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as result:
        rows = list(csv.DictReader(io.StringIO(result.read("results.csv").decode("utf-8-sig"))))
        assert [row["processing_status"] for row in rows] == ["Success", "Failure"]
        assert rows[0]["quality_class"] == "0"
        diagnostics = json.loads(result.read("diagnostics.json"))
        errors = [entry for entry in diagnostics if "error_type" in entry]
        assert len(errors) == 1
        assert errors[0]["error_type"] == "ValueError"
        assert diagnostics[0]["projection"] == "unknown"


def test_request_body_limit():
    with TestClient(create_app(FakePredictor(), max_upload_bytes=100)) as client:
        response = client.post("/v1/batch", files={"files": ("x.dcm", b"x" * 101)})
        assert response.status_code == 413


def test_unsafe_zip_returns_400():
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w") as bundle:
        bundle.writestr("../bad", b"x")
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post("/v1/batch", files={"files": ("x.zip", source.getvalue())})
        assert response.status_code == 400


def test_conflicting_zip_returns_400():
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w") as bundle:
        bundle.writestr("a", b"x")
        bundle.writestr("a/image.dcm", b"x")
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post("/v1/batch", files={"files": ("x.zip", source.getvalue())})
    assert response.status_code == 400


def test_oversized_filename_returns_400():
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post("/v1/batch", files={"files": ("x" * 256, b"x")})
    assert response.status_code == 400


def test_review_preview_failure_preserves_prediction(monkeypatch):
    from kosti import api
    def reader(path):
        return DicomImage(path, np.zeros((2, 2)), "1.2", "1.3", "abc")
    def broken_preview(*args, **kwargs):
        raise RuntimeError("Preview encoder failed")
    monkeypatch.setattr(pipeline, "read_dicom", reader)
    monkeypatch.setattr(api, "read_dicom", reader)
    monkeypatch.setattr(api, "thumbnail_png", broken_preview)
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post("/v1/review", files={"files": ("x.dcm", b"x")})
    assert response.status_code == 200
    assert response.json()["rows"][0]["processing_status"] == "Success"
    assert response.json()["previews"] == {}


def test_streamed_body_limit_without_content_length():
    import asyncio
    from kosti.api import _BodyLimitMiddleware
    called, sent = [], []
    async def application(scope, receive, send):
        called.append(True)
    messages = iter([{"type": "http.request", "body": b"123", "more_body": True},
                     {"type": "http.request", "body": b"456", "more_body": False}])
    async def receive():
        return next(messages)
    async def send(message):
        sent.append(message)
    middleware = _BodyLimitMiddleware(application, max_bytes=5, model_ready=True)
    asyncio.run(middleware({"type": "http", "method": "POST", "path": "/v1/batch", "headers": []}, receive, send))
    assert not called
    assert sent[0]["status"] == 413


def test_response_cleanup_when_client_disconnects(tmp_path):
    import asyncio
    import tempfile
    from pathlib import Path
    from kosti.api import _TemporaryFileResponse
    temporary = tempfile.TemporaryDirectory(dir=tmp_path)
    root = Path(temporary.name)
    artifact = root / "results.zip"
    artifact.write_bytes(b"result")
    response = _TemporaryFileResponse(artifact, temporary=temporary)
    async def disconnected_send(message):
        raise asyncio.CancelledError()
    async def receive():
        return {"type": "http.disconnect"}
    try:
        asyncio.run(response({"type": "http", "method": "GET", "headers": [], "extensions": {}}, receive, disconnected_send))
    except asyncio.CancelledError:
        pass
    assert not root.exists()


def test_repeated_unicode_filenames_are_preserved(monkeypatch):
    def reader(path):
        return DicomImage(path, np.zeros((2, 2)), "1.2", "1.3", "abc")
    monkeypatch.setattr(pipeline, "read_dicom", reader)
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post("/v1/batch", files=[("files", ("снимок.dcm", b"1")), ("files", ("снимок.dcm", b"1"))])
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as result:
        rows = list(csv.DictReader(io.StringIO(result.read("results.csv").decode("utf-8-sig"))))
    assert len(rows) == 2
    assert rows[0]["path_to_study"] != rows[1]["path_to_study"]
    assert all(row["path_to_study"].endswith("снимок.dcm") for row in rows)


def test_real_dicom_end_to_end_with_failed_decode_uid_retention(tmp_path):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian
    path = tmp_path / "image.dcm"
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.StudyInstanceUID, ds.SOPInstanceUID = "1.2.3", "1.2.4"
    ds.Rows, ds.Columns = 2, 2
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = np.array([[0, 10], [20, 30]], dtype=np.uint16).tobytes()
    ds.save_as(path)
    good = path.read_bytes()
    del ds.PixelData
    ds.save_as(path)
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post("/v1/batch", files=[("files", ("good.dcm", good)), ("files", ("bad.dcm", path.read_bytes()))])
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as result:
        rows = list(csv.DictReader(io.StringIO(result.read("results.csv").decode("utf-8-sig"))))
    assert [row["processing_status"] for row in rows] == ["Success", "Failure"]
    assert all(row["study_uid"] == "1.2.3" and row["image_uid"] == "1.2.4" for row in rows)
    assert rows[1]["quality_class"] == ""


def test_archive_paths_are_clean_and_macos_junk_is_skipped(monkeypatch):
    def reader(path):
        return DicomImage(path, np.zeros((2, 2)), "1.2", "1.3", "abc")
    monkeypatch.setattr(pipeline, "read_dicom", reader)
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w") as bundle:
        bundle.writestr("study/a.dcm", b"good")
        bundle.writestr("__MACOSX/study/._a.dcm", b"junk")
        bundle.writestr(".DS_Store", b"junk")
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post("/v1/batch", files={"files": ("batch.zip", source.getvalue())})
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as result:
        rows = list(csv.DictReader(io.StringIO(result.read("results.csv").decode("utf-8-sig"))))
    assert [row["path_to_study"] for row in rows] == ["study/a.dcm"]
    assert rows[0]["processing_status"] == "Success"


def test_colliding_archive_paths_are_disambiguated(monkeypatch):
    def reader(path):
        return DicomImage(path, np.zeros((2, 2)), "1.2", "1.3", "abc")
    monkeypatch.setattr(pipeline, "read_dicom", reader)
    def archive():
        source = io.BytesIO()
        with zipfile.ZipFile(source, "w") as bundle:
            bundle.writestr("study/a.dcm", b"good")
        return source.getvalue()
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post("/v1/batch", files=[("files", ("first.zip", archive())),
                                                   ("files", ("second.zip", archive()))])
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as result:
        rows = list(csv.DictReader(io.StringIO(result.read("results.csv").decode("utf-8-sig"))))
    assert [row["path_to_study"] for row in rows] == ["0000/study/a.dcm", "0001/study/a.dcm"]
