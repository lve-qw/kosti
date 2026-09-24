"""Offline, injectable batch API and local review UI."""
import base64
from dataclasses import asdict
from importlib import resources
import json
from pathlib import Path
import tempfile
import zipfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.responses import FileResponse, JSONResponse, Response

from .archive import ArchiveLimits, UnsafeArchiveError, safe_extract_zip
from .contracts import CSV_COLUMNS
from .dicom import read_dicom
from .pipeline import process_files, write_csv
from .preview import thumbnail_png


class _BodyLimitMiddleware:
    def __init__(self, app, max_bytes, model_ready):
        self.app, self.max_bytes, self.model_ready = app, max_bytes, model_ready

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") not in ("/v1/batch", "/v1/review"):
            return await self.app(scope, receive, send)
        if not self.model_ready:
            return await JSONResponse({"detail": "Quality predictor is not configured"}, status_code=503)(scope, receive, send)
        # Validate actual streamed bytes before the multipart parser allocates files.
        # Bodies larger than 1 MiB spill to disk; absent Content-Length is safe.
        with tempfile.SpooledTemporaryFile(max_size=1024 * 1024) as body:
            count = 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                count += len(chunk)
                if count > self.max_bytes:
                    return await JSONResponse({"detail": "Upload size limit exceeded"}, status_code=413)(scope, receive, send)
                body.write(chunk)
                if not message.get("more_body", False):
                    break
            body.seek(0)
            completed = False
            async def replay_receive():
                nonlocal completed
                if completed:
                    return await receive()
                chunk = body.read(1024 * 1024)
                completed = body.tell() == count
                return {"type": "http.request", "body": chunk, "more_body": not completed}
            await self.app(scope, replay_receive, send)


class _TemporaryFileResponse(FileResponse):
    def __init__(self, *args, temporary, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporary = temporary

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.temporary.cleanup()


def create_app(predictor=None, max_upload_bytes=512 * 1024 * 1024, archive_limits=None):
    if isinstance(max_upload_bytes, bool) or not isinstance(max_upload_bytes, int) or max_upload_bytes <= 0:
        raise ValueError("max_upload_bytes must be a positive integer")
    limits = archive_limits or ArchiveLimits()
    app = FastAPI(title="Kosti offline DXA quality API")
    app.add_middleware(_BodyLimitMiddleware, max_bytes=max_upload_bytes, model_ready=predictor is not None)

    @app.get("/")
    def interface():
        page = resources.files("kosti").joinpath("web", "index.html").read_text(encoding="utf-8")
        return Response(page, media_type="text/html", headers={
            "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self'",
            "Referrer-Policy": "no-referrer", "Cache-Control": "no-store",
        })

    @app.get("/ui/{asset}")
    def interface_asset(asset: str):
        if asset not in ("app.js", "style.css"):
            raise HTTPException(404, "Unknown UI asset")
        body = resources.files("kosti").joinpath("web", asset).read_bytes()
        return Response(body, media_type="text/javascript" if asset.endswith(".js") else "text/css",
                        headers={"Cache-Control": "no-store"})

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if predictor is None:
            raise HTTPException(503, "Quality predictor is not configured")
        return {"status": "ready"}

    def _prepare_batch(files):
        temporary = None
        try:
            if predictor is None:
                raise HTTPException(503, "Quality predictor is not configured")
            if len(files) > limits.max_files:
                raise HTTPException(413, "File count limit exceeded")
            temporary = tempfile.TemporaryDirectory(prefix="kosti-batch-")
            root = Path(temporary.name)
            inputs = root / "inputs"
            inputs.mkdir()
            paths, total_uploaded, expanded_total = [], 0, 0
            for index, upload in enumerate(files):
                # Prefix preserves repeated input filenames without overwrites.
                folder = inputs / f"{index:04d}"
                folder.mkdir()
                filename = Path((upload.filename or "image.dcm").replace("\\", "/")).name
                if filename in ("", ".", ".."):
                    raise HTTPException(400, "Invalid upload filename")
                target = folder / filename
                with target.open("xb") as output:
                    while chunk := upload.file.read(1024 * 1024):
                        total_uploaded += len(chunk)
                        if total_uploaded > max_upload_bytes:
                            raise HTTPException(413, "Upload size limit exceeded")
                        output.write(chunk)
                if zipfile.is_zipfile(target) or filename.lower().endswith(".zip"):
                    extracted = safe_extract_zip(target, folder / "extracted", limits)
                    target.unlink()
                    paths.extend(extracted)
                    expanded_total += sum(path.stat().st_size for path in extracted)
                else:
                    if target.stat().st_size > limits.max_file_bytes:
                        raise HTTPException(413, "File size limit exceeded")
                    paths.append(target)
                    expanded_total += target.stat().st_size
                if len(paths) > limits.max_files or expanded_total > limits.max_total_bytes:
                    raise HTTPException(413, "Expanded batch limit exceeded")
            if not paths:
                raise HTTPException(400, "Batch contains no files")
            rows, diagnostics = process_files(paths, predictor, display_root=inputs)
            csv_path = root / "results.csv"
            write_csv(rows, csv_path)
            diagnostic_path = root / "diagnostics.json"
            diagnostic_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
            result = temporary, root, paths, rows, diagnostics
            temporary = None
            return result
        except (UnsafeArchiveError, zipfile.BadZipFile, NotImplementedError) as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            if temporary is not None:
                temporary.cleanup()

    @app.post("/v1/batch")
    def batch(files: list[UploadFile] = File(...)):
        temporary = None
        try:
            temporary, root, _, _, _ = _prepare_batch(files)
            result = root / "results.zip"
            with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as bundle:
                bundle.write(root / "results.csv", "results.csv")
                bundle.write(root / "diagnostics.json", "diagnostics.json")
            response = _TemporaryFileResponse(result, media_type="application/zip", filename="results.zip",
                                              temporary=temporary, headers={"Cache-Control": "no-store"})
            temporary = None  # Response owns cleanup, including after streaming.
            return response
        finally:
            if temporary is not None:
                temporary.cleanup()
            for upload in files:
                upload.file.close()

    @app.post("/v1/review")
    def review(files: list[UploadFile] = File(...)):
        temporary = None
        try:
            temporary, root, paths, rows, diagnostics = _prepare_batch(files)
            previews = {}
            for index, (path, row) in enumerate(zip(paths, rows)):
                try:
                    image = read_dicom(path)
                    previews[str(index)] = "data:image/png;base64," + base64.b64encode(thumbnail_png(image, max_side=360)).decode("ascii")
                except ValueError:
                    # A damaged DICOM still has its own result row and diagnostic.
                    pass
            return JSONResponse({
                "columns": list(CSV_COLUMNS),
                "rows": [asdict(row) for row in rows],
                "diagnostics": diagnostics,
                "previews": previews,
                "csv": (root / "results.csv").read_text(encoding="utf-8"),
            }, headers={"Cache-Control": "no-store"})
        finally:
            if temporary is not None:
                temporary.cleanup()
            for upload in files:
                upload.file.close()

    return app


app = create_app()
