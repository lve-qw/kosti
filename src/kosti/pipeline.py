"""Per-image inference and competition output, with isolated failures."""
import csv
import math
import time
import warnings
from dataclasses import asdict
from pathlib import Path

from .contracts import CSV_COLUMNS, VIOLATIONS, Prediction, ResultRow
from .dicom import read_dicom


def _probability(value, label):
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{label} must be a finite probability")
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be a finite probability") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


def validate_prediction(prediction: Prediction):
    region = prediction.anatomical_region
    if region not in VIOLATIONS:
        raise ValueError("Unknown anatomical region")
    expected = set(VIOLATIONS[region])
    if set(prediction.violation_probabilities) != expected:
        raise ValueError("Prediction must contain exactly the region's required violations")
    if not set(prediction.thresholds).issubset(expected):
        raise ValueError("Threshold supplied for unknown violation")
    overall = _probability(prediction.quality_prob, "quality_prob")
    violations = []
    for label in VIOLATIONS[region]:
        probability = _probability(prediction.violation_probabilities[label], label)
        threshold = _probability(prediction.thresholds.get(label, 0.5), "threshold")
        if probability >= threshold:
            violations.append(label)
    return region, overall, violations


def process_files(paths, predictor, display_root=None, display_paths=None):
    """Preserve order and duplicates; failures carry no guessed predictions."""
    items = list(paths)
    if display_paths is not None:
        display_paths = list(display_paths)
        if len(display_paths) != len(items):
            raise ValueError("display_paths must align with paths")
    rows, diagnostics = [], []
    root = Path(display_root).resolve() if display_root is not None else None
    for index, item in enumerate(items):
        path = Path(item)
        if display_paths is not None:
            shown = str(display_paths[index])
        else:
            shown = str(path)
            if root is not None:
                try:
                    shown = str(path.resolve().relative_to(root))
                except ValueError:
                    pass
        row = ResultRow(path_to_study=shown)
        started = time.perf_counter()
        try:
            image = read_dicom(path)
            row.study_uid, row.image_uid = image.study_uid, image.image_uid
            region, overall, violations = validate_prediction(predictor.predict(image))
            row.anatomical_region = region
            row.quality_prob = overall
            row.quality_class = int(bool(violations))
            row.violation_type = "; ".join(violations)
            row.processing_status = "Success"
            diagnostic = {"path_to_study": shown, "projection": image.projection,
                          "projection_source": image.projection_source}
            if image.warnings:
                diagnostic["warnings"] = list(image.warnings)
            diagnostics.append(diagnostic)
        except Exception as exc:
            if not row.study_uid or not row.image_uid:
                # A bad pixel payload need not erase readable DICOM identifiers.
                try:
                    import pydicom
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        metadata = pydicom.dcmread(path, stop_before_pixels=True,
                                                  specific_tags=["StudyInstanceUID", "SOPInstanceUID"])
                    row.study_uid = str(getattr(metadata, "StudyInstanceUID", "")).strip()
                    row.image_uid = str(getattr(metadata, "SOPInstanceUID", "")).strip()
                except Exception:
                    pass
            diagnostics.append({"path_to_study": shown, "error_type": type(exc).__name__, "error": str(exc)})
        finally:
            row.time_of_processing = time.perf_counter() - started
        rows.append(row)
    return rows, diagnostics


def write_csv(rows, path):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            values = asdict(row)
            # CSV quoting does not stop spreadsheet formulas. Preserve original
            # values in API JSON/XLSX; escape dangerous text only in CSV export.
            for key, value in values.items():
                if isinstance(value, str) and value and (
                    value[0] in '\t\r\n' or value.lstrip().startswith(('=', '+', '-', '@'))
                ):
                    values[key] = "'" + value
            writer.writerow(values)


def write_xlsx(rows, path):
    from openpyxl import Workbook
    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet("results")
    sheet.append(list(CSV_COLUMNS))
    # Explicit text cells keep file names beginning '=' from becoming formulas.
    from openpyxl.cell import WriteOnlyCell
    for row in rows:
        values = asdict(row)
        cells = []
        for key in CSV_COLUMNS:
            cell = WriteOnlyCell(sheet, value=values[key])
            if isinstance(values[key], str):
                cell.data_type = "s"
            cells.append(cell)
        sheet.append(cells)
    workbook.save(path)
