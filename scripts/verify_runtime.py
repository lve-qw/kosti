"""Exercise a trained bundle with real local DICOM through the API, without outbound sockets."""
import argparse
import csv
import io
import json
from pathlib import Path
import socket
import zipfile


def verify_runtime(model, inputs, output):
    import torch
    from fastapi.testclient import TestClient
    from kosti.api import create_app
    from kosti.contracts import CSV_COLUMNS
    from kosti.ml import load_quality_predictor
    torch.set_num_threads(1)
    paths = [Path(p) for p in inputs]
    if not paths:
        raise ValueError('At least one real DICOM input is required')
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w') as archive:
        for index, path in enumerate(paths):
            archive.writestr(f'renamed-{index}.dcm', path.read_bytes())
        archive.writestr('duplicate.dcm', paths[0].read_bytes())
        archive.writestr('broken.dcm', b'not a DICOM file')
    original_connect = socket.socket.connect
    def no_network(*args, **kwargs):
        raise AssertionError('Inference attempted a socket connection')
    socket.socket.connect = no_network
    try:
        predictor = load_quality_predictor(model)
        with TestClient(create_app(predictor)) as client:
            assert client.get('/ready').status_code == 200
            response = client.post('/v1/batch', files={'files': ('review.zip', payload.getvalue(), 'application/zip')})
            assert response.status_code == 200
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                reader = csv.DictReader(io.StringIO(archive.read('results.csv').decode('utf-8-sig')))
                assert reader.fieldnames == list(CSV_COLUMNS)
                rows = list(reader)
            assert len(rows) == len(paths) + 2
            assert all(r['processing_status'] == 'Success' for r in rows[:-1])
            assert rows[-1]['processing_status'] == 'Failure'
            assert rows[-1]['quality_class'] == rows[-1]['quality_prob'] == ''
            assert all(rows[0][k] == rows[-2][k] for k in CSV_COLUMNS
                       if k not in ('path_to_study', 'time_of_processing'))
            review = client.post('/v1/review', files={'files': ('review.zip', payload.getvalue(), 'application/zip')})
            assert review.status_code == 200
            body = review.json()
            assert len(body['rows']) == len(rows)
            assert len(body['previews']) == len(paths) + 1
            for row, previous in zip(body['rows'], rows):
                assert row['processing_status'] == previous['processing_status']
            report = {'passed': True, 'trained_model': str(Path(model).resolve()),
                      'files': len(rows), 'success': len(rows) - 1, 'expected_failure': 1,
                      'csv_columns': list(CSV_COLUMNS), 'preview_count': len(body['previews']),
                      'duplicate_preserved': True, 'outbound_socket_connections_blocked': True,
                      'scope': 'Local in-process HTTP integration; not a Linux/container validation.'}
    finally:
        socket.socket.connect = original_connect
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path, nargs='+')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_runtime(args.model, args.input, args.output), ensure_ascii=False))
