"""Build a source-and-model release without source DICOM, credentials or run manifests."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from kosti.ml import load_quality_predictor


def package_release(root, model, output):
    root, model, output = Path(root).resolve(), Path(model).resolve(), Path(output)
    if output.exists():
        raise FileExistsError('Release already exists')
    predictor = load_quality_predictor(model)
    metadata = dict(predictor.meta)
    weights = (model.parent / metadata['weights']).resolve()
    metadata['weights'] = 'weights.pth'
    files = {}
    for name in ('pyproject.toml', 'README.md', '4.md', 'Dockerfile',
                 'requirements-core.lock', 'requirements-ml.lock', 'requirements-test.lock'):
        files[name] = (root / name).read_bytes()
    for folder, suffixes in (('src/kosti', {'.py', '.html', '.css', '.js'}),
                             ('scripts', {'.py', '.sh'}), ('docs', {'.md', '.pptx'}),
                             ('tests', {'.py'})):
        for path in sorted((root / folder).rglob('*')):
            if path.is_file() and path.suffix in suffixes and '__pycache__' not in path.parts:
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    raise ValueError('Release sources must stay inside the repository')
                files[path.relative_to(root).as_posix()] = path.read_bytes()
    files['models/manifest.json'] = json.dumps(metadata, ensure_ascii=False, indent=2).encode()
    files['models/weights.pth'] = weights.read_bytes()
    files['START_HERE.md'] = '''# Запуск Kosti

Это исследовательская модель контроля качества DXA. Прочитайте docs/RELEASE.md.
Архив содержит код и обученные веса; исходные исследования и ключи исключены.

## Установка Python 3.11–3.13

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-core.lock -r requirements-ml.lock
.venv/bin/python -m pip install --no-deps .
.venv/bin/kosti serve --model models/manifest.json
```

Открыть http://127.0.0.1:8000/. Установка требует доступа к пакетам;
после установки инференс использует только локальные файлы.
Пакетная обработка: `.venv/bin/kosti batch input.zip --model models/manifest.json --output outputs/run1`.

## Docker

```sh
sh scripts/build.sh
sh scripts/run.sh
```

Linux/Docker необходимо отдельно проверить на целевой машине.
SHA256SUMS содержит контрольные суммы всех остальных файлов архива.
'''.encode()
    checksums = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
                        for name, data in sorted(files.items()))
    files['SHA256SUMS'] = checksums.encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    return {'path': str(output.resolve()), 'files': len(files),
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package_release(Path(__file__).resolve().parents[1], args.model, args.output)))
