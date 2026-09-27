import hashlib
import json
from types import SimpleNamespace
import zipfile

import pytest

from scripts import package_release


def test_release_contains_model_and_checksums_but_no_private_artifacts(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    for name in ('pyproject.toml', 'README.md', '4.md', 'Dockerfile', 'requirements-core.lock',
                 'requirements-ml.lock', 'requirements-test.lock'):
        (root / name).write_text('test')
    (root / 'src/kosti').mkdir(parents=True)
    (root / 'src/kosti/example.py').write_text('pass')
    (root / 'tests').mkdir()
    (root / 'tests/test_example.py').write_text('def test_example(): assert True')
    (root / 'tests/private.dcm').write_bytes(b'not part of source tests')
    (root / 'artifacts').mkdir()
    (root / 'artifacts/private.dcm').write_bytes(b'private')
    (root / 'kaggle.json').write_text('secret')
    model = root / 'artifacts/manifest.json'
    model.write_text('{}')
    (model.parent / 'weights.pth').write_bytes(b'test-weights')
    monkeypatch.setattr(package_release, 'load_quality_predictor',
                        lambda path: SimpleNamespace(meta={'weights': 'weights.pth'}))
    output = tmp_path / 'release.zip'
    report = package_release.package_release(root, model, output)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert 'models/weights.pth' in names and 'src/kosti/example.py' in names
        assert '4.md' in names and 'tests/test_example.py' in names
        assert not any('private' in name or 'kaggle.json' in name or 'artifacts/' in name for name in names)
        checksums = archive.read('SHA256SUMS').decode().splitlines()
        assert len(checksums) == len(names) - 1
        for entry in checksums:
            digest, name = entry.split('  ', 1)
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        assert json.loads(archive.read('models/manifest.json'))['weights'] == 'weights.pth'
    assert report['sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        package_release.package_release(root, model, output)
