import json
import sys
import types

from kosti.cli import main


def test_audit_cli_writes_json_without_ml(tmp_path, monkeypatch):
    fake = types.ModuleType("kosti.dataset")
    fake.audit_dataset = lambda root, labels_path: {"file_count": 3, "message": "проверено"}
    monkeypatch.setitem(sys.modules, "kosti.dataset", fake)
    output = tmp_path / "nested" / "audit.json"
    assert main(["audit", str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["file_count"] == 3


def test_batch_missing_bundle_fails_without_outputs(tmp_path, capsys):
    output = tmp_path / "results"
    assert main(["batch", str(tmp_path), "--model", str(tmp_path / "absent.json"), "--output", str(output)]) == 2
    assert not output.exists()
    assert "Cannot load quality bundle" in capsys.readouterr().err


def test_installed_package_runs_outside_source_tree(tmp_path):
    import os
    import subprocess
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-m", "kosti.cli", "--help"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "audit" in result.stdout and "batch" in result.stdout
