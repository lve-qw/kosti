"""Stage and optionally publish a private Kaggle GPU baseline run."""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pydicom

from kosti.training import load_manifest

SAFE_DICOM_TAGS = {
    'SOPClassUID', 'SOPInstanceUID', 'StudyInstanceUID', 'SeriesInstanceUID',
    'Modality', 'Rows', 'Columns', 'SamplesPerPixel', 'PhotometricInterpretation',
    'BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation', 'PixelData',
    'PixelPaddingValue', 'PixelPaddingRangeLimit', 'RescaleSlope',
    'RescaleIntercept', 'WindowCenter', 'WindowWidth', 'PixelSpacing',
    'VOILUTFunction', 'BurnedInAnnotation',
}


def validate_package(package: Path) -> tuple[str, str]:
    required = {'folds.json', 'images.zip', 'project.zip',
                'imagenet-resnet18.pth', 'dataset-metadata.json'}
    if not package.is_dir() or {p.name for p in package.iterdir()} != required:
        raise ValueError('Package must contain exactly the four generated input files')
    metadata = json.loads((package / 'dataset-metadata.json').read_text(encoding='utf-8'))
    if set(metadata) != {'id', 'title', 'licenses'} or metadata['licenses'] != [{'name': 'unknown'}]:
        raise ValueError('Unexpected Kaggle dataset metadata')
    dataset_id = metadata['id']
    if not isinstance(dataset_id, str) or not re.fullmatch(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+', dataset_id):
        raise ValueError('Invalid dataset ID')
    manifest = load_manifest(package / 'folds.json')
    expected = {s['path'] for s in manifest['samples']}
    if any(not p.startswith('images/') or not p.endswith('.dcm') for p in expected):
        raise ValueError('Unexpected package image paths')
    with zipfile.ZipFile(package / 'images.zip') as archive:
        if set(archive.namelist()) != expected:
            raise ValueError('Image archive does not match reviewed manifest')
        by_path = {s['path']: s for s in manifest['samples']}
        for name in archive.namelist():
            ds = pydicom.dcmread(io.BytesIO(archive.read(name)))
            if any(element.tag.is_private or element.keyword not in SAFE_DICOM_TAGS
                   for element in ds.iterall()):
                raise ValueError('Image archive contains unexpected DICOM metadata')
            sample = by_path[name]
            if (str(ds.StudyInstanceUID) != sample['study_uid'] or
                str(ds.SOPInstanceUID) != sample['image_uid'] or
                ds.BurnedInAnnotation != 'NO'):
                raise ValueError('Image archive does not match clean manifest')
            if sample['label_source'] != 'reviewed local annotation':
                raise ValueError('Package contains original annotation provenance')
    with zipfile.ZipFile(package / 'project.zip') as archive:
        if 'src/kosti/training.py' not in archive.namelist():
            raise ValueError('Current training source is missing from project archive')
    return dataset_id, dataset_id.split('/')[1]


def stage_kernel(package: Path, kernel_dir: Path, kernel_id: str) -> dict:
    dataset_id, dataset_slug = validate_package(package)
    if not isinstance(kernel_id, str) or not re.fullmatch(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+', kernel_id):
        raise ValueError('Kernel ID must be owner/slug')
    if kernel_id.split('/')[0] != dataset_id.split('/')[0]:
        raise ValueError('Private dataset and kernel must have the same owner')
    if kernel_dir.exists():
        raise FileExistsError('Choose a new kernel staging directory')
    kernel_dir.mkdir(parents=True)
    try:
        runner = '''import json, os, subprocess, sys, zipfile
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pydicom==3.0.1'])
# The dataset mount layout differs between Kaggle images: locate the package by
# content and support both stored zips and automatically extracted directories.
package = next((p.parent for p in Path('/kaggle/input').rglob('folds.json')), None)
if package is None:
    raise FileNotFoundError('Training package not found under /kaggle/input')
print('package dir:', package)
if (package / 'project.zip').is_file() and (package / 'images.zip').is_file():
    stage = Path('/tmp/kosti-input')
    stage.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package / 'project.zip') as archive:
        archive.extractall(stage / 'project')
    with zipfile.ZipFile(package / 'images.zip') as archive:
        archive.extractall(stage / 'images')
    project_dir, data_root = stage / 'project', stage / 'images'
else:
    project_dir, data_root = package / 'project', package / 'images'
if not (project_dir / 'src').is_dir() or not (data_root / 'images').is_dir():
    raise FileNotFoundError('Unexpected dataset layout: ' + str(package))
sys.path.insert(0, str(project_dir / 'src'))
from kosti.training import create_config, run_training
work = Path('/kaggle/working')
config_path = work / 'train.json'
create_config(package / 'folds.json', data_root,
              package / 'imagenet-resnet18.pth', work / 'results', config_path)
config = json.loads(config_path.read_text(encoding='utf-8'))
config['device'] = 'cuda'
config_path.write_text(json.dumps(config), encoding='utf-8')
summary = run_training(config_path, execute=True)
print(summary)
print((work / 'results' / 'metrics.json').read_text(encoding='utf-8'))
'''.replace('DATASET_SLUG', dataset_slug)
        (kernel_dir / 'runner.py').write_text(runner, encoding='utf-8')
        metadata = {
            'id': kernel_id, 'title': 'DXA quality ResNet18 baseline',
            'code_file': 'runner.py', 'language': 'python', 'kernel_type': 'script',
            'is_private': True, 'enable_gpu': True, 'enable_internet': True,
            'dataset_sources': [dataset_id], 'competition_sources': [],
            'kernel_sources': [], 'model_sources': [],
        }
        (kernel_dir / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    except Exception:
        shutil.rmtree(kernel_dir)
        raise
    return {'dataset_id': dataset_id, 'kernel_id': kernel_id, 'kernel_dir': str(kernel_dir)}


def kaggle_command(credential: Path, *args: str) -> None:
    """Use the legacy credential only from a temporary mode-0700 directory."""
    if not credential.is_file() or credential.stat().st_mode & 0o077:
        raise ValueError('Credential must exist and be readable only by its owner')
    with tempfile.TemporaryDirectory(prefix='kaggle-config-') as folder:
        config_dir = Path(folder)
        config_dir.chmod(0o700)
        target = config_dir / 'kaggle.json'
        shutil.copyfile(credential, target)
        target.chmod(0o600)
        env = {**os.environ, 'KAGGLE_CONFIG_DIR': str(config_dir)}
        command = [sys.executable, '-m', 'kaggle', *args]
        result = subprocess.run(command, env=env, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, check=False)
        if result.returncode:
            raise RuntimeError(f'Kaggle command failed (exit {result.returncode}); credential and server output suppressed')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--kernel-dir', type=Path, required=True)
    parser.add_argument('--kernel-id', required=True)
    parser.add_argument('--credential', type=Path)
    parser.add_argument('--action', choices=('stage', 'upload-dataset', 'push-kernel'), default='stage')
    args = parser.parse_args()
    if args.action == 'stage':
        print(json.dumps(stage_kernel(args.package, args.kernel_dir, args.kernel_id)))
    else:
        if args.credential is None:
            parser.error('--credential is required for upload and push')
        dataset_id, _ = validate_package(args.package)
        if args.action == 'upload-dataset':
            kaggle_command(args.credential, 'datasets', 'create', '-t', '-p', str(args.package.resolve()))
            print(json.dumps({'uploaded_private_dataset': dataset_id}))
        else:
            metadata = json.loads((args.kernel_dir / 'kernel-metadata.json').read_text(encoding='utf-8'))
            if metadata.get('dataset_sources') != [dataset_id] or metadata.get('is_private') is not True:
                raise ValueError('Kernel metadata does not refer to the private training dataset')
            kaggle_command(args.credential, 'kernels', 'push', '-p', str(args.kernel_dir.resolve()))
            print(json.dumps({'launched_private_kernel': metadata['id'],
                              'note': 'A first push may register a title-derived slug; verify with '
                                      'kaggle kernels list --mine -s <slug>'}))


if __name__ == '__main__':
    main()
