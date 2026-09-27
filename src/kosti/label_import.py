"""Join supplied expert scores to visually indexed images without inventing labels."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import pydicom

from .dicom import read_dicom
from .preparation import FIELDS, IDENTITY, LABELS, audit_records, read_json


def import_expert_labels(audit_path, review_path, data_root, output):
    audit, review = read_json(audit_path), read_json(review_path)
    records = audit_records(audit)
    if audit.get('failed_count') or audit.get('failures'):
        raise ValueError('Resolve audit failures first')
    if review.get('schema_version') != 1 or not str(review.get('reviewer', '')).strip():
        raise ValueError('Review needs schema_version=1 and named reviewer/method')
    indexed = {}
    for item in review.get('images', []):
        key = item.get('pixel_hash')
        if key in indexed or item.get('reviewed') is not True or item.get('pixel_reviewed') is not True:
            raise ValueError('Every unique image needs explicit region and pixel review')
        if item.get('region') not in ('spine', 'hip'):
            raise ValueError('Invalid reviewed region')
        shaft = item.get('shaft_image_side')
        if shaft not in ('left', 'right', '') or (item['region'] == 'hip' and not shaft):
            raise ValueError('Hip review needs visible shaft side in image coordinates')
        indexed[key] = item
    groups = defaultdict(list)
    for row in records:
        groups[row['pixel_hash']].append(row)
    if set(groups) != set(indexed):
        raise ValueError('Review must cover exactly the audited unique images')
    expert = audit.get('labels') or {}
    if expert.get('invalid_labels') or expert.get('duplicate_study_rows'):
        raise ValueError('Resolve ambiguous Excel entries first')
    sources = {r['study_folder']: r for r in expert.get('records', [])}
    root, output = Path(data_root).resolve(), Path(output)
    if not root.is_dir() or output.exists():
        raise ValueError('Input root must exist and output directory must be new')
    annotations, decisions, unique = [], [], []
    for key, copies in sorted(groups.items()):
        item = indexed[key]
        orientations = set()
        for copy in copies:
            path = (root / copy['path']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Audit path escapes input root')
            image = read_dicom(path)
            if any(getattr(image, name) != copy[name] for name in IDENTITY[1:]):
                raise ValueError('DICOM changed since audit')
            ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=['PatientOrientation'])
            if ds.get('PatientOrientation'):
                orientations.add(tuple(ds.PatientOrientation))
        region, side = item['region'], ''
        if region == 'hip' and len(orientations) == 1:
            orientation = next(iter(orientations))
            if orientation in (('L', 'F'), ('R', 'F')):
                side = item['shaft_image_side']
                # L\\F means the patient's left is toward image-right.
                if orientation[0] == 'L':
                    side = 'left' if side == 'right' else 'right'
        values = {name: [] for name in LABELS}
        source_rows = set()
        for copy in copies:
            source = sources.get(copy.get('label_study_folder'))
            if source is None:
                raise ValueError('Audit image has no unambiguous Excel study row')
            source_rows.add(source['excel_row'])
            labels = source['labels']
            for name in LABELS:
                if not name.startswith(region + '_'):
                    values[name].append(None)
                    continue
                if region == 'spine':
                    value = labels[name]
                elif side:
                    value = labels[side + '_' + name]
                else:
                    left, right = labels['left_' + name], labels['right_' + name]
                    value = left if left == right else None
                    if left != right:
                        decisions.append({'pixel_hash': key, 'excel_row': source['excel_row'],
                                          'label': name, 'reason': 'unknown_laterality',
                                          'original': [left, right]})
                values[name].append(value)
        merged = {}
        for name, observed in values.items():
            known = set(v for v in observed if v is not None)
            merged[name] = next(iter(known)) if len(known) == 1 else None
            if len(known) > 1:
                decisions.append({'pixel_hash': key, 'label': name,
                                  'reason': 'conflicting_duplicate_labels', 'original': sorted(known)})
        parts = [merged[k] for k in LABELS if k.startswith(region + '_') and not k.endswith('_quality')]
        quality_key = region + '_quality'
        quality = merged[quality_key]
        if quality is not None and ((quality == 0 and 1 in parts) or
                                   (all(v is not None for v in parts) and quality != int(any(parts)))):
            decisions.append({'pixel_hash': key, 'label': quality_key,
                              'reason': 'quality_conflicts_with_types', 'original': quality})
            merged[quality_key] = None
        source_text = (f"supplied expert workbook rows {','.join(map(str, sorted(source_rows)))}; "
                       f"image indexing: {review['reviewer']}; "
                       f"laterality: {'DICOM PatientOrientation' if side else 'unknown; per-label L/R consensus'}")
        unique.append({'pixel_hash': key, 'region': region, 'side': side, 'labels': merged})
        for copy in copies:
            annotations.append({**{k: copy[k] for k in IDENTITY}, 'region': region, 'side': side,
                                'reviewed': '1', 'label_source': source_text,
                                **{k: '' if v is None else v for k, v in merged.items()}})
    summary = {'schema_version': 1, 'reviewer': review['reviewer'], 'files': len(annotations),
               'unique_images': len(unique), 'unknown_hip_side': sum(r['region'] == 'hip' and not r['side'] for r in unique),
               'region_counts': {r: sum(s['region'] == r for s in unique) for r in ('spine', 'hip')},
               'label_support': {k: {'observed': sum(s['labels'][k] is not None for s in unique),
                                     'positive': sum(s['labels'][k] == 1 for s in unique)} for k in LABELS},
               'decisions': decisions,
               'audit_sha256': hashlib.sha256(Path(audit_path).read_bytes()).hexdigest(),
               'review_sha256': hashlib.sha256(Path(review_path).read_bytes()).hexdigest()}
    output.mkdir(parents=True)
    with (output / 'annotations.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(annotations)
    with (output / 'pixel-review.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pixel_hash', 'reviewed'])
        writer.writerows((key, '1') for key in sorted(groups))
    (output / 'provenance.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary
