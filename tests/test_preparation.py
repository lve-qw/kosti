import csv
import json

import pytest

from kosti.cli import main
from kosti.preparation import FIELDS, LABELS, annotation_template, prepare_folds, evaluate_predictions


def inputs(tmp_path):
    # Shared pixels bridge two studies; a second image in the latter must stay in the same fold.
    records = [dict(path=f'{i}.dcm', study_uid=s, image_uid=str(i), pixel_hash=h)
               for i, (s, h) in enumerate([('a', 'x'), ('b', 'x'), ('b', 'y'), ('c', 'z'), ('d', 'w')])]
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps({'manifest': records, 'failed_count': 0}))
    annotations = tmp_path / 'review.csv'
    annotation_template(audit, annotations)
    with annotations.open(newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row.update(region='spine', side='', reviewed='1', label_source='expert-review:row-3',
                   spine_position='1', spine_axis='', spine_artifact='0', spine_quality='1')
    save(annotations, rows)
    return audit, annotations, rows


def save(path, rows, fields=FIELDS):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_template_is_blank_and_never_overwrites(tmp_path):
    audit, annotations, _ = inputs(tmp_path)
    blank = tmp_path / 'blank.csv'
    assert annotation_template(audit, blank) == 5
    with blank.open() as f:
        rows = list(csv.DictReader(f))
    assert all(r['reviewed'] == '0' and r['region'] == '' and all(r[k] == '' for k in LABELS) for r in rows)
    before = annotations.read_bytes()
    with pytest.raises(FileExistsError):
        annotation_template(audit, annotations)
    assert annotations.read_bytes() == before


def test_group_before_dedup_and_preserve_unknowns(tmp_path):
    audit, annotations, _ = inputs(tmp_path)
    result = prepare_folds(audit, annotations, tmp_path / 'folds.json')
    by_hash = {s['pixel_hash']: s for s in result['samples']}
    assert result['source_images'] == 5 and result['unique_images'] == 4
    assert len(by_hash['x']['copies']) == 2
    assert by_hash['x']['fold'] == by_hash['y']['fold']
    assert by_hash['x']['labels']['spine_axis'] is None
    assert by_hash['x']['labels']['hip_quality'] is None
    assert {s['fold'] for s in result['samples']} == {0, 1, 2}
    second = prepare_folds(audit, annotations, tmp_path / 'again.json')
    assert result == second


@pytest.mark.parametrize('changes,match', [
    ({'reviewed': '0'}, 'reviewed'),
    ({'label_source': ''}, 'label_source'),
    ({'study_uid': 'other'}, 'identity'),
    ({'hip_roi': '0'}, 'Inapplicable'),
    ({'spine_axis': 'maybe'}, 'Labels'),
    ({'spine_quality': '0'}, 'Conflicting'),
    ({'region': 'hip', 'side': ''}, 'left/right'),
    ({'spine_position': '0'}, 'inconsistent'),
])
def test_invalid_review_blocks_output(tmp_path, changes, match):
    audit, annotations, rows = inputs(tmp_path)
    rows[0].update(changes)
    save(annotations, rows)
    output = tmp_path / 'folds.json'
    with pytest.raises(ValueError, match=match):
        prepare_folds(audit, annotations, output)
    assert not output.exists()


def test_missing_row_and_failed_audit_block(tmp_path):
    audit, annotations, rows = inputs(tmp_path)
    save(annotations, rows[:-1])
    with pytest.raises(ValueError, match='every audited'):
        prepare_folds(audit, annotations, tmp_path / 'out.json')
    save(annotations, rows)
    data = json.loads(audit.read_text())
    data['failed_count'] = 1
    audit.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='failures'):
        prepare_folds(audit, annotations, tmp_path / 'out.json')


def predictions(tmp_path):
    audit, annotations, _ = inputs(tmp_path)
    manifest = tmp_path / 'folds.json'
    prepared = prepare_folds(audit, annotations, manifest)
    rows = [dict(pixel_hash=s['pixel_hash'], fold=s['fold'], **{k: '0.9' for k in LABELS})
            for s in prepared['samples']]
    path = tmp_path / 'probabilities.csv'
    save(path, rows, ('pixel_hash', 'fold', *LABELS))
    return manifest, path, rows


def test_evaluate_known_labels_only(tmp_path):
    manifest, path, _ = predictions(tmp_path)
    result = evaluate_predictions(manifest, path, tmp_path / 'metrics.json')
    metrics = result['metrics']['per_label']
    assert metrics['spine_position']['f1'] == 1
    assert metrics['spine_position']['n'] == 4
    assert metrics['spine_position']['roc_auc'] is None
    assert metrics['hip_roi']['n'] == 0
    assert metrics['hip_roi']['f1'] is None
    assert metrics['spine_axis']['n'] == 0


@pytest.mark.parametrize('change,match', [('fold', 'fold'), ('missing', 'exactly'), ('duplicate', 'Duplicate'),
                                        ('nan', 'Invalid'), ('range', 'Invalid')])
def test_prediction_checks(tmp_path, change, match):
    manifest, path, rows = predictions(tmp_path)
    if change == 'fold':
        rows[0]['fold'] = 99
    elif change == 'missing':
        rows.pop()
    elif change == 'duplicate':
        rows.append(rows[0])
    else:
        rows[0]['spine_axis'] = 'nan' if change == 'nan' else '1.2'
    save(path, rows, ('pixel_hash', 'fold', *LABELS))
    with pytest.raises(ValueError, match=match):
        evaluate_predictions(manifest, path, tmp_path / 'out.json')
    assert not (tmp_path / 'out.json').exists()


def test_cli_full_preparation_and_evaluation(tmp_path):
    audit, annotations, _ = inputs(tmp_path)
    manifest = tmp_path / 'folds.json'
    assert main(['prepare-folds', str(audit), '--annotations', str(annotations), '--output', str(manifest)]) == 0
    assert main(['annotation-template', str(audit), '--output', str(annotations)]) == 2
    samples = json.loads(manifest.read_text())['samples']
    probs = tmp_path / 'predictions.csv'
    save(probs, [dict(pixel_hash=s['pixel_hash'], fold=s['fold'], **{k: '0.5' for k in LABELS}) for s in samples],
         ('pixel_hash', 'fold', *LABELS))
    assert main(['evaluate', str(manifest), '--predictions', str(probs), '--output', str(tmp_path / 'metrics.json')]) == 0
