import numpy as np
import pytest
from kosti.evaluation import calibration_metrics


def test_calibration_masks_unknown_and_includes_probability_one():
    report = calibration_metrics([0, 1, np.nan], [0, 1, .8])
    assert report['n'] == 2
    assert report['brier_score'] == report['ece'] == 0
    assert sum(b['n'] for b in report['bins']) == 2


def test_calibration_reports_overconfidence():
    report = calibration_metrics([0, 1], [.9, .9])
    assert report['brier_score'] == pytest.approx(.41)
    assert report['ece'] == pytest.approx(.4)


def test_unobserved_calibration_is_undefined():
    assert calibration_metrics([np.nan], [.4])['brier_score'] is None
