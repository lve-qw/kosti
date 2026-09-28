import numpy as np

from kosti.geometry import DEFAULT_SCANNER_SPACING_MM, spine_axis_probability


def synthetic_axis(slope, h=200, w=160):
    yy, xx = np.mgrid[:h, :w]
    return np.exp(-((xx - (65 + slope*yy))/8)**2)


def test_spine_axis_score_tracks_physical_angle():
    straight = spine_axis_probability(synthetic_axis(0), (1., 1.))
    tilted = spine_axis_probability(synthetic_axis(.2), (1., 1.))
    anisotropic = spine_axis_probability(synthetic_axis(.2), (2., 1.))
    assert straight < anisotropic < tilted
    assert tilted > .5


def test_axis_geometry_fails_closed_on_unusable_pixels():
    assert spine_axis_probability(np.zeros((8, 8))) is None
    broken = np.zeros((100, 100)); broken[0, 0] = np.nan
    assert spine_axis_probability(broken) is None
    assert DEFAULT_SCANNER_SPACING_MM == (1.05, .6)
