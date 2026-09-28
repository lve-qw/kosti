"""Small synthetic checks for offline experiments; no patient data required."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'scripts' / (name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_regularized_probe_masks_unknown_labels():
    torch = pytest.importorskip('torch')
    probe = script('fit_nested_probe')
    x = torch.tensor([[-2., 1.], [-1., 1.], [1., 1.], [2., 1.]])
    y = torch.tensor([0., 0., 1., 1.])
    w, b = probe.fit(x, y, .1)
    wx, bx = probe.fit(torch.cat([x, torch.tensor([[100., -100.]])]),
                       torch.cat([y, torch.tensor([float('nan')])]), .1)
    torch.testing.assert_close(w, wx)
    torch.testing.assert_close(b, bx)
    assert ((x @ w + b > 0).float() == y).all()
    assert torch.isfinite(w).all() and torch.isfinite(b)


def test_probe_requires_inner_folds_and_observed_training_data():
    torch = pytest.importorskip('torch')
    probe = script('fit_nested_probe')
    x, y = torch.ones(4, 2), torch.zeros(4)
    with pytest.raises(ValueError, match='inner folds'):
        probe.select(x, y, torch.zeros(4))
    with pytest.raises(ValueError, match='No observed'):
        probe.fit(x, torch.full((4,), float('nan')), .1)
    with pytest.raises(ValueError, match='binary'):
        probe.fit(x, torch.full((4,), 2.), .1)


def test_cached_last_block_matches_full_model_without_bn_drift():
    torch = pytest.importorskip('torch')
    from torchvision.models import resnet18
    torch.set_num_threads(1)
    model = resnet18(weights=None).eval()
    x = torch.randn(2, 3, 32, 32)
    stem = torch.nn.Sequential(*list(model.children())[:-3])
    head = torch.nn.Sequential(model.layer4, model.avgpool, torch.nn.Flatten(), model.fc)
    before = model.layer4[0].bn1.running_mean.clone()
    with torch.no_grad():
        torch.testing.assert_close(model(x), head(stem(x)))
    assert torch.equal(before, model.layer4[0].bn1.running_mean)


def test_axis_geometry_uses_physical_aspect_ratio():
    axis = script('check_axis_geometry')
    yy, xx = np.mgrid[:200, :160]
    image = np.exp(-((xx - (65 + .15*yy))/8)**2)
    angle = axis.estimate_axis(image, (1., 1.))
    physical = axis.estimate_axis(image, (2., 1.))
    assert 6 < angle < 11
    assert 2.5 < physical < 6
    assert axis.estimate_axis(np.exp(-((xx-80)/8)**2), (1.,1.)) < .01
