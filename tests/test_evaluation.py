import numpy as np
import pytest
from kosti.evaluation import grouped_folds, binary_metrics, multilabel_metrics, grouped_bootstrap_ci, linked_study_groups, masked_bce_with_logits


def test_transitive_duplicates_keep_studies_together():
    study = ["a", "a", "b", "b", "c", "d"]
    hashes = ["x", "y", "y", "z", "w", "v"]
    folds = grouped_folds(study, hashes, n_splits=3)
    assert len(set(folds[:4])) == 1
    assert len(set(folds)) == 3
    np.testing.assert_array_equal(folds, grouped_folds(study, hashes, n_splits=3))
    with pytest.raises(ValueError):
        grouped_folds(study, hashes, n_splits=4)
    with pytest.raises(ValueError):
        grouped_folds(["a", ""], ["x", "y"], n_splits=2)


def test_metrics_mask_missing_and_credit_auc_ties():
    m = binary_metrics([1, 0, np.nan, 1, 0], [.5, .5, .9, 1, 0])
    assert m["n"] == 4 and m["tp"] == 2 and m["fp"] == 1
    assert m["f1"] == .8 and m["roc_auc"] == .875
    assert binary_metrics([0, 0], [.1, .2])["roc_auc"] is None
    assert binary_metrics([np.nan], [.1])["f1"] is None
    with pytest.raises(ValueError):
        binary_metrics([1], [float("nan")])


def test_macro_excludes_entirely_missing_label():
    m = multilabel_metrics([[1, np.nan], [0, np.nan]], [[.9, .1], [.1, .9]], ["known", "missing"])
    assert m["macro_f1"] == 1
    assert m["per_label"]["missing"]["n"] == 0


def test_bootstrap_is_reproducible_and_reports_undefined_draws():
    args = ([1, 1, 0, 0], [.9, .9, .1, .1], ["a", "a", "b", "b"])
    result = grouped_bootstrap_ci(*args, metric="roc_auc", repeats=100)
    assert result == grouped_bootstrap_ci(*args, metric="roc_auc", repeats=100)
    assert result["lower"] == result["upper"] == 1
    assert 0 < result["valid_resamples"] < 100


def test_masked_loss_does_not_train_unknown_targets():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0., 5.]], requires_grad=True)
    loss = masked_bce_with_logits(logits, torch.tensor([[1., float("nan")]]))
    loss.backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] == 0
    empty = masked_bce_with_logits(logits, torch.full_like(logits, float("nan")))
    assert empty.item() == 0


def test_average_precision_groups_ties_and_balanced_accuracy():
    metrics = binary_metrics([1, 0, 1, 0], [.9, .8, .8, .1])
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["balanced_accuracy"] == .75
    assert binary_metrics([0, 0], [.1, .2])["average_precision"] is None


def test_bootstrap_rejects_missing_groups():
    with pytest.raises(ValueError):
        grouped_bootstrap_ci([0, 1], [.1, .9], ["a", ""], repeats=10)


def test_bootstrap_groups_connect_studies_through_exact_copies():
    samples = [
        {'copies': [{'study_uid': 'a', 'pixel_hash': 'x'}, {'study_uid': 'b', 'pixel_hash': 'x'}]},
        {'copies': [{'study_uid': 'b', 'pixel_hash': 'y'}]},
        {'copies': [{'study_uid': 'c', 'pixel_hash': 'z'}]},
    ]
    groups = linked_study_groups(samples)
    assert groups[0] == groups[1] != groups[2]
    with pytest.raises(ValueError, match='duplicate provenance'):
        linked_study_groups([{'copies': []}])
