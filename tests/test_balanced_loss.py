import torch
from kosti.evaluation import masked_bce_with_logits


def test_positive_weights_are_per_output_and_missing_targets_have_no_gradient():
    logits = torch.zeros((2, 2), requires_grad=True)
    targets = torch.tensor([[1., float('nan')], [0., 1.]])
    loss = masked_bce_with_logits(logits, targets, torch.tensor([3., 2.]))
    assert torch.allclose(loss, torch.log(torch.tensor(2.)) * 2)
    loss.backward()
    assert logits.grad[0, 1] == 0
    assert torch.allclose(logits.grad, torch.tensor([[-.5, 0.], [1/6, -1/3]]))
