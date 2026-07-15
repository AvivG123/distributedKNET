import sys
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.DistributedKalmanNet import (
    AdaptiveMeanConv,
    EdgeKalmanFilter,
    loss_function,
)


def test_default_loss_preserves_legacy_flow():
    torch.manual_seed(0)
    x_pred = torch.randn(2, 3, 4, 2, 1)
    x_true = torch.randn(2, 3, 2, 1)
    diff_x = x_pred - x_true[..., None, :, :]
    expected = torch.sqrt(torch.sum(diff_x ** 2, dim=(-1, -2))).mean()

    assert torch.equal(loss_function(x_pred, x_true), expected)


def test_position_only_loss_has_finite_gradient_at_zero_error():
    x_pred = torch.zeros(1, 1, 1, 4, 1, requires_grad=True)
    x_true = torch.zeros(1, 1, 4, 1)

    loss = loss_function(x_pred, x_true, position_indices=[0, 2])
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(x_pred.grad).all()
    assert torch.count_nonzero(x_pred.grad) == 0


def test_adaptive_mean_conv_is_learnable():
    torch.manual_seed(0)

    x = torch.randn(4, 2)
    feat = torch.randn(4, 6)
    edge_index = torch.tensor(
        [
            [0, 2, 1, 3, 0, 1, 2, 3],
            [1, 1, 3, 3, 0, 0, 2, 2],
        ],
        dtype=torch.long,
    )

    adaptive_conv = AdaptiveMeanConv(in_channels=6, out_channels=2)

    adaptive_output = adaptive_conv(x, edge_index, feat)
    loss = adaptive_output.sum()
    loss.backward()

    assert adaptive_output.shape == x.shape
    assert any(
        param.grad is not None and torch.count_nonzero(param.grad).item() > 0
        for param in adaptive_conv.parameters()
    )


class _ScalarObservation:
    def __init__(self, node_count):
        self.node_count = node_count

    def __call__(self, state):
        return state

    def jacobian(self, state, n_expansions=0):
        return torch.ones(
            state.shape[0], 1, self.node_count, 1, 1,
            dtype=state.dtype, device=state.device,
        )


def test_edge_kalman_accepts_heterogeneous_per_node_noise():
    edge_filter = EdgeKalmanFilter(r_array=[0.1, 1.0], signal_dim=1)
    correction, _ = edge_filter(
        x_pred=torch.zeros(1, 2, 1, 1),
        measurements=torch.tensor([[[[10.0]], [[10.0]]]]),
        h_system=_ScalarObservation(node_count=2),
        adj_matrix=torch.ones(2, 2),
        node_number=2,
    )

    assert torch.allclose(correction, torch.full((1, 2, 1, 1), 505.0))
