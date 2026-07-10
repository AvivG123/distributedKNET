import sys
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.DistributedKalmanNet import AdaptiveMeanConv


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
