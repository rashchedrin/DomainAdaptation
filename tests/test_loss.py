import numpy as np
import torch
from .dann_loss import _loss_DANN


def test_loss_DANN_():
    cpl = torch.Tensor(
        np.array([
            [1.0, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ]))
    lpt = torch.Tensor(np.array([-5.1, 5, -6, 8]))
    ils = np.array([1, 2, -100, 2], dtype='int')
    ist = np.array([0, 1, 0, 1], dtype='int')
    assert abs(0.7448 - _loss_DANN(cpl, lpt, ils, ist, 1, 1)) < 1e-4
    print("OK test_loss_DANN_")
    return True
