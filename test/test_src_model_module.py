import torch

from src.model.module import *


def test_DinoBackbone1():
    spatial_encoder = DinoBackbone(
        "model/facebook/dinov2-base",
        224,
        [2, 6, 10],
    ).to("cuda:3")

    x = torch.randn(6, 3, 224, 224).to("cuda:3")

    y = spatial_encoder(x)

    assert tuple(y.shape) == (6, 257, 768)


def test_DinoBackbone2():
    spatial_encoder = DinoBackbone(
        "model/facebook/dinov2-base",
        224,
        None,
    ).to("cuda:3")

    x = torch.randn(6, 3, 224, 224).to("cuda:3")

    y = spatial_encoder(x)

    assert tuple(y.shape) == (6, 257, 768)


if __name__ == "__main__":
    test_DinoBackbone1()
    test_DinoBackbone2()