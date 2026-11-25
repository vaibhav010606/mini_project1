"""Convolutional autoencoders over 5-frame clips.

BigAE      - "cloud" model (wider, more accurate)
StudentAE  - "edge" model (about 15x fewer parameters), trained with knowledge distillation
"""
import torch
import torch.nn as nn


def _down(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, inplace=True))


def _up(i, o):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class ConvAE(nn.Module):
    def __init__(self, in_ch: int = 5, widths=(64, 128, 256, 256)):
        super().__init__()
        enc, c = [], in_ch
        for w in widths:
            enc.append(_down(c, w))
            c = w
        self.encoder = nn.Sequential(*enc)
        dec = []
        rev = list(widths[::-1]) + [widths[0]]
        for a, b in zip(rev[:-1], rev[1:]):
            dec.append(_up(a, b))
        self.decoder = nn.Sequential(*dec, nn.Conv2d(widths[0], in_ch, 3, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))


def BigAE():
    return ConvAE(widths=(64, 128, 256, 256))


def StudentAE():
    return ConvAE(widths=(16, 32, 64, 64))


MODELS = {"big": BigAE, "student": StudentAE}


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for k, f in MODELS.items():
        m = f()
        y = m(torch.zeros(2, 5, 128, 128))
        print(k, n_params(m), tuple(y.shape))
