import torch
from torch import nn
# from tools.compute_metrics_norm import pesq2
import numpy as np
from joblib import Parallel, delayed
# from pesq import pesq, PesqError


def pesq_loss(clean, noisy, sr=16000):
    clean = clean.numpy()
    noisy = noisy.numpy()
    pesq_score = pesq2(clean, noisy, sr, 'wideband')

    return pesq_score


def pkg_pesq_loss(clean, noisy, sr=16000):
    clean = clean.numpy()
    noisy = noisy.numpy()
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        pesq_score = -1
    return pesq_score


def batch_pesq(clean, noisy, device, mode='pkg'):
    if mode == 'pkg':
        pesq_score = Parallel(n_jobs=-1)(delayed(pkg_pesq_loss)(c, n)
                                         for c, n in zip(clean, noisy))
    else:
        pesq_score = Parallel(n_jobs=-1)(delayed(pesq_loss)(c, n)
                                         for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to(device)


class Discriminator_Stride2_SN(nn.Module):
    def __init__(self, ndf, in_channel=2):
        super().__init__()
        ndf = ndf//2
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channel, ndf, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf, ndf*2, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf*2, ndf*4, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf*4, ndf*8, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf*8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf*8, ndf*4)),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(ndf*4, 1)),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        # conditional GAN
        xy = torch.cat([x, y], dim=1)  # to shape (batch, channel, H, W)
        return self.layers(xy)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=2):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
        """
        super(NLayerDiscriminator, self).__init__()

        use_bias = False
        kw = 4
        padw = 1
        # try padding mode reflect
        sequence = [nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw,
                                                     stride=2, padding=padw, padding_mode="reflect")), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                                 kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                nn.InstanceNorm2d(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                             kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        sequence += [nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult, 1,
                                                      kernel_size=kw, stride=2, padding=padw)),
                     nn.Sigmoid()]
        # apply sigmoid at the end or
        # BETTER: Use BCEWITHLOGITSLOSS, comibing sigmoid and BCE loss
        self.model = nn.Sequential(*sequence)

    def forward(self, x, y):
        """Standard forward."""
        xy = torch.cat([x, y], dim=1)  # to shape (batch, channel, H, W)
        # return size is [1,1,16,16] (16x16 patch discriminator)
        return self.model(xy)


def test():
    x = torch.randn((2, 1, 256, 256))
    y = torch.randn((2, 1, 256, 256))
    model = NLayerDiscriminator(input_nc=2)
    preds = model(x, y)
    print(model)
    print(preds.shape)
    preds_mean = torch.mean(preds, (2, 3))
    # print(preds)
    print(preds_mean.shape)


if __name__ == "__main__":
    test()
