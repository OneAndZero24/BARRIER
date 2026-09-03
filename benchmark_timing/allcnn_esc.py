"""
ESC-style AllCNN, key-compatible with the model released by the ESC repo
(https://github.com/KHU-VGI/ESC, models.py).

The ESC released checkpoints are saved as raw state_dicts with keys such as
    features.0.0.weight   (Conv2d inside a Conv = Sequential(Conv2d, BN, ReLU))
    features.0.1.weight   (BatchNorm2d)
    head.0.weight         (Linear inside head = Sequential(Linear))

This file is a verbatim mirror of the ESC AllCNN class (plus a `normalize`
module attribute that the SalUn/SEMU harnesses assign after construction),
so that ONE checkpount (the ESC cifar10_ori_allcnn.pth) can be loaded into
salun, semu, esc and the intact/barrier harness with the exact same weights.
"""

import torch
from torch import nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)


class AllCNN(nn.Module):
    """Identical architecture (and thus state_dict keys) to ESC's AllCNN."""

    def __init__(self, n_channels=3, num_classes=10, dropout=False, filters_percentage=0.5, batch_norm=True):
        super(AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)

        self.embed_dim = n_filter2

        # The SalUn/SEMU harnesses replace this with a NormalizeByChannelMeanStd
        # module right after construction (utils.setup_model_dataset).
        self.normalize = Identity()

        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )

        # for consistency with other models (ViT)
        self.head = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

    def forward(self, x, all=False):
        features = self.features(self.normalize(x))
        output = self.head(features)

        if all:
            res = dict()
            res['pre_logits'] = features
            res['logits'] = output
            return res

        return output