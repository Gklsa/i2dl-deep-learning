"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_ch, out_ch, kernel=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).view(x.shape[0], x.shape[1], 1, 1)


class KeypointModel(nn.Module):
    """Facial keypoint detection: U-Net encoder-decoder with soft-argmax."""

    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server

        """
        super().__init__()
        self.hparams = hparams

        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architectures, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################

        # ---------- Encoder ----------
        # Stage 1: 96×96 → 48×48
        self.enc1 = nn.Sequential(
            _conv_bn_relu(1, 64),
            _conv_bn_relu(64, 64),
            nn.MaxPool2d(2),
        )
        # Stage 2: 48×48 → 24×24  (saved for skip connection)
        self.enc2 = nn.Sequential(
            _conv_bn_relu(64, 128),
            _conv_bn_relu(128, 128),
            nn.MaxPool2d(2),
        )
        # Stage 3: 24×24 → 12×12
        self.enc3 = nn.Sequential(
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
            nn.MaxPool2d(2),
        )
        # Bottleneck: 12×12 (dilated conv + SE channel attention)
        self.bottleneck = nn.Sequential(
            _conv_bn_relu(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            _SEBlock(256, reduction=16),
        )

        # ---------- Decoder ----------
        # Upsample 12×12 → 24×24, then concat with enc2 skip (128ch)
        self.up = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec = nn.Sequential(
            _conv_bn_relu(128 + 128, 128),   # 128 from up + 128 from skip
            nn.Dropout2d(0.1),
            _conv_bn_relu(128, 64),
        )

        # 1×1 conv: 64 channels → 15 heatmaps at 24×24
        self.heatmap_head = nn.Conv2d(64, 15, kernel_size=1)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def _soft_argmax(self, heatmaps):
        """Convert (N,15,H,W) heatmaps to (N,30) coordinates via soft-argmax."""
        N, K, H, W = heatmaps.shape
        h = F.softmax(heatmaps.reshape(N, K, -1), dim=-1).reshape(N, K, H, W)
        gx = torch.linspace(-1, 1, W, device=heatmaps.device)
        gy = torch.linspace(-1, 1, H, device=heatmaps.device)
        x = (h.sum(dim=2) * gx).sum(dim=-1)
        y = (h.sum(dim=3) * gy).sum(dim=-1)
        return torch.stack([x, y], dim=-1).reshape(N, 30)

    def forward(self, x, return_heatmaps=False):
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)

        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################

        # Encoder
        e1 = self.enc1(x)          # (N, 64,  48, 48)
        e2 = self.enc2(e1)         # (N, 128, 24, 24)  ← skip
        e3 = self.enc3(e2)         # (N, 256, 12, 12)
        b  = self.bottleneck(e3)   # (N, 256, 12, 12)

        # Decoder with skip connection
        d = self.up(b)                      # (N, 128, 24, 24)
        d = torch.cat([d, e2], dim=1)       # (N, 256, 24, 24)
        d = self.dec(d)                     # (N,  64, 24, 24)

        heatmaps = self.heatmap_head(d)       # (N, 15, 24, 24)
        coords = self._soft_argmax(heatmaps)  # (N, 30)
        if return_heatmaps:
            return coords, heatmaps
        return coords

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################


class DummyKeypointModel(nn.Module):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
