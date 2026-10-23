"""SegmentationNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x



class SegmentationNN(nn.Module):

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp or {}
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################

        # My comment: the dataset only applies ToTensor() (values in [0, 1], no
        # normalization allowed in the dataset class itself), so normalize here
        # with ImageNet stats to match what the pretrained backbone expects.
        self.register_buffer("mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

        # My comment: only 276 training images -> reuse ImageNet features via
        # transfer learning instead of training an encoder from scratch.
        mobilenet = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        features = mobilenet.features
        self.stem = features[0:2]      # stride 2,  16 channels
        self.stage1 = features[2:4]    # stride 4,  24 channels
        self.stage2 = features[4:7]    # stride 8,  32 channels
        self.stage3 = features[7:14]   # stride 16, 96 channels
        self.stage4 = features[14:18]  # stride 32, 320 channels

        base_channels = self.hp.get("base_channels", 48)
        c4, c3, c2, c1 = base_channels * 4, base_channels * 2, base_channels, base_channels // 2

        self.decoder4 = nn.Sequential(
            ConvLayer(320 + 96, c4),
            ConvLayer(c4, c4),
        )
        self.decoder3 = nn.Sequential(
            ConvLayer(c4 + 32, c3),
            ConvLayer(c3, c3),
        )
        self.decoder2 = nn.Sequential(
            ConvLayer(c3 + 24, c2),
            ConvLayer(c2, c2),
        )
        self.decoder1 = nn.Sequential(
            ConvLayer(c2 + 16, c1),
            ConvLayer(c1, c1),
        )

        # My comment: CrossEntropyLoss expects raw class scores, so no softmax is
        # applied here.
        self.classifier = nn.Conv2d(c1, num_classes, kernel_size=1)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################

        input_size = x.shape[2:]
        x = (x - self.mean) / self.std

        # My comment: Store skip features from every backbone stage so the
        # decoder can recover spatial detail after upsampling.
        skip0 = self.stem(x)
        skip1 = self.stage1(skip0)
        skip2 = self.stage2(skip1)
        skip3 = self.stage3(skip2)
        x = self.stage4(skip3)

        x = F.interpolate(x, size=skip3.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat((x, skip3), dim=1)
        x = self.decoder4(x)

        x = F.interpolate(x, size=skip2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat((x, skip2), dim=1)
        x = self.decoder3(x)

        x = F.interpolate(x, size=skip1.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat((x, skip1), dim=1)
        x = self.decoder2(x)

        x = F.interpolate(x, size=skip0.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat((x, skip0), dim=1)
        x = self.decoder1(x)

        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        x = self.classifier(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return x

    # @property
    # def is_cuda(self):
    #     """
    #     Check if model parameters are allocated on the GPU.
    #     """
    #     return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

if __name__ == "__main__":
    from torchinfo import summary
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")
