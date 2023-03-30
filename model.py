import torch.nn as nn
import torch

from constants import num_classes


class SegmentationModel(nn.Module):
    """
    A segmentation model that takes an input image and outputs a segmented image.

    Args:
        in_channels (int): The number of input channels.
        num_classes (int): The number of output classes.
    """
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = nn.Identity()
        self.decoder = UpSample(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward method of the SegmentationModel.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The segmented output tensor.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class UpSample(nn.Module):
    """
    A class head that upsamples the input tensor.

    Args:
        in_channels (int): The number of input channels.
        num_classes (int): The number of output classes.
    """
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.trns1 =nn.ConvTranspose2d(in_channels,360,kernel_size=3, stride = 2)
        self.conv1 = nn.Conv2d(360, 180, kernel_size=3)
        self.B1 = nn.BatchNorm2d(180)
        self.pool = nn.AdaptiveAvgPool2d((512,1024))
        self.trns2 =nn.ConvTranspose2d(180,90,kernel_size=3, stride = 2)
        self.conv2 = nn.Conv2d(90, num_classes, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward method of the UpSample.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x =self.trns1(x)
        x = self.conv1(x)
        x = self.B1(x)
        x = self.relu(x)
        x =self.trns2(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
    
