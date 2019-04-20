"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Pytorch and settings
import torch
from torch import nn

torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
        """

        super(ConvNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # TODO: crosscheck this parameter
        maxpooling_dilation = 1

        self.conv = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=self.n_channels, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # maxpool1
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1, dilation=maxpooling_dilation),
            # conv2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            # maxpool2
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1, dilation=maxpooling_dilation),
            # conv3_a
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            # conv3_b
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            # maxpool3
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1, dilation=maxpooling_dilation),
            # conv4_a
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            # conv4_b
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            # maxpool4
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1, dilation=maxpooling_dilation),
            # conv5_a
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            # conv5_b
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            # maxpool5
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1, dilation=maxpooling_dilation),
            # avgpool
            nn.AvgPool2d(kernel_size=(1, 1), stride=1, padding=0),
        )
        # linear
        self.linear = nn.Linear(512, self.n_classes)

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """

        out = self.conv(x)
        out = self.linear(out.squeeze())

        return out
