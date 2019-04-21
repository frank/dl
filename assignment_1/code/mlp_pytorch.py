"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

from custom_batchnorm import CustomBatchNormAutograd

# Torch settings
torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, dropout=0., batchnorm=False):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
        """
        super(MLP, self).__init__()
        # list with all of the sizes between layers
        self.layer_sizes = [n_inputs] + n_hidden + [n_classes]

        # list to append all layers to
        layers = []
        for layer_n in range(len(self.layer_sizes) - 1):
            if layer_n < (len(self.layer_sizes) - 2):
                # every hidden layer also gets a ReLU nonlinearity
                layers.append(nn.Linear(self.layer_sizes[layer_n],
                                        self.layer_sizes[layer_n + 1]))
                layers.append(nn.ReLU())
                layers.append(CustomBatchNormAutograd(self.layer_sizes[layer_n + 1]))
                if dropout > 0.:
                    layers.append(nn.Dropout(0.2))
            else:
                # the output layer gets no ReLU
                layers.append(nn.Linear(self.layer_sizes[layer_n],
                                        self.layer_sizes[layer_n + 1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        out = self.model(x)
        return out
