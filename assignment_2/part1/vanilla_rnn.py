################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn


# Torch settings
torch.set_default_tensor_type(torch.cuda.FloatTensor)  # fixme
dtype = torch.FloatTensor


################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()

        # Non-trainable parameters
        self.seq_length = seq_length
        self.input_size = input_dim
        self.hidden_size = num_hidden
        self.output_size = num_classes
        self.batch_size = batch_size
        self.device = device

        stdv = 1.0 / math.sqrt(num_hidden)

        # Trainable parameters
        self.w_hx = nn.Parameter(stdv * torch.randn(num_hidden, input_dim))
        self.w_hh = nn.Parameter(stdv * torch.randn(num_hidden, num_hidden))
        self.b_h = nn.Parameter(stdv * torch.randn(num_hidden))
        self.w_ph = nn.Parameter(stdv * torch.randn(num_classes, num_hidden))
        self.b_p = nn.Parameter(stdv * torch.randn(num_classes))

    def forward(self, x):
        assert x.size()[1] == self.seq_length

        # Initialize 0th hidden state
        prev_h = torch.zeros(self.hidden_size, self.batch_size)

        # Go over each digit in the sequence
        for t in range(self.seq_length):
            x_t = x[:, t].unsqueeze(0)
            h_t = torch.tanh(torch.mm(self.w_hx, x_t) + torch.mm(self.w_hh, prev_h) + self.b_h)
            prev_h = h_t

        # Get output
        p = torch.mm(self.w_ph, h_t) + self.b_p[:, None]

        return p.permute(1, 0)
