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
torch.set_default_tensor_type(torch.FloatTensor)  # fixme
dtype = torch.FloatTensor
device = "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # fixme


################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()

        # Non-trainable parameters
        self.seq_length = seq_length
        self.input_size = input_dim
        self.hidden_size = num_hidden
        self.output_size = num_classes
        self.batch_size = batch_size
        self.device = device

        stdv = 1.0 / math.sqrt(num_hidden)

        # Trainable parameters
        self.w_g = nn.Parameter(stdv * torch.randn(num_hidden, num_hidden + input_dim))
        self.b_g = nn.Parameter(stdv * torch.randn(num_hidden))
        self.w_i = nn.Parameter(stdv * torch.randn(num_hidden, num_hidden + input_dim))
        self.b_i = nn.Parameter(stdv * torch.randn(num_hidden))
        self.w_f = nn.Parameter(stdv * torch.randn(num_hidden, num_hidden + input_dim))
        self.b_f = nn.Parameter(stdv * torch.randn(num_hidden))
        self.w_o = nn.Parameter(stdv * torch.randn(num_hidden, num_hidden + input_dim))
        self.b_o = nn.Parameter(stdv * torch.randn(num_hidden))
        self.w_p = nn.Parameter(stdv * torch.randn(num_hidden, num_hidden))
        self.b_p = nn.Parameter(stdv * torch.randn(num_hidden))

    def forward(self, x):
        assert x.size()[1] == self.seq_length

        # Initialize 0th hidden state
        prev_h = torch.zeros(self.hidden_size, self.batch_size)
        prev_c = torch.zeros(self.hidden_size, self.batch_size)

        # Go over each digit in the sequence
        for t in range(self.seq_length):
            hx_t = torch.cat((x[:, t].unsqueeze(0), prev_h), 0)
            g_t = torch.tanh(torch.mm(self.w_g, hx_t) + self.b_g)
            i_t = torch.sigmoid(torch.mm(self.w_i, hx_t) + self.b_i)
            f_t = torch.sigmoid(torch.mm(self.w_f, hx_t) + self.b_f)
            o_t = torch.sigmoid(torch.mm(self.w_o, hx_t) + self.b_o)
            c_t = g_t * i_t + prev_c * f_t
            h_t = torch.tanh(c_t) * o_t
            prev_c = c_t
            prev_h = h_t

        # Get output
        p = torch.mm(self.w_p, h_t) + self.b_p[:, None]

        return p.permute(1, 0)
