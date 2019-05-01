# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocab_size, dropout, device,
                 lstm_num_hidden=256, lstm_num_layers=2):
        super(TextGenerationModel, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = lstm_num_hidden
        self.num_layers = lstm_num_layers
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.device = device
        self.model = nn.LSTM(input_size=self.vocab_size,
                             hidden_size=self.hidden_size,
                             num_layers=self.num_layers,
                             batch_first=True,
                             dropout=self.dropout)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x):
        # Get all hidden states (B, L, hidden_size)
        h, (_, _) = self.model(x)

        # Map hidden states to predictions (B, L, vocab_size)
        y = self.linear(h)
        return y
