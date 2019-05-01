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

import torch
import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocab_size, dropout=0.0, device='cpu',
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

    def sample_char(self, c_prob, temp):
        # Sample from the softmax distribution using the temperature parameter
        c_prob = torch.softmax(c_prob / temp, 0)
        return torch.multinomial(c_prob, 1)

    def get_one_hot(self, c_ix):
        c_oh = torch.zeros(self.vocab_size, device=self.device)
        c_oh[c_ix] = 1
        return c_oh

    def generate(self, c, temp):
        sentence = []

        # Initialize starting hidden and cell states
        h_n = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)
        c_n = torch.zeros_like(h_n)

        for t in range(self.seq_length):
            # Get a hidden state (1, 1, hidden_size)
            h, (h_n, c_n) = self.model(c[None, None, :], (h_n, c_n))

            # Get next character's probability distribution y (vocab_size)
            y = self.linear(h.squeeze())

            # Get next character's index
            next_c_ix = self.sample_char(y, temp)

            # Get next character's one-hot representation
            c = self.get_one_hot(next_c_ix)

            sentence.append(next_c_ix.item())

        return sentence

    def cmd_generate(self, start, temp, length=30):
        # start is one-hot encoded (length x vocab_size)
        sentence = []

        # Get all hidden states (1, L, hidden_size) and initial hidden and cell states for the next step
        h, (h_n, c_n) = self.model(start[None, :, :])

        # Get output
        p = self.linear(h[0, -1, :])

        # Get first new character
        c_ix = self.sample_char(p, temp)
        sentence.append(c_ix.item())
        c = self.get_one_hot(c_ix)

        for t in range(length):
            # Get a hidden state (1, 1, hidden_size)
            h, (h_n, c_n) = self.model(c[None, None, :], (h_n, c_n))

            # Get next character's probability distribution y (vocab_size)
            y = self.linear(h.squeeze())

            # Get next character's index
            next_c_ix = self.sample_char(y, temp)

            # Get next character's one-hot representation
            c = self.get_one_hot(next_c_ix)

            sentence.append(next_c_ix.item())

        return sentence
