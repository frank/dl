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

import os
import time
from datetime import datetime
import argparse

import numpy as np
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

from tensorboardX import SummaryWriter

################################################################################

# Some lighthearted tomfoolery
quit_msgs = ["Have it your way! I quit.",
             "Stop! I can't breathe!",
             "Why start the job only to end it so soon?",
             "\"Pain is temporary. Quitting lasts forever.\" -Lance Armstrong",
             "\"If you quit once it becomes a habit. Never quit.\" -Michael Jordan",
             "\"A man is not finished when he is defeated. He is finished when he quits.\" -Richard Nixon"]


def get_one_hot(batch, batch_size, seq_length, vocab_size):
    one_hot = torch.zeros(batch_size, seq_length, vocab_size).scatter_(2, batch.unsqueeze(-1), 1)
    return one_hot


def get_accuracy(predictions, t):
    y = torch.argmax(predictions, 2).flatten()
    accuracy = sum(y == t.flatten()).item() / len(y)
    return accuracy


def train():
    # Torch settings
    device = torch.device(config.device)
    if device == 'cpu':
        torch.set_default_tensor_type(torch.FloatTensor)
    elif device == 'cuda:0':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    dtype = torch.float

    # Tensorboard summary writer
    if config.tensorboard:
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_" + config.model_type.lower() + '_' + str(config.input_length))
        log_dir = 'tensorboard/' + config.model_type.lower() + '/' + run_id
        writer = SummaryWriter(log_dir=log_dir)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Model parameters
    # TODO: LR decay
    # TODO: dropout and weight decay
    lr = config.learning_rate
    lr_decay = config.learning_rate_decay
    lr_step = config.learning_rate_step
    dropout = 1.0 - config.dropout_keep_prob

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size,
                                config.seq_length,
                                dataset.vocab_size,
                                dropout,
                                device).to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    try:
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            # Get batches as tensors of size (batch_size x seq_length)
            batch_inputs = torch.stack(batch_inputs).permute((1, 0))
            batch_targets = torch.stack(batch_targets).permute((1, 0)).to(device)

            # Convert batches to one-hot representation (batch_size x seq_length x vocab_size)
            batch_inputs = get_one_hot(batch_inputs,
                                       config.batch_size,
                                       config.seq_length,
                                       dataset.vocab_size).to(device)

            # Forward pass
            model.train()
            predictions = model.forward(batch_inputs)

            # Compute loss
            loss = criterion(predictions.permute(0, 2, 1), batch_targets)

            # Reset gradients before backwards pass
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Clipping gradients to avoid exploding gradient problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

            # Update weights
            optimizer.step()

            # Compute accuracy
            accuracy = get_accuracy(predictions, batch_targets)

            # Add accuracy and loss to the writer
            if config.tensorboard:
                writer.add_scalars('Accuracy_and_Loss', {'accuracy': accuracy, 'loss': loss}, step)
                writer.add_scalar('Learning_Rate', lr, step)

            # Update learning rate
            if (step % lr_step == 0) and step != 0:
                lr *= lr_decay
                for group in optimizer.param_groups:
                    group['lr'] = lr

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size / float(t2 - t1)

            if step % config.print_every == 0:
                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                                step,
                                                                int(config.train_steps),
                                                                config.batch_size,
                                                                examples_per_second,
                                                                accuracy,
                                                                loss))

            if (step % config.sample_every == 0) and step != 0:
                model.eval()
                # TODO: Generate some sentences by sampling from the model
                pass

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

        print('Done training.')
    except (KeyboardInterrupt, BrokenPipeError):
        print("\n" + random.choice(quit_msgs))


################################################################################
################################################################################

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default='data/dumas.txt', help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to be used to run the model')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=10, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=1000, help='How often to sample from the model')
    parser.add_argument('--tensorboard', type=bool, default=False, help='Whether to store the tensorboard data')

    config = parser.parse_args()

    # Train the model
    train()
