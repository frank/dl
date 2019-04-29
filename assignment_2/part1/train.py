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

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# You may want to look into tensorboardX for logging
from tensorboardX import SummaryWriter

import pickle
import os


################################################################################

def get_accuracy(predictions, t):
    y = torch.argmax(predictions, 1)
    assert len(y) == len(t)
    accuracy = sum(y == t).item() / len(y)
    return accuracy


def save_results(accuracies, losses, run_id, model_type, input_length, accuracy):
    dir_name = 'results/' + model_type.lower() + '/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    with open(dir_name + run_id + '.pkl', 'wb') as file:
        pickle.dump({'accuracies': accuracies,
                     'losses': losses,
                     'input_length': input_length,
                     'final_accuracy': accuracy}, file)


def train(config, device="cpu"):
    assert config.model_type in ('RNN', 'LSTM')

    # Tensorboard summary writer
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_" + config.model_type.lower() + '_' + str(config.input_length))
    log_dir = 'tensorboard/' + config.model_type.lower() + '/' + run_id
    writer = SummaryWriter(log_dir=log_dir)

    # Torch settings
    if device == 'cpu':
        torch.set_default_tensor_type(torch.FloatTensor)
    elif device == 'cuda:0':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    dtype = torch.float

    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        model = VanillaRNN(config.input_length, config.input_dim,
                           config.num_hidden, config.num_classes,
                           config.batch_size, device=device).to(device)
    elif config.model_type == 'LSTM':
        model = LSTM(config.input_length, config.input_dim,
                     config.num_hidden, config.num_classes,
                     config.batch_size, device=device).to(device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=config.learning_rate)

    # Accuracy and loss to be saved
    accuracies = []
    losses = []

    # Useful for convergence check
    avg_range = 200
    last_accuracy = 0
    convergence_threshold = 1e-4

    model.train()
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # Load batches in the GPU
        batch_inputs = batch_inputs.to(device=device)
        batch_targets = batch_targets.to(device=device)

        # Forward pass
        predictions = model.forward(batch_inputs)

        # Compute loss
        loss = criterion(predictions, batch_targets)

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

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)

        # Add accuracy and loss to the writer
        writer.add_scalars('accuracy_and_loss', {'acc': accuracy, 'loss': loss}, step)

        # Store accuracy and loss
        accuracies.append(accuracy)
        losses.append(loss)

        # Print information
        if step % 100 == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                config.train_steps, config.batch_size, examples_per_second,
                accuracy, loss
            ))

        # Check for convergence
        if step % avg_range == 0 and step != 0:
            avg_accuracy = np.mean(accuracies[-avg_range:])
            if np.abs(avg_accuracy - last_accuracy) < convergence_threshold:
                print("The model has converged with accuracy", avg_accuracy, "(" +
                      ("+" if avg_accuracy > last_accuracy else "-") + str(np.abs(avg_accuracy - last_accuracy)) + ")")
                break
            last_accuracy = avg_accuracy

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    save_results(accuracies, losses, run_id, config.model_type, config.input_length, last_accuracy)
    writer.close()
    print('Done training. Accuracy:', avg_accuracy)


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config, device=config.device)
