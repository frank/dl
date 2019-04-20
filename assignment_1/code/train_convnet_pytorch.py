"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch
from convnet_pytorch import ConvNet
import cifar10_utils

from tensorboardX import SummaryWriter
from datetime import datetime

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

# Torch settings
torch.set_default_tensor_type(torch.cuda.FloatTensor)
dtype = torch.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters
n_channels = 3
n_classes = 10


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    """
    _, y = predictions.max(1)
    _, t = targets.max(1)
    if len(y) != len(t):
        print("WARNING: size mismatch between predictions and targets")
    accuracies = [1 if y[i] == t[i] else 0 for i in range(len(y))]
    return np.mean(accuracies)


def train():
    """
    Performs training and evaluation of ConvNet model.

    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    # DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    # initialize tensorboard
    tb_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_convnet")
    log_dir = 'tensorboard/' + tb_name
    writer = SummaryWriter(log_dir=log_dir)

    # get the dataset
    print("Loading data...", end='')
    data_set = cifar10_utils.get_cifar10(FLAGS.data_dir)
    print("Loaded data    ")

    # get the necessary components
    classifier = ConvNet(n_channels, n_classes).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=FLAGS.learning_rate)

    max_epochs = FLAGS.max_steps
    n_batches = {'train': int(data_set['train']._num_examples / FLAGS.batch_size),
                 'validation': int(data_set['validation']._num_examples / FLAGS.batch_size),
                 'test': int(data_set['test']._num_examples / FLAGS.batch_size)
    }

    # list of training accuracies and losses
    train_accuracies = []
    train_losses = []

    # list of test accuracies and losses
    test_accuracies = []
    test_losses = []

    # training loop
    for epoch in range(max_epochs):

        print("Epoch", epoch+1)

        # list of training batch accuracies and losses for this epoch
        epoch_train_accuracies = []
        epoch_train_losses = []

        # list of test batch accuracies and losses for this epoch
        epoch_test_accuracies = []
        epoch_test_losses = []

        # iterate over the dataset
        for batch in range(n_batches['train']):
            # get current batch...
            images, labels = data_set['train'].next_batch(FLAGS.batch_size)

            # ...in the gpu
            images = torch.from_numpy(images).type(dtype).to(device=device)
            labels = torch.from_numpy(labels).type(dtype).to(device=device)

            # forward pass
            predictions = classifier.forward(images)

            # compute loss
            class_labels = labels.argmax(dim=1)
            loss = loss_function(predictions, class_labels)

            # reset gradients before backwards pass
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()

            # get accuracy and loss for the batch
            epoch_train_accuracies.append(accuracy(predictions, labels))
            epoch_train_losses.append(loss.item())

        # store accuracy and loss for this epoch
        epoch_train_accuracy = np.mean(epoch_train_accuracies)
        train_accuracies.append(epoch_train_accuracy)
        writer.add_scalar("Training accuracy vs epochs", epoch_train_accuracy, epoch)

        epoch_train_loss = np.mean(epoch_train_losses)
        train_losses.append(epoch_train_loss)
        writer.add_scalar("Training loss vs epochs", epoch_train_accuracy, epoch)

        print("\tTRAIN:", round(epoch_train_accuracy * 100, 1), "%")

        # run evaluation every eval_freq epochs
        if epoch % FLAGS.eval_freq == 0:

            # get accuracy on the test set
            for batch in range(n_batches['test']):
                # get current batch...
                images, labels = data_set['test'].next_batch(FLAGS.batch_size)

                # ...in the gpu
                images = torch.from_numpy(images).type(dtype).to(device=device)
                labels = torch.from_numpy(labels).type(dtype).to(device=device)

                # forward pass
                predictions = classifier(images)

                # compute loss
                class_labels = labels.argmax(dim=1)
                loss = loss_function(predictions, class_labels)

                # get accuracy and loss for the batch
                epoch_test_accuracies.append(accuracy(predictions, labels))
                epoch_test_losses.append(loss.item())

            # store accuracy and loss
            epoch_test_accuracy = np.mean(epoch_test_accuracies)
            test_accuracies.append(epoch_test_accuracy)
            writer.add_scalar("Test accuracy vs epochs", epoch_test_accuracy, epoch)

            epoch_test_loss = np.mean(epoch_test_losses)
            test_losses.append(epoch_test_loss)
            writer.add_scalar("Test loss vs epochs", epoch_test_loss, epoch)

            print("\tTEST:", round(epoch_test_accuracy * 100, 1), "%")

    writer.close()


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
