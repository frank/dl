"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

from tensorboardX import SummaryWriter
import pickle as pkl
from datetime import datetime
import torch

# Torch settings
torch.set_default_tensor_type(torch.cuda.FloatTensor)
dtype = torch.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters
weight_decay = 1e-3
dropout = 0.2
batchnorm = True

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '512, 512, 512, 512'  # default: 100
LEARNING_RATE_DEFAULT = 1e-4  # default: 2e-3
MAX_STEPS_DEFAULT = 5000
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 500

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, labels):
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
    _, t = labels.max(1)
    if len(y) != len(t):
        print("WARNING: size mismatch between predictions and targets")
    accuracies = [1 if y[i] == t[i] else 0 for i in range(len(y))]
    return np.mean(accuracies)
    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    # DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    # Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    # -------------------------- UNCKECKED -------------------
    # initialize tensorboard
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_mlp")
    if batchnorm:
        run_id = run_id + '_batchnorm'
    log_dir = 'tensorboard/' + run_id
    writer = SummaryWriter(log_dir=log_dir)

    # get the dataset
    data_set = cifar10_utils.get_cifar10(FLAGS.data_dir)

    # get dataset information
    n_batches = {'train': int(data_set['train']._num_examples / FLAGS.batch_size),
                 'validation': int(data_set['validation']._num_examples / FLAGS.batch_size),
                 'test': int(data_set['test']._num_examples / FLAGS.batch_size)}
    image_shape = data_set['train'].images[0].shape
    n_inputs = image_shape[0] * image_shape[1] * image_shape[2]
    n_classes = data_set['train'].labels[0].shape[0]

    # get the necessary components
    classifier = MLP(n_inputs,
                     dnn_hidden_units,
                     n_classes,
                     dropout,
                     batchnorm).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(),
                                 lr=FLAGS.learning_rate,
                                 weight_decay=weight_decay)

    # list of training accuracies and losses
    train_accuracies = []
    train_losses = []

    # list of test accuracies and losses
    test_accuracies = []
    test_losses = []

    epoch_test_accuracy = 0
    epoch_test_loss = 0

    # training loop
    for step in range(FLAGS.max_steps):

        # get current batch...
        images, labels = data_set['train'].next_batch(FLAGS.batch_size)
        images = images.reshape(FLAGS.batch_size, n_inputs)

        # ...in the gpu
        images = torch.from_numpy(images).type(dtype).to(device=device)
        labels = torch.from_numpy(labels).type(dtype).to(device=device)

        # forward pass
        classifier.train()
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
        train_accuracy = accuracy(predictions, labels)
        train_accuracies.append(train_accuracy)

        writer.add_scalar("Training accuracy vs steps", train_accuracy, step)

        train_losses.append(loss.item())
        writer.add_scalar("Training loss vs steps", loss.item(), step)

        if ((step + 1) % 100) == 0 or step == 0:
            print("\nStep", step + 1)
            print("\tTRAIN:", round(train_accuracy * 100, 1), "%")

        # run evaluation every eval_freq epochs
        if (step + 1) % FLAGS.eval_freq == 0 or (step + 1) == FLAGS.max_steps:

            # list of test batch accuracies and losses for this step
            step_test_accuracies = []
            step_test_losses = []

            # get accuracy on the test set
            classifier.eval()
            for batch in range(n_batches['test']):
                # get current batch...
                images, labels = data_set['test'].next_batch(FLAGS.batch_size)
                images = images.reshape(FLAGS.batch_size, n_inputs)

                # ...in the gpu
                images = torch.from_numpy(images).type(dtype).to(device=device)
                labels = torch.from_numpy(labels).type(dtype).to(device=device)

                # forward pass
                predictions = classifier(images)

                # compute loss
                class_labels = labels.argmax(dim=1)
                loss = loss_function(predictions, class_labels)

                # get accuracy and loss for the batch
                step_test_accuracies.append(accuracy(predictions, labels))
                step_test_losses.append(loss.item())

            # store accuracy and loss
            epoch_test_accuracy = np.mean(step_test_accuracies)
            test_accuracies.append(epoch_test_accuracy)

            epoch_test_loss = np.mean(step_test_losses)
            test_losses.append(epoch_test_loss)

            print("\tTEST:", round(epoch_test_accuracy * 100, 1), "%")

        writer.add_scalar("Test accuracy vs epochs", epoch_test_accuracy, step)
        writer.add_scalar("Test loss vs epochs", epoch_test_loss, step)

    print("\nBest TEST:", round(max(test_accuracies) * 100, 1), "%")

    # save results
    results = {
        'train_accuracies': train_accuracies,
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'test_losses': test_losses,
        'eval_freq': FLAGS.eval_freq
    }

    if not os.path.exists("results/"):
        os.makedirs("results/")
    with open("results/" + run_id + "_results.pkl", "wb") as file:
        pkl.dump(results, file)

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
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
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
