import matplotlib.pyplot as plt
import pickle
import sys
import os


def plot_vanilla():
    with open('results/rnn/results_5_35.pkl', 'rb') as file:
        data = pickle.load(file)

    input_lengths = data['input_lengths']
    accuracies = data['accuracies']

    fig = plt.figure()
    plt.plot(input_lengths, accuracies)
    plt.title("Vanilla RNN Accuracy vs Input Length", fontsize=14)
    plt.xlabel("Input length")
    plt.ylabel("Accuracy")
    fig.set_size_inches(7.5, 4.5)
    plt.show()


def plot_lstm():
    with open('results/lstm/results_5_35.pkl', 'rb') as file:
        data = pickle.load(file)

    input_lengths = data['input_lengths']
    accuracies = data['accuracies']

    fig = plt.figure()
    plt.plot(input_lengths, accuracies)
    plt.title("LSTM Accuracy vs Input Length", fontsize=14)
    plt.xlabel("Input length")
    plt.ylabel("Accuracy")
    fig.set_size_inches(7.5, 4.5)
    plt.show()


if __name__ == '__main__':
    plot_lstm()
