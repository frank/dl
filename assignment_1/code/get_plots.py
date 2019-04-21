import matplotlib.pyplot as plt
import pickle
import sys


def plot_convnet():
    # load data
    with open('results/2019-04-21_00-29-22_convnet_results.pkl', 'rb') as file:
        data = pickle.load(file)

    # make data readable
    train_accuracies = data['train_accuracies'] # 5000 values
    train_losses = data['train_losses'] # 5000 values
    test_accuracies = data['test_accuracies'] # 10 values
    test_losses = data['test_losses'] # 10 values
    eval_freq = data['eval_freq'] # 1 value = 500

    # print the final test accuracy and loss
    print("\nConvNet:")
    print("Best test accuracy:", round(max(test_accuracies) * 100, 1), "%")
    print("Best test loss:", round(min(test_losses), 5))

    # setting indices for training and testing
    train_indices = list(range(1, 5001))
    test_indices = list(range(500, 5001, eval_freq))

    # plotting
    fig, axs = plt.subplots(2, 1, sharex=True)

    # accuracy plot
    axs[0].plot(train_indices, train_accuracies, color='b', linewidth=0.2)
    axs[0].plot(test_indices, test_accuracies, color='r', linewidth=2.)
    axs[0].set_ylabel('Accuracy')
    axs[0].set_ylim(0, 1)
    axs[0].grid(True, linestyle=':')

    # loss plot
    axs[1].plot(train_indices, train_losses, color='b', linewidth=0.2)
    axs[1].plot(test_indices, test_losses, color='r', linewidth=2.)
    axs[1].set_xlabel('Steps')
    axs[1].set_ylabel('Loss')
    axs[1].grid(True, linestyle=':')

    fig.suptitle('Accuracy and Loss for the ConvNet Model', fontsize=14)
    axs[0].legend(['Training', 'Testing'], loc='lower right')

    plt.show()


def plot_mlp():
    # load data
    with open('results/2019-04-21_20-59-37_mlp_results.pkl', 'rb') as file:
        data = pickle.load(file)

    # make data readable
    train_accuracies = data['train_accuracies']  # 5000 values
    train_losses = data['train_losses']  # 5000 values
    test_accuracies = data['test_accuracies']  # 10 values
    test_losses = data['test_losses']  # 10 values
    eval_freq = data['eval_freq']  # 1 value = 500

    # print the final test accuracy and loss
    print("\nMLP:")
    print("Best test accuracy:", round(max(test_accuracies) * 100, 1), "%")
    print("Best test loss:", round(min(test_losses), 5))

    print("\n@2500:")
    print("Last train accuracy:", round(train_accuracies[2499] * 100, 1), "%")
    print("Best test accuracy:", round(max(test_accuracies[0:5]) * 100, 1), "%")

    # setting indices for training and testing
    train_indices = list(range(1, 5001))
    test_indices = list(range(500, 5001, eval_freq))

    # plotting
    fig, axs = plt.subplots(2, 1, sharex=True)

    # accuracy plot
    axs[0].plot(train_indices, train_accuracies, color='b', linewidth=0.2)
    axs[0].plot(test_indices, test_accuracies, color='r', linewidth=2.)
    axs[0].set_ylabel('Accuracy')
    axs[0].set_ylim(0, 1)
    axs[0].grid(True, linestyle=':')

    # loss plot
    axs[1].plot(train_indices, train_losses, color='b', linewidth=0.2)
    axs[1].plot(test_indices, test_losses, color='r', linewidth=2.)
    axs[1].set_xlabel('Steps')
    axs[1].set_ylabel('Loss')
    axs[1].grid(True, linestyle=':')

    fig.suptitle('Accuracy and Loss for the MLP Model', fontsize=14)
    axs[0].legend(['Training', 'Testing'], loc='lower right')

    plt.show()


if __name__ == '__main__':
    # plot_convnet()
    plot_mlp()

