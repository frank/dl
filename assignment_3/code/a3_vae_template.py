import os
import argparse
import numpy as np
from datetime import datetime
from scipy.stats import norm

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets.bmnist import bmnist

# Automatically setting correct device
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type(torch.FloatTensor)


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.input_dim = 28 * 28
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.hidden_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU()
        )

        self.mu_phi = nn.Linear(self.hidden_dim, self.z_dim)

        self.sigma_phi = nn.Sequential(
            nn.Linear(self.hidden_dim, self.z_dim),
            nn.ReLU()
        )

    def forward(self, input):
        h = self.hidden_layer(input)
        mean = self.mu_phi(h)
        std = self.sigma_phi(h)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.output_dim = 28 * 28

        self.model = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        mean = self.model(input)

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.encoder = Encoder(self.hidden_dim, self.z_dim)
        self.decoder = Decoder(self.hidden_dim, self.z_dim)

    def forward(self, input):
        # preprocess batch
        input = torch.flatten(input, 1, 3)
        batch_size = input.size(0)

        # small e for numerical stability
        e = 1e-6

        # get mu and sigma for sampling the latent variable
        p_mu, p_sigma = self.encoder(input)

        # sample using the reparametrization trick
        epsilon = torch.randn(self.z_dim)
        z = (epsilon * p_sigma) + p_mu

        # generate sample
        output = self.decoder(z)

        # compute loss
        l_recon = - torch.sum(input * torch.log(output + e) + (1 - input) * torch.log(1 - output))
        l_reg = 0.5 * torch.sum(p_sigma**2 + p_mu**2 - 1 - torch.log(p_sigma**2 + e))
        average_negative_elbo = torch.sum(l_recon + l_reg) / batch_size

        return average_negative_elbo

    def sample(self, n_samples, z=None):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        if z is None:
            z = torch.randn(n_samples, self.z_dim)

        im_means = self.decoder(z).reshape(n_samples, 1, 28, 28)

        sampled_ims = torch.bernoulli(im_means)

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    epoch_elbo = np.array([])

    for batch in data:
        batch_elbo = model(batch.to(device))

        if model.training:
            model.zero_grad()
            batch_elbo.backward()
            optimizer.step()

        epoch_elbo = np.append(epoch_elbo, batch_elbo.item())

    return np.mean(epoch_elbo, axis=0)


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    # ID for the current run
    run_id = datetime.now().strftime("vae_%Y-%m-%d_%H-%M-%S")
    dir_path = 'vae/' + run_id + '/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim).to(device)  # send to cuda if available
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos

        model.train()
        train_curve.append(train_elbo)

        model.eval()
        val_curve.append(val_elbo)

        print(f"[Epoch {epoch+1:02d}] train elbo: {train_elbo:.2f} val_elbo: {val_elbo:.2f}")

        # plot samples every 5 epochs
        if ((epoch + 1) % 5 == 0) or epoch == 0:
            sampled_ims, im_means = model.sample(25)
            sample_plot = make_grid(im_means, nrow=5).cpu().detach().numpy().transpose(1, 2, 0)
            path = f'samples_epoch_{epoch:02d}.png'
            plt.imsave(dir_path + path, sample_plot)

    # plot manifold at the end of the training
    if ARGS.zdim == 2:
        e = 1e-3
        linspace = np.linspace(0+e, 1-e, 20)
        X = norm.ppf(linspace)
        Y = norm.ppf(linspace)
        z = torch.tensor([[x, y] for x in X for y in Y])
        sampled_ims, im_means = model.sample(20*20, z)
        sample_plot = make_grid(im_means, nrow=20).cpu().detach().numpy().transpose(1, 2, 0)
        path = f'manifold.png'
        plt.imsave(dir_path + path, sample_plot)

    save_elbo_plot(train_curve, val_curve, dir_path + 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
