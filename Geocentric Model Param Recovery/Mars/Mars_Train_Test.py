# -*- coding: utf-8 -*-

# %%

import pandas as pd
import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time


def load_dataset(rel_path, file_name):
    """
    Import the dataset and split input and output data
    Return data in the form of numpy arrays
    """
    os.chdir(rel_path)
    df = pd.read_pickle(file_name)
    # Convert degrees to radians
    df[['Dec']] = df[['Dec']] / 360.00 * 2.00 * np.pi
    # Convert hours to radians
    df[['RA']] = df[['RA']] / 24.00 * 2.00 * np.pi
    df.sort_values(by='JD', ascending=True)
    x = df.JD.to_numpy()
    y = df[['RA', 'Dec']].to_numpy()
    return x, y


def sample_data(x, y, i_start, i_end, step):
    z, w = x[i_start:i_end], y[i_start:i_end]
    z = z[::step]
    w = w[::step]
    return z, w


class ParameterReoveryModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.Omega = nn.Parameter(torch.tensor([0.017],
                                               requires_grad=True,
                                               dtype=torch.float64))
        self.omega = nn.Parameter(torch.tensor([0.0091],
                                               requires_grad=True,
                                               dtype=torch.float64))
        self.k = nn.Parameter(torch.tensor([2.00/50.00],
                                           requires_grad=True,
                                           dtype=torch.float64))
        self.rr = nn.Parameter(torch.tensor([20./50.],
                                            requires_grad=True,
                                            dtype=torch.float64))
        self.N = nn.Parameter(torch.tensor([0.80],
                                           requires_grad=True,
                                           dtype=torch.float64))
        self.i = nn.Parameter(torch.tensor([0.50],
                                           requires_grad=True,
                                           dtype=torch.float64))
        self.N2 = nn.Parameter(torch.tensor([0.00],
                                            requires_grad=True,
                                            dtype=torch.float64))
        self.i2 = nn.Parameter(torch.tensor([0.00],
                                            requires_grad=True,
                                            dtype=torch.float64))
        self.phi = nn.Parameter(torch.tensor([0.00],
                                             requires_grad=True,
                                             dtype=torch.float64))
        self.phi2 = nn.Parameter(torch.tensor([0.00],
                                              requires_grad=True,
                                              dtype=torch.float64))

    def normalize_angle(self, ang):
        return ang - torch.floor((ang/(2.00 * np.pi))) * 2.00 * np.pi

    def forward(self, t):
        # Computes the outputs / predictions
        t = torch.reshape(t, (len(t), 1))
        sqrt_term = torch.sqrt(1-self.k**2 *
                               torch.sin(self.Omega*t + self.phi)**2)
        sin_alpha = torch.sin(self.Omega * t + self.phi) * \
            (sqrt_term - self.k*torch.cos(self.Omega*t + self.phi)**2)
        theta = torch.atan2((torch.tan(self.Omega * t + self.phi)),
                            (1-(2*self.k /
                                (self.k*torch.cos(self.Omega*t+self.phi)
                                 - sqrt_term))))
        rho = sin_alpha/torch.sin(theta)
        # Rectangular coordinates of the body on
        # the epicycle relative to its center
        xcyc = self.rr * torch.cos(self.omega*t + self.phi2)
        ycyc = self.rr * torch.sin(self.omega*t + self.phi2)
        zcyc = 0.00
        # Rotation of epicycle
        # Rotation 1: around y z plane
        xcyc1 = xcyc
        ycyc1 = torch.cos(self.i+self.i2) * ycyc + \
            torch.sin(self.i+self.i2) * zcyc
        zcyc1 = -torch.sin(self.i+self.i2) * ycyc + \
            torch.cos(self.i+self.i2) * zcyc
        # Rotation 2: around the new x y plane
        xcyc2 = torch.cos(self.N+self.N2) * xcyc1 + \
            torch.sin(self.N+self.N2) * ycyc1
        ycyc2 = -torch.sin(self.N+self.N2) * xcyc1 + \
            torch.cos(self.N+self.N2) * ycyc1
        zcyc2 = zcyc1
        x = rho * torch.cos(theta)
        y = rho * torch.sin(theta)
        z = 0.00
        xr1 = x
        yr1 = torch.cos(self.i) * y + torch.sin(self.i) * z
        zr1 = -torch.sin(self.i) * y + torch.cos(self.i) * z
        xr2 = torch.cos(self.N) * xr1 + torch.sin(self.N) * yr1
        yr2 = -torch.sin(self.N) * xr1 + torch.cos(self.N) * yr1
        zr2 = zr1
        xtot = xr2 + xcyc2
        ytot = yr2 + ycyc2
        ztot = zr2 + zcyc2
        RA = self.normalize_angle((torch.atan2(ytot, xtot)))
        Dec = self.normalize_angle((torch.atan2(ztot,
                                                torch.sqrt(xtot**2+ytot**2))))
        return torch.hstack((RA, Dec))


def train_model(grad_weights, l_rate, n_epochs):
    print('', end='')
    optimizer = optim.Adam(model.parameters(), lr=l_rate)
    losses = []
    t0 = time()
    for epoch in range(n_epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            model.train()  # Set the model to training mode
            yhat = model(x_batch)
            er1 = torch.abs(yhat-y_batch)
            er2 = (2*np.pi) - er1
            error = torch.min(er1, er2)
            loss = (error**2).mean()
            losses.append(loss.item())
            optimizer.zero_grad()
            model.N2.retain_grad()
            loss.backward()
            # We use weights to reduce certain gradients
            param_names = list(grad_weights.keys())
            for param_name in param_names:  # For each locked param
                w = grad_weights[param_name]
                attr = getattr(model, param_name)  # Get pointer
                grad_attr = getattr(attr, 'grad')  # same for gradient
                setattr(attr, 'grad', grad_attr * w)  # set to zero
            # Take a step using the modified gradients
            optimizer.step()
            if (time() - t0) > 1:
                print('\r'+"{:e}".format(loss.item()), end='')
                t0 = time()
    print()
    print('Training complete.')
    return losses


def print_parameter_values(file=False):
    x = []
    for name, param in model.named_parameters():
        x.append([name, "{:e}".format(param.item())])
    print(pd.DataFrame(x))
    if file:
        pd.DataFrame(x).to_csv('mars_parameters.csv')


def plot_prediction():
    yhat = model(torch.from_numpy(x_train).to(device))
    y_pred = yhat.to('cpu').detach().numpy()
    y_train_normalized = y_train
    y_train_normalized[y_train_normalized < 0] = \
        y_train_normalized[y_train_normalized < 0] + 2 * np.pi
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].set_title('Right Ascension')
    ax[0].set_xlabel('JD')
    ax[0].set_ylabel('RA (rad)')
    ax[0].plot(x_train, y_train[:, 0], label='Training Data')
    ax[0].plot(x_train, y_pred[:, 0], label='Prediction')
    ax[0].legend()
    ax[1].set_title('Declination')
    ax[1].set_xlabel('JD')
    ax[1].set_ylabel('Dec (rad)')
    y_train_norm = y_train[:, 1]
    y_train_norm[y_train_norm >= 3.00] = \
        y_train_norm[y_train_norm >= 3.00] - 2.00*np.pi
    y_pred_norm = y_pred[:, 1]
    y_pred_norm[y_pred_norm >= 3.00] = \
        y_pred_norm[y_pred_norm >= 3.00] - 2.00*np.pi
    ax[1].plot(x_train, y_train_norm, label='Training Data')
    ax[1].plot(x_train, y_pred_norm, label='Prediction')
    ax[1].legend()
    plt.show()


def plot_residual(x_tens, y_tens, plt_density):
    n = len(x_tens)
    s = int(n/plt_density)
    yhat = model(x_tens[::s])
    er1 = torch.abs(yhat-y_tens[::s])
    er2 = (2*np.pi) - er1
    error = torch.min(er1, er2)
    e = error.to('cpu').detach().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].set_title('Right Ascension Residuals')
    ax[0].set_xlabel('JD')
    ax[0].set_ylabel('RA (rad)')
    ax[0].scatter(x_train[::s], e[:, 0], alpha=0.2)
    ax[1].set_title('Declination Residuals')
    ax[1].set_xlabel('JD')
    ax[1].set_ylabel('Dec (rad)')
    ax[1].scatter(x_train[::s], e[:, 1], alpha=0.2)
    plt.show()


def brute(name, current_value, rng, current_loss, figure=True):
    vals = rng
    losses = np.empty((len(vals)))
    for i in tqdm(range(len(vals))):
        val = vals[i]
        with torch.no_grad():
            setattr(model, name, nn.Parameter(torch.tensor([val],
                    requires_grad=True, dtype=torch.float64).to(device)))
        yhat = model(x_train_tensor)
        er1 = torch.abs(yhat-y_train_tensor)
        er2 = (2*np.pi) - er1
        error = torch.min(er1, er2)
        loss = (error**2).mean()
        loss2 = loss.to('cpu')
        losses[i] = loss2.item()
    losses = np.array(losses)
    minimizer = vals[np.argmin(losses)]
    with torch.no_grad():
        setattr(model, name, nn.Parameter(torch.tensor([minimizer],
                requires_grad=True, dtype=torch.float64).to(device)))
    if figure:
        fig = plt.figure()
        plt.plot(vals, losses, linewidth=0.5)
        plt.scatter(current_value, current_loss)
        plt.scatter(minimizer, np.min(losses))
        fig.suptitle('Brute force minimization w.r.t. '+name)
        plt.xlabel(name+' variable. '+' Prev. Value: ' +
                   str("{:e}".format(current_value)) +
                   '\n' + 'Minimizer: ' + str("{:e}".format(minimizer)))
        plt.ylabel('Loss (MSE)')
        plt.show()
    return minimizer, np.min(losses)


def loss_plot(losses_list, npoints, log=False):
    n = len(losses_list)
    sampl_rate = int(n/npoints)
    plt.figure()
    plt.plot(losses[::sampl_rate])
    plt.xlabel('Step')
    plt.ylabel('Loss (MSE)')
    if log:
        plt.yscale('log')
    plt.show()


def test_error(x_tens, y_tens):
    yhat = model(x_tens)
    er1 = torch.abs(yhat-y_tens)
    er2 = (2*np.pi) - er1
    error = torch.min(er1, er2)
    e = error.to('cpu').detach().numpy()
    print(np.mean(e**2))


def plot_prediction_test():
    yhat_train = model(torch.from_numpy(x_train).to(device))
    y_pred_train = yhat_train.to('cpu').detach().numpy()
    y_train_normalized = y_train
    y_train_normalized[y_train_normalized < 0] = \
        y_train_normalized[y_train_normalized < 0] + 2 * np.pi
    y_test_normalized = y_test
    y_test_normalized[y_test_normalized < 0] = \
        y_test_normalized[y_test_normalized < 0] + 2 * np.pi

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].set_title('Right Ascension')
    ax[0].set_xlabel('JD')
    ax[0].set_ylabel('RA (rad)')
    ax[0].scatter(x_train, y_train[:, 0], s=0.80, label='Training Data')
    ax[0].scatter(x_test, y_test[:, 0], s=0.80, label='Test Data')
    ax[0].plot(x_train, y_pred_train[:, 0], linestyle='dashed', color='black',
               label='Prediction')
    ax[0].legend()

    ax[1].set_title('Declination')
    ax[1].set_xlabel('JD')
    ax[1].set_ylabel('Dec (rad)')
    y_train_norm = y_train[:, 1]
    y_train_norm[y_train_norm >= 3.00] = \
        y_train_norm[y_train_norm >= 3.00] - 2.00*np.pi
    y_test_norm = y_test[:, 1]
    y_test_norm[y_test_norm >= 3.00] = \
        y_test_norm[y_test_norm >= 3.00] - 2.00*np.pi
    y_pred_norm = y_pred_train[:, 1]
    y_pred_norm[y_pred_norm >= 3.00] = \
        y_pred_norm[y_pred_norm >= 3.00] - 2.00*np.pi
    ax[1].scatter(x_train, y_train_norm, s=0.80, label='Training Data')
    ax[1].scatter(x_test, y_test_norm, s=0.80, label='Test Data')
    ax[1].plot(x_train, y_pred_norm, linestyle='dashed', color='black',
               label='Prediction')
    ax[1].legend()
    plt.show()


if __name__ == '__main__':
    # Set the seed for reproducibility
    # torch.manual_seed(42)  # No random numbers so far
    t_start = time()
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Instantiate untrained model
    model = ParameterReoveryModel().to(device)

    # Load the dataset
    x_test, y_test = load_dataset('../Geo_Data', 'Mars_test.df')
    x_train, y_train = load_dataset('../Geo_Data', 'Mars_training.df')

    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Phase 1 - Coarse parameter recovery')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # Convert training data to tensors
    x_train_tensor = torch.from_numpy(x_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).to(device)
    # Create a dataset and a dataloader
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=10000, shuffle=False)

    # Specify gradient weights
    grad_weights = {
        'Omega': 0.00,
        'omega': 0.00,
        'N2': 0.00,
        'i2': 0.00
    }
    # Train the model
    losses = train_model(grad_weights, 1.0E-2, 300)
    # Plot the losses
    loss_plot(losses, 10)

    # Plot the predictions
    plot_prediction()

    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Phase 2 - Frequency optimization   ')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    rng = np.linspace(model.Omega.item()*0.95, model.Omega.item()*1.05, 1000)
    Ominimizer, new_loss = brute('Omega', model.Omega.item(), rng, losses[-1])

    rng = np.linspace(model.omega.item()*0.9, model.omega.item()*1.1, 2000)
    ominimizer, new_loss = brute('omega', model.omega.item(), rng, new_loss)

    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Phase 3 - Training                 ')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    # Specify gradient weights
    grad_weights = {
        'Omega': 1.00E-18,
        'omega': 1.00E-18,
        'N2': 0.00,
        'i2': 0.00
    }
    # Train the model
    losses = train_model(grad_weights, 1.0E-2, 300)
    # Plot the losses
    loss_plot(losses, 10)

    # Plot the predictions
    plot_prediction()

    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Phase 4 - Frequency optimization   ')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    rng = np.linspace(model.Omega.item()*0.95, model.Omega.item()*1.05, 1000)
    Ominimizer, new_loss = brute('Omega', model.Omega.item(), rng, losses[-1])

    rng = np.linspace(model.omega.item()*0.95, model.omega.item()*1.05, 2000)
    ominimizer, new_loss = brute('omega', model.omega.item(), rng, new_loss)

    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Phase 5 - Training                 ')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # Specify gradient weights
    grad_weights = {
        'Omega': 1.00E-18,
        'omega': 1.00E-18,
    }
    # Train the model
    losses = train_model(grad_weights, 1.0E-2, 400)
    # Plot the losses
    loss_plot(losses, 10)

    # Plot the predictions
    plot_prediction()

    # Print learned parameter values
    print_parameter_values()

    t_end = time()

    print()
    print('Total elapsed time:', t_end - t_start)

    print()
    print('~~~~~~~~~~~~~~~~~~~~~')
    print('Residuals w.r.t. time')
    print('~~~~~~~~~~~~~~~~~~~~~')

    plot_residual(x_train_tensor, y_train_tensor, 500)

    plot_prediction_test()

    print()
    print('~~~~~~~~~~~~~~~~~~~~~')
    print('Error on the test set')
    print('~~~~~~~~~~~~~~~~~~~~~')
    # Convert test data to tensors
    x_test_tensor = torch.from_numpy(x_test).to(device)
    y_test_tensor = torch.from_numpy(y_test).to(device)
    test_error(x_test_tensor, y_test_tensor)
