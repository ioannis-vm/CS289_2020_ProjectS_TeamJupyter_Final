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
    df = pd.read_csv(file_name)
    # Convert degrees to radians
    df[['Dec']] = df[['Dec']] / 360.00 * 2.00 * np.pi
    # Convert hours to radians
    df[['RA']] = df[['RA']] / 24.00 * 2.00 * np.pi
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

        self.M0 = nn.Parameter(torch.tensor([0.00],
                                            requires_grad=True,
                                            dtype=torch.float64))
        self.dm = nn.Parameter(torch.tensor([0.03],
                                            requires_grad=True,
                                            dtype=torch.float64))
        self.e0 = nn.Parameter(torch.tensor([0.00],
                                            requires_grad=True,
                                            dtype=torch.float64))

        self.w0 = nn.Parameter(torch.tensor([0.00],
                                            requires_grad=True,
                                            dtype=torch.float64))
        self.ecl0 = nn.Parameter(torch.tensor([0.00],
                                              requires_grad=True,
                                              dtype=torch.float64))
        self.a0 = nn.Parameter(torch.tensor([1.00],
                                            requires_grad=True,
                                            dtype=torch.float64))
        # secondary terms
        self.de = nn.Parameter(torch.tensor([0.00],
                                            requires_grad=True,
                                            dtype=torch.float64))
        self.da = nn.Parameter(torch.tensor([0.00],
                                            requires_grad=True,
                                            dtype=torch.float64))
        self.dw = nn.Parameter(torch.tensor([0.00],
                                            requires_grad=True,
                                            dtype=torch.float64))
        self.decl = nn.Parameter(torch.tensor([0.00],
                                              requires_grad=True,
                                              dtype=torch.float64))

    def normalize_angle(self, ang):
        return ang - torch.floor((ang/(2.00 * np.pi))) * 2.00 * np.pi

    def forward(self, x):
        # Shift time
        # x = x - 2451543.5
        # Computes the outputs / predictions
        e0nneg = torch.exp(-self.e0-3.00)
        x = torch.reshape(x, (len(x), 1))
        M = self.normalize_angle(self.M0 + self.dm * x)
        e = e0nneg + self.de * x
        a = self.a0 + self.da * x
        w = self.normalize_angle(self.w0 + self.dw * x)
        E = M + e * torch.sin(M) * (1.00 + e * torch.cos(M))
        E1 = E - (E - e * torch.sin(E) - M)/(1 - e * torch.cos(E))
        E2 = E1 - (E1 - e * torch.sin(E1) - M)/(1 - e * torch.cos(E1))
        E3 = E2 - (E2 - e * torch.sin(E2) - M)/(1 - e * torch.cos(E2))
        xv = a * (torch.cos(E3)-e)
        yv = a * (torch.sqrt(1-e**2)*torch.sin(E3))
        v = a * torch.atan2(yv, xv)
        r = torch.sqrt(xv**2+yv**2)
        lonsun = v + w
        xs = r * torch.cos(lonsun)
        ys = r * torch.sin(lonsun)
        xe = xs
        ye = ys * torch.cos(self.ecl0 + self.decl * x)
        ze = ys * torch.sin(self.ecl0 + self.decl * x)
        RA = self.normalize_angle(torch.atan2(ye, xe))
        Dec = self.normalize_angle(torch.atan2(ze, torch.sqrt(xe**2+ye**2)))
        return torch.hstack((RA, Dec))


def normalize_angle(ang):
    return ang - np.floor((ang/(2.00 * np.pi))) * 2.00 * np.pi


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
        pd.DataFrame(x).to_csv('earth')
    x = []
    print('Converted Parameter Values')
    x.append(['M', (normalize_angle(model.M0.item()
                    + 2451544.5*model.dm.item()))/np.pi*180.])
    x.append(['w', (normalize_angle(model.w0.item()
                    + 2451544.5*model.dw.item()))/np.pi*180.])
    x.append(['e', (np.exp(-model.e0.item()-3.00)+2451544.5*model.de.item())])
    x.append(['a', (model.a0.item()+2451544.5*model.da.item())])
    x.append(['ecl', (model.ecl0.item()
                      + 2451544.5*model.decl.item())/np.pi*180])
    print(pd.DataFrame(x))


def plot_prediction(i_start, i_end, plt_density):
    x_train, y_train = sample_data(x_all, y_all, i_start, i_end, 1)
    step = int(np.ceil(len(x_train)/plt_density))
    x_train, y_train = x_train[::step], y_train[::step]
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
    fig, ax = plt.subplots(1, 2, figsize=(12, 1.5))
    ax[0].set_title('Right Ascension Residuals')
    ax[0].set_xlabel('JD')
    ax[0].set_ylabel('RA (rad)')
    ax[0].scatter(x_train[::s], e[:, 0], alpha=0.2)
    # plt.show()
    # plt.figure
    ax[1].set_title('Declination Residuals')
    ax[1].set_xlabel('JD')
    ax[1].set_ylabel('Dec (rad)')
    ax[1].scatter(x_train[::s], e[:, 1], alpha=0.2)
    plt.show()


def brute(dmrange, current_loss, figure=True):
    """
    Use brute force to find the optimal value for dm,
    which is related to the orbital period of the Earth
    """
    current_value = model.dm.item()
    vals = dmrange
    losses = np.empty((len(vals)))
    for i in tqdm(range(len(vals))):
        val = vals[i]
        with torch.no_grad():
            setattr(model, 'dm', nn.Parameter(torch.tensor([val],
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
        setattr(model, 'dm', nn.Parameter(torch.tensor([minimizer],
                requires_grad=True, dtype=torch.float64).to(device)))
    if figure:
        fig = plt.figure()
        plt.plot(vals, losses, linewidth=0.5)
        plt.scatter(current_value, current_loss)
        plt.scatter(minimizer, np.min(losses))
        fig.suptitle('Brute force minimization w.r.t. dm')
        plt.xlabel('dm variable. '+' Prev. Value: ' +
                   str("{:e}".format(current_value)) +
                   '\n' + 'Minimizer: ' + str("{:e}".format(minimizer)))
        plt.ylabel('Loss (MSE)')
        plt.show()
    return minimizer, loss.item()


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

# %%


if __name__ == '__main__':
    # Set the seed for reproducibility
    # torch.manual_seed(42)  # No random numbers so far
    t_start = time()
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Instantiate untrained model
    model = ParameterReoveryModel().to(device)

    # Load the dataset
    x_all, y_all = load_dataset('../Heliocentric_Data', 'Sun.csv')
    n = len(x_all)

    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Phase 1 - Coarse parameter recovery')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # Resample - 1 year of data
    x_train, y_train = sample_data(x_all, y_all, 0+int(n/2), 365+int(n/2), 1)
    # Convert training data to tensors
    x_train_tensor = torch.from_numpy(x_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).to(device)
    # Create a dataset and a dataloader
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=1000, shuffle=False)
    # Specify gradient weights
    grad_weights = {
        'dm': 0.00,
        'de': 0.00,
        'a0': 0.00,
        'dw': 0.00,
        'da': 0.00,
        'decl': 0.00
    }
    # Train the model
    losses = train_model(grad_weights, 1.0E-1, 100)
    # Save the model
    PATH = './sun_LR.pth'
    torch.save(model.state_dict(), PATH)

    # Print learned parameter values
    print_parameter_values()
    # Plot the predictions
    plot_prediction(0, 365, 200)
    # Plot the losses
    loss_plot(losses, 20)

    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Phase 2 - Orbital Period Estimation')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Using brute-force to pick a global minimizer for dm...')
    # Load the model
    model = ParameterReoveryModel().to(device)
    model.load_state_dict(torch.load(PATH))
    # Include 10 years of data
    x_train, y_train = sample_data(x_all, y_all, 0+int(n/2),
                                   365*10+int(n/2), 10)
    # Convert training data to tensors
    x_train_tensor = torch.from_numpy(x_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).to(device)
    # Define grid-search points
    dmrange = np.linspace(0.5*2*np.pi/365., 2.00*2*np.pi/365., 2000)
    minimizer, new_loss = brute(dmrange, losses[-1])
    # refine, repeat
    dmrange = np.linspace(minimizer*0.95, minimizer*1.05, 500)
    minimizer, new_loss = brute(dmrange, losses[-1])
    # Save the model
    torch.save(model.state_dict(), PATH)

    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Phase 3 - Train on 10 years of data        ')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # Load the model
    model = ParameterReoveryModel().to(device)
    model.load_state_dict(torch.load(PATH))
    # Include 10 years of data
    x_train, y_train = sample_data(x_all, y_all, 0+int(n/2),
                                   365*10+int(n/2), 10)
    # Create a dataset and a dataloader
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=10000, shuffle=False)
    # Train the model
    losses = train_model(grad_weights, 1e-2, 100)

    # Print learned parameter values
    print_parameter_values()
    # Plot the losses
    loss_plot(losses, 20)

    # Save the model
    torch.save(model.state_dict(), PATH)

    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Phase 4 - Orbital Period Estimation v2')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Using brute-force to pick a global minimizer for dm...')
    # Load the model
    model = ParameterReoveryModel().to(device)
    model.load_state_dict(torch.load(PATH))
    # Include all data
    x_train, y_train = sample_data(x_all, y_all, 0, len(x_all), 500)
    # Convert training data to tensors
    x_train_tensor = torch.from_numpy(x_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).to(device)
    # Define grid-search points
    dmrange = np.linspace(minimizer*0.995, minimizer*1.005, 1000)
    minimizer, new_loss = brute(dmrange, losses[-1])
    # refine, repeat
    dmrange = np.linspace(minimizer*0.9995, minimizer*1.0005, 1000)
    minimizer, new_loss = brute(dmrange, losses[-1])
    # Save the model
    torch.save(model.state_dict(), PATH)

    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Phase 4 - Train on 200 years of data ')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # Resample - 200 year of data
    x_train, y_train = sample_data(x_all, y_all, 0+int(n/2),
                                   365*200+int(n/2), 100)
    # Convert training data to tensors
    x_train_tensor = torch.from_numpy(x_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).to(device)
    # Create a dataset and a dataloader
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=10000, shuffle=True)
    # Load the model
    torch.cuda.empty_cache()
    model = ParameterReoveryModel().to(device)
    model.load_state_dict(torch.load(PATH))
    # Train the model
    losses = train_model(grad_weights, 1.0E-3, 200)

    # Print learned parameter values
    print_parameter_values()
    # Plot the losses
    loss_plot(losses, 20)

    # Save the model
    torch.save(model.state_dict(), PATH)

    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Phase 5 - Train on the entire dataset')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # Resample - all
    x_train, y_train = sample_data(x_all, y_all, 0, len(x_all), 10)
    # Convert training data to tensors
    x_train_tensor = torch.from_numpy(x_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).to(device)
    # Create a dataset and a dataloader
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=100, shuffle=True)
    # Load the model
    torch.cuda.empty_cache()
    model = ParameterReoveryModel().to(device)
    model.load_state_dict(torch.load(PATH))
    # Specify gradient weights - all can change now
    grad_weights = {
        'dm': 1E-18,
        'de': 1E-18,
        'a0': 1E-18,
        'dw': 1E-18,
        'da': 1E-19,
        'decl': 1E-18
    }
    # Train the model
    losses = train_model(grad_weights, 1.0E-30, 400)

    ####################
    # Refine as needed #
    ####################
    # losses = train_model(grad_weights, 1.0E-4, 999999999)

    # Print learned parameter values
    print_parameter_values()
    # Plot the losses
    loss_plot(losses, 20, True)

    # Save the model
    torch.save(model.state_dict(), PATH)

    t_end = time()

    print()
    print('Total elapsed time:', t_end - t_start)

    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Check - Verify global minimum w.r.t. dw')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Verify dm is global argmin')
    # Load the model
    model = ParameterReoveryModel().to(device)
    model.load_state_dict(torch.load(PATH))
    # Include all data
    x_train, y_train = sample_data(x_all, y_all, 0, len(x_all), 500)
    # Convert training data to tensors
    x_train_tensor = torch.from_numpy(x_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).to(device)
    # Define grid-search points
    dmrange = np.linspace(minimizer*0.9995, minimizer*1.0005, 1000)
    minimizer, new_loss = brute(dmrange, losses[-1])
    # Load the model (restore)
    model = ParameterReoveryModel().to(device)
    model.load_state_dict(torch.load(PATH))

    print()
    print('~~~~~~~~~~~~~~~~~~~~~')
    print('Residuals w.r.t. time')
    print('~~~~~~~~~~~~~~~~~~~~~')
    # Load the model (restore)
    model = ParameterReoveryModel().to(device)
    model.load_state_dict(torch.load(PATH))
    plot_residual(x_train_tensor, y_train_tensor, 2000)

    #######################################
    # Output learned parameters to a file #
    #######################################
    # Load the model
    model = ParameterReoveryModel().to(device)
    model.load_state_dict(torch.load(PATH))
    # Output parameters
    print_parameter_values(file=True)
