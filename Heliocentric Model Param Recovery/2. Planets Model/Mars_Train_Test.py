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
    x = df.JD.to_numpy()
    y = df[['RA', 'Dec']].to_numpy()
    return x, y


def load_earth_params(rel_path, file_name):
    rel_path = '../Heliocentric_Data/'
    file_name = 'earth'
    df = pd.read_csv(rel_path+file_name, index_col=0)
    df = df.set_index('0')
    dct = df.T.to_dict('list')
    return dct


def sample_data(x, y, i_start, i_end, step):
    z, w = x[i_start:i_end], y[i_start:i_end]
    z = z[::step]
    w = w[::step]
    return z, w


class ParameterReoveryModel(nn.Module):
    def __init__(self, param_dict):
        super().__init__()
        # Results from the model of the Earth's orbit (fixed parameters)
        self.earth_M0 = torch.tensor(param_dict['M0'][0], dtype=torch.float64)
        self.earth_dm = torch.tensor(param_dict['dm'][0], dtype=torch.float64)
        self.earth_e0 = torch.tensor(param_dict['e0'][0], dtype=torch.float64)
        self.earth_w0 = torch.tensor(param_dict['w0'][0], dtype=torch.float64)
        self.earth_ecl0 = torch.tensor(param_dict['ecl0'][0],
                                       dtype=torch.float64)
        self.earth_a0 = torch.tensor(param_dict['a0'][0], dtype=torch.float64)
        self.earth_de = torch.tensor(param_dict['de'][0], dtype=torch.float64)
        self.earth_da = torch.tensor(param_dict['da'][0], dtype=torch.float64)
        self.earth_dw = torch.tensor(param_dict['dw'][0], dtype=torch.float64)
        self.earth_decl = torch.tensor(param_dict['decl'][0],
                                       dtype=torch.float64)
        # Parameters to learn
        self.M0 = nn.Parameter(torch.tensor([0.00],
                                            requires_grad=True,
                                            dtype=torch.float64))
        self.dm = nn.Parameter(torch.tensor([1.00e-2],
                                            requires_grad=True,
                                            dtype=torch.float64))
        self.e0 = nn.Parameter(torch.tensor([0.00],
                                            requires_grad=True,
                                            dtype=torch.float64))
        self.w0 = nn.Parameter(torch.tensor([0.00],
                                            requires_grad=True,
                                            dtype=torch.float64))
        self.a0 = nn.Parameter(torch.tensor([1.00],
                                            requires_grad=True,
                                            dtype=torch.float64))
        self.N = nn.Parameter(torch.tensor([0.00],
                                           requires_grad=True,
                                           dtype=torch.float64))
        self.i = nn.Parameter(torch.tensor([0.00],
                                           requires_grad=True,
                                           dtype=torch.float64))
        self.de = nn.Parameter(torch.tensor([0.00],
                                            requires_grad=True,
                                            dtype=torch.float64))
        self.dw = nn.Parameter(torch.tensor([0.00],
                                            requires_grad=True,
                                            dtype=torch.float64))
        self.dN = nn.Parameter(torch.tensor([0.00],
                                            requires_grad=True,
                                            dtype=torch.float64))
        self.di = nn.Parameter(torch.tensor([0.00],
                                            requires_grad=True,
                                            dtype=torch.float64))

    def normalize_angle(self, ang):
        return ang - torch.floor((ang/(2.00 * np.pi))) * 2.00 * np.pi

    def forward(self, x):
        # Computes the outputs / predictions
        x = torch.reshape(x, (len(x), 1))
        # Determine the relative Earth - Sun position
        # (All of the following lines involve quantities
        # that we won't be updating)
        earth_M = self.normalize_angle(self.earth_M0 + self.earth_dm * x)
        earth_e0nneg = torch.exp(-self.earth_e0-3.00)
        earth_e = earth_e0nneg + self.earth_de * x
        earth_a = self.earth_a0 + self.earth_da * x
        earth_w = self.normalize_angle(self.earth_w0 + self.earth_dw * x)
        earth_ecl = self.earth_ecl0 + self.earth_decl * x
        earth_E = earth_M + earth_e * torch.sin(earth_M) \
            * (1.00 + earth_e * torch.cos(earth_M))
        earth_E1 = earth_E - (earth_E - earth_e * torch.sin(earth_E)
                              - earth_M)/(1 - earth_e * torch.cos(earth_E))
        earth_E2 = earth_E1 - (earth_E1 - earth_e * torch.sin(earth_E1)
                               - earth_M)/(1 - earth_e * torch.cos(earth_E1))
        earth_E3 = earth_E2 - (earth_E2 - earth_e * torch.sin(earth_E2)
                               - earth_M)/(1 - earth_e * torch.cos(earth_E2))
        earth_xv = earth_a * (torch.cos(earth_E3)-earth_e)
        earth_yv = earth_a * (torch.sqrt(1-earth_e**2)*torch.sin(earth_E3))
        earth_v = earth_a * torch.atan2(earth_yv, earth_xv)
        earth_r = torch.sqrt(earth_xv**2+earth_yv**2)
        lonsun = earth_v + earth_w
        xs = earth_r * torch.cos(lonsun)
        ys = earth_r * torch.sin(lonsun)
        # Determine the quantities for the celestial body of interest
        N = self.N + self.dN * x
        i = self.i + self.di * x
        M = self.normalize_angle(self.M0 + self.dm * x)
        e0nneg = torch.exp(-self.e0-3.00)
        e = e0nneg + self.de * x
        a = self.a0
        w = self.normalize_angle(self.w0 + self.dw * x)
        E = M + e * torch.sin(M) \
            * (1.00 + e * torch.cos(M))
        E1 = E - (E - e * torch.sin(E) - M)/(1 - e * torch.cos(E))
        E2 = E1 - (E1 - e * torch.sin(E1) - M)/(1 - e * torch.cos(E1))
        E3 = E2 - (E2 - e * torch.sin(E2) - M)/(1 - e * torch.cos(E2))
        xv = a * (torch.cos(E3)-e)
        yv = a * (torch.sqrt(1-e**2)*torch.sin(E3))
        v = torch.atan2(yv, xv)
        r = torch.sqrt(xv**2+yv**2)
        # Determine the position in space
        xh = r * (torch.cos(N) * torch.cos(v + w)
                  - torch.sin(N) * torch.sin(v + w) * torch.cos(i))
        yh = r * (torch.sin(N) * torch.cos(v + w)
                  + torch.cos(N) * torch.sin(v + w) * torch.cos(i))
        zh = r * (torch.sin(v + w) * torch.sin(i))
        lonecl = torch.atan2(yh, xh)
        latecl = torch.atan2(zh, torch.sqrt(xh**2 + yh**2))
        # Convert to geocentric coordinates
        xh = r * torch.cos(lonecl) * torch.cos(latecl)
        yh = r * torch.sin(lonecl) * torch.cos(latecl)
        zh = r * torch.sin(latecl)
        xs = earth_r * torch.cos(lonsun)
        ys = earth_r * torch.sin(lonsun)
        # Convert from heliocentric to geocentric
        xg = xh + xs
        yg = yh + ys
        zg = zh
        # Convert to equatorial coordinates
        xe = xg
        ye = yg * torch.cos(earth_ecl) - zg * torch.sin(earth_ecl)
        ze = yg * torch.sin(earth_ecl) + zg * torch.cos(earth_ecl)
        # Lastly, compute RA and Dec
        RA = self.normalize_angle(torch.atan2(ye, xe))
        Dec = self.normalize_angle(torch.atan2(ze, torch.sqrt(xe**2+ye**2)))
        return torch.hstack((RA, Dec))


def normalize_angle(ang):
    return ang - np.floor((ang/(2.00 * np.pi))) * 2.00 * np.pi


def train_model(grad_weights, l_rate, n_epochs, tol=1E-6):
    print('', end='')
    optimizer = optim.Adam(model.parameters(), lr=l_rate)
    losses = []
    t0 = time()
    for epoch in range(n_epochs):
        for x_batch, y_batch in train_loader:
            prev_params = get_parameter_vector()
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
            new_params = get_parameter_vector()
            # Termination criterion
            c = np.linalg.norm(prev_params-new_params) / \
                (np.linalg.norm(new_params)+tol)
            if c < tol:
                print()
                print('Algorithm Converged')
                break
            if (time() - t0) > 1:
                print('\r'+"{:e}".format(loss.item()), end='')
                t0 = time()
        else:
            continue  # only executed if the inner loop did NOT break
        break         # only executed if the inner loop DID break
    print()
    print('Training complete.')
    return losses


def print_parameter_values(file=False):
    x = []
    for name, param in model.named_parameters():
        x.append([name, "{:e}".format(param.item())])
    print(pd.DataFrame(x))
    if file:
        pd.DataFrame(x).to_csv('Mars_test_train.csv')
    x = []
    print('Converted Parameter Values')
    x.append(['M', (normalize_angle(model.M0.item()
                    + 2451544.5*model.dm.item()))/np.pi*180.])
    x.append(['w', (normalize_angle(model.w0.item()
                    + 2451544.5*model.dw.item()))/np.pi*180.])
    x.append(['i', (normalize_angle(model.i.item()
                    + 2451544.5*model.di.item()))/np.pi*180.])
    x.append(['N', (normalize_angle(model.N.item()
                    + 2451544.5*model.dN.item()))/np.pi*180.])
    x.append(['e', (np.exp(-model.e0.item()-3.00)
                    + 2451544.5*model.de.item())])
    x.append(['a', model.a0.item()])
    print(pd.DataFrame(x))


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
    if n != 0:
        sampl_rate = int(n/npoints)
        if sampl_rate == 0:
            print("loss_plot: No losses to plot")
            return
        plt.figure()
        plt.plot(losses[::sampl_rate])
        plt.xlabel('Step')
        plt.ylabel('Loss (MSE)')
        if log:
            plt.yscale('log')
        plt.show()


def get_parameter_vector():
    x = []
    for name, param in model.named_parameters():
        x.append(param.item())
    return np.array(x)


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
    ax[0].plot(x_train, y_pred_train[:, 0], linestyle='dashed',
               color='black', label='Prediction')
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

    cel_body = 'Mars'
    x_test, y_test = load_dataset('../Heliocentric_Data', 'Mars_test.df')
    x_train, y_train = load_dataset('../Heliocentric_Data', 'Mars_training.df')

    params = load_earth_params('../Heliocentric_Data/', 'earth_parameters.csv')

    t_start = time()
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Instantiate untrained model
    model = ParameterReoveryModel(params).to(device)

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
                              batch_size=10000000, shuffle=False)
    # Specify gradient weights
    grad_weights = {
        'dm': 0.00,
        'de': 0.00,
        'dw': 0.00,
        'dN': 0.00,
        'di': 0.00
    }
    # Train the model
    losses = train_model(grad_weights, 1.0E-2, 1000, tol=1E-3)
    # Save the model
    PATH = './Mars_Train_Test.pth'
    torch.save(model.state_dict(), PATH)

    # Print learned parameter values
    print_parameter_values()
    # Plot the predictions
    plot_prediction()
    # Plot the losses
    loss_plot(losses, 20)

    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Phase 2 - Orbital Period Estimation')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Using brute-force to pick a global minimizer for dm...')
    # Load the model
    model = ParameterReoveryModel(params).to(device)
    model.load_state_dict(torch.load(PATH))
    # Define grid-search points
    dmrange = np.linspace(6.00E-5, 1.00E-1, 10000)
    minimizer, new_loss = brute(dmrange, losses[-1])
    # refine, repeat
    dmrange = np.linspace(minimizer*0.95, minimizer*1.05, 500)
    minimizer, new_loss = brute(dmrange, losses[-1])
    # Save the model
    torch.save(model.state_dict(), PATH)

    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Phase 3 - Resume SGD                       ')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # Load the model
    model = ParameterReoveryModel(params).to(device)
    model.load_state_dict(torch.load(PATH))
    # Specify gradient weights
    grad_weights = {
        'dm': 0.00,
        'de': 0.00,
        'dw': 0.00,
        'dN': 0.00,
        'di': 0.00,
        'a0': 1e1
    }
    # Train the model
    losses = train_model(grad_weights, 1E-2, 200)

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
    # Define grid-search points
    dmrange = np.linspace(minimizer*0.9, minimizer*1.1, 1000)
    minimizer, new_loss = brute(dmrange, losses[-1])
    # refine, repeat
    dmrange = np.linspace(minimizer*0.9995, minimizer*1.0005, 1000)
    minimizer, new_loss = brute(dmrange, losses[-1])
    # Save the model
    torch.save(model.state_dict(), PATH)

    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Phase 4 - Resume SGD                 ')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # Load the model
    torch.cuda.empty_cache()
    model = ParameterReoveryModel(params).to(device)
    model.load_state_dict(torch.load(PATH))
    # Specify gradient weights
    grad_weights = {
        'dm': 1.00E-23,
        'de': 0.00,
        'dw': 0.00,
        'dN': 0.00,
        'di': 0.00
    }

    # Train the model
    losses = train_model(grad_weights, 1.0E-2, 1000)

    # Print learned parameter values
    print_parameter_values()
    # Plot the losses
    loss_plot(losses, 20)

    # Save the model
    torch.save(model.state_dict(), PATH)

    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Phase 2 - Orbital Period Estimation v3')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Using brute-force to pick a global minimizer for dm...')
    # Load the model
    model = ParameterReoveryModel(params).to(device)
    model.load_state_dict(torch.load(PATH))
    # Define grid-search points
    dmrange = np.linspace(minimizer*0.9, minimizer*1.1, 2000)
    minimizer, new_loss = brute(dmrange, losses[-1])
    # refine, repeat
    dmrange = np.linspace(minimizer*0.95, minimizer*1.05, 2000)
    minimizer, new_loss = brute(dmrange, losses[-1])
    # Save the model
    torch.save(model.state_dict(), PATH)

    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Phase 5 - Allowing 1st order terms   ')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # Load the model
    torch.cuda.empty_cache()
    model = ParameterReoveryModel(params).to(device)
    model.load_state_dict(torch.load(PATH))
    # Specify gradient weights - all can change now
    grad_weights = {
        'dm': 1.00E-21,
        'de': 1.00E-17,
        'dw': 1.00E-17,
        'dN': 1.00E-17,
        'di': 1.00E-17,
        'w0': 1.00E3,
        'N': 1.00E3,
        'i': 1.00E3
    }

    # Train the model
    losses = train_model(grad_weights, 1.0E-3, 3000, tol=1E-10)

    # Print learned parameter values
    print_parameter_values()
    # Plot the losses
    loss_plot(losses, 20)

    # Save the model
    torch.save(model.state_dict(), PATH)

    t_end = time()

    print()
    print('Total elapsed time:', t_end - t_start)

    print()
    print('~~~~~~~~~~~~~~~~~~~~~')
    print('Residuals w.r.t. time')
    print('~~~~~~~~~~~~~~~~~~~~~')
    # Load the model
    model = ParameterReoveryModel(params).to(device)
    model.load_state_dict(torch.load(PATH))
    plot_residual(x_train_tensor, y_train_tensor, 1000)
    # Load the model (restore)
    model = ParameterReoveryModel(params).to(device)
    model.load_state_dict(torch.load(PATH))

    #######################################
    # Output learned parameters to a file #
    #######################################
    # Load the model
    model = ParameterReoveryModel(params).to(device)
    model.load_state_dict(torch.load(PATH))
    # Output parameters
    print_parameter_values(file=True)

    print()
    print('~~~~~~~~~~~~~~~~~~~~~')
    print('Validation Error     ')
    print('~~~~~~~~~~~~~~~~~~~~~')
    # Convert test data to tensors
    x_test_tensor = torch.from_numpy(x_test).to(device)
    y_test_tensor = torch.from_numpy(y_test).to(device)
    test_error(x_test_tensor, y_test_tensor)

    plot_prediction_test()
