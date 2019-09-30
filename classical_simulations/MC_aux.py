"""
This file simulates the Black-Scholes model using MonteCarlo samples
It will perform the calculations and return a matrix with data for all timesteps


"""

import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def MC(S0, r, sigma, T, dt, N, checks=10, bins=0):
    foldname = _name_folder(S0, r, sigma, T, dt, N, checks)
    if foldname.replace('classical_simulations/data/', '') in os.listdir('classical_simulations/data'):
        print('Already computed')
        data, opt, time = _read_file((S0, r, sigma, T, dt, N, checks))
    # if estos datos ya están calculados --> Copiarlos de la carpeta
    else:
        print('Not computed yet')
        paths, time = _many_paths(S0, r, sigma, T, dt, N, checks)
        data = _create_hists(paths, bins)
        opt = _fit_hists(data)
        variables = (S0, r, sigma, T, dt, N, checks)
        _write_file(data, opt, time, variables)
    # if no estan calculados --> hacerlos

    return data, opt, time



def _name_folder(S0, r, sigma, T, dt, N, checks):
    return 'classical_simulations/data/S0:{}_r:{}_sigma:{}_T:{}_dt:{}_N:{}_checks:{}'.format(S0, r, sigma, T, dt, N, checks)

def _data_file(name):
    return name + '/data.txt'

def _opt_file(name):
    return name + '/opt.txt'

def _time_file(name):
    return name + '/time.txt'


def _write_file(data, opt, time, variables):
    S0, r, sigma, T, dt, N, checks = variables
    name = _name_folder(S0, r, sigma, T, dt, N, checks)
    _create_folder(name)
    data = np.array(data)
    np.savetxt(_data_file(name), data.flatten())
    np.savetxt(_opt_file(name), opt)
    np.savetxt(_time_file(name), time)


def _read_file(variables):
    S0, r, sigma, T, dt, N, checks = variables
    name = _name_folder(S0, r, sigma, T, dt, N, checks)
    data = np.loadtxt(_data_file(name))
    L = len(data)
    b = L // (2 * checks)
    data = data.reshape((checks, 2, b))
    opt = np.loadtxt(_opt_file(name))
    time = np.loadtxt(_time_file(name))

    return data, opt, time

def _create_folder(directory):
    """
    Auxiliar function for creating directories with name directory

    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def _increment(St, r, sigma, dt):
    dSt = St * r * dt + St * sigma * np.sqrt(dt) * np.random.normal(0, 1)
    return dSt


def _path(S0, r, sigma, t, dt, check_points):
    S = S0
    path = np.zeros_like(check_points)
    k = 0
    for i in range(len(t)):
        S += _increment(S, r, sigma, dt)
        if i in check_points:
            path[k] = S
            k += 1

    return path

def _many_paths(S0, r, sigma, T, dt, N, checks):
    t = np.arange(0, T, dt)
    check_points = np.arange(len(t) - 1, 0, -len(t) // checks)
    t_s = np.empty_like(check_points)
    paths = np.empty((len(check_points), N))
    for j in range(N):
        paths[:, j] = _path(S0, r, sigma, t, dt, check_points)

    for i, c in enumerate(check_points):
        t_s[i] = t[c]

    return paths, t_s


def _create_hists(paths, bins):
    if bins==0:
        bins = paths.shape[1] // 50

    paths_data = [[]]*paths.shape[0]
    i = 0
    for path in paths:
        print(path.shape)
        bin_edges = np.logspace(np.log10(max(np.min(path), 1e-5)), np.log10(np.max(path)), bins + 1)
        centers = np.array([np.sqrt(bin_edges[i] * bin_edges[[1 + i]]) for i in range(bins)])
        print(np.max(path), np.min(path))
        hist, bin_edges = np.histogram(path, bins=bin_edges, density=True)
        path_data = np.empty((2, len(hist)))
        path_data[0] = centers.flatten()
        path_data[1] = hist.flatten()
        paths_data[i] = path_data
        i += 1

    return paths_data


def _fit_hist(hist_data):
    popt, pcov = curve_fit(_log_normal, hist_data[0], hist_data[1])

    return popt

def _fit_hists(paths_data):
    popts = [[]]*len(paths_data)
    for i, path_data in enumerate(paths_data):
        popt = _fit_hist(path_data)
        popts[i] = popt

    return popts

def _log_normal(x, mu, sig):
    return 1 / (x * sig * np.sqrt(2 * np.pi)) * np.exp(- np.power(np.log(x) - mu, 2.) / (2 * np.power(sig, 2.)))


def painting_MC(S0, r, sigma, T, dt, N, checks=10, bins=0):
    data, opt, time = MC(S0, r, sigma, T, dt, N, checks=checks, bins=bins)
    for d, (o, t) in zip(data, zip(opt, time)):
        fig, ax = plt.subplots()
        ax.bar(d[0], d[1], color='black', alpha=0.5, label='MonteCarlo Simulation')
        fitted = _log_normal(d[0], o[0], o[1])
        ax.scatter(d[0], fitted, color='red', alpha=1, s=2, label='Fit to LogNormal')
        theory = _log_normal(d[0], (r - 0.5 * sigma ** 2) * t + np.log(S0), sigma * np.sqrt(t))
        ax.scatter(d[0], theory, color='blue', alpha=1, s=2, label='Theoretical Solution')
        ax.set(xlabel=r'$S_t$', ylabel='Probability distribution')
        ax.legend()
        fig.savefig(_name_folder(S0, r, sigma, T, dt, N, checks) + '/t:{}.png'.format(t))
