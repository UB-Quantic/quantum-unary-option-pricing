from classical_simulations.MC_aux import MC, _log_normal
import matplotlib.pyplot as plt
import numpy as np

def painting(S0, r, sigma, T, dt, N, checks=10, bins=0):
    data, opt, time = MC(S0, r, sigma, T, dt, N, checks=checks, bins=bins)
    for d, (o, t) in zip(data, zip(opt, time)):
        fig, ax = plt.subplots()
        ax.bar(d[0], d[1], color='black', alpha=0.5, label='MonteCarlo Simulation')
        fitted = _log_normal(d[0], o[0], o[1])
        ax.scatter(d[0], fitted, color='red', alpha=0.8, s=2, label='Fit to LogNormal')
        theory = _log_normal(d[0], (r - 0.5 * sigma ** 2) * t + np.log(S0), sigma * np.sqrt(t))
        ax.scatter(d[0], theory, color='blue', alpha=0.8, s=2, label='Theoretical Solution')
        ax.set(xlabel=r'$S_t$', ylabel='Probability distribution')
        fig.savefig('')