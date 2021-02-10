from unary import rw_parameters, get_pdf
import numpy as np
import matplotlib.pyplot as plt
from aux_functions import log_normal

S0 = 2
K = 1.9
sig = 0.4
r = 0.05
T = .1
qu=20

fig, ax = plt.subplots()
lognormal_parameters = []
(values, pdf), (mu, mean, variance) = get_pdf(qu, S0, sig, r, T)
for t in np.linspace(0.01, T, 15):
    mu = (r - 0.5 * sig ** 2) * t + np.log(S0)
    pdf = log_normal(values, mu, sig * np.sqrt(t))
    ax.plot(values, pdf)
    lognormal_parameters.append(rw_parameters(qu, pdf))

plt.show()
lognormal_parameters = np.array(lognormal_parameters).transpose()
fig, ax = plt.subplots()

from scipy.optimize import curve_fit
def func(x, a, b, c):
    return a * x**b + c



for i in range(qu - 1):
    #ax.plot(np.linspace(0.01, T, 15), lognormal_parameters[i])
    popt, pcov = curve_fit(func, np.linspace(0.01, T, 15), lognormal_parameters[i], maxfev=10000)
    print(popt)
    print(np.mean((func(np.linspace(0.01, T, 15), *popt) - lognormal_parameters[i])**2))
    ax.plot(np.linspace(0.01, T, 15), lognormal_parameters[i])

plt.show()
