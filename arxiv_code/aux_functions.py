import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt

def log_normal(x, mu, sig):
    """
    Lognormal probability distribution function normalized for representation in finite intervals
    :param x: variable of the probability distribution
    :param mu: variable mu
    :param sig: variable sigma
    :return: Probability distribution for x according to log_normal(mu, sig)
    """
    dx = x[1]-x[0]
    log_norm = 1 / (x * sig * np.sqrt(2 * np.pi)) * np.exp(- np.power(np.log(x) - mu, 2.) / (2 * np.power(sig, 2.)))
    f = log_norm*dx/(np.sum(log_norm * dx))
    return f

def classical_payoff(S0, sig, r, T, K, samples=10000):
    """
    Function computing the payoff classically given some data.
    :param S0: initial price
    :param sig: volatilities
    :param r: interest rate
    :param T: maturity date
    :param K: strike
    :return: classical payoff
    """
    mu = (r - 0.5 * sig ** 2) * T + np.log(S0)  # Define all the parameters to be used in the computation
    mean = np.exp(mu + 0.5 * T * sig ** 2)  # Set the relevant zone of study and create the mapping between qubit and option price, and
    # generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
    Sp = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), samples)
    lnp = log_normal(Sp, mu, sig * np.sqrt(T))
    cl_payoff = 0
    for i in range(len(Sp)):
        if K < Sp[i]:
            cl_payoff += lnp[i] * (Sp[i] - K)

    return cl_payoff

def KL(p, q):
    return np.sum(p * np.log(p / q))


def find_next_j(j, theta_l, theta_h, up):
    J_ = 4 * j + 2
    theta_min = J_ * theta_l
    theta_max = J_ * theta_h
    print(theta_h - theta_l)
    J_max = (np.floor(np.pi / (theta_h - theta_l)))
    J = J_max - (J_max - 2) % 4

    while J >= 2 * J_:
        q = J / J_
        if np.remainder(q * theta_max, 2 * np.pi) < np.pi and np.remainder(q * theta_min, 2 * np.pi) < np.pi:
            J_ = J
            up = True
            j = (J_ - 2) / 4
            return j, up
        elif np.remainder(q * theta_max, 2 * np.pi) >= np.pi and np.remainder(q * theta_min, 2 * np.pi) >= np.pi:
            J_ = J
            up = False
            j = (J_ - 2) / 4
            return j, up

        else:
            J = J - 4

    return (j, up)

def chernoff(m, c):
    try:
        delta_plus = np.real(newton(bound, 1, args=(m, c / 2))) # Aquí hay problemas
        delta_minus = np.real(newton(bound, -1, args=(m, c / 2)))
    except:
        delta_plus = np.sqrt(2 / m * np.log(2 / c))
        delta_minus = - delta_plus
    m_max, m_min = m * (1 + delta_plus), m * (1 + delta_minus)
    return m_min, m_max

def bound(delta, mu, constant):
    return np.power((np.exp(delta))/(1 + delta)**(1+delta), mu) - constant