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
    return np.power((np.exp(delta))/(1 + delta)**(1 + delta), mu) - constant


def max_likelihood(theta, m_s, ones_s, zeroes_s, f = .1, spread=.1):
    if len(m_s) != len(ones_s) or len(m_s) != len(zeroes_s):
        raise ValueError('Dimension mismatch')
    length=len(theta)
    L = 1
    theta_max_s = np.zeros(len(m_s))
    theta_max_s.fill(np.nan)
    error_s = np.zeros(len(m_s))
    error_s.fill(np.nan)
    for i in range(len(m_s)):
        # print(m_s[i], ones_s[i], zeroes_s[i])
        L *= (np.sin((2 * m_s[i] + 1) * theta)**(2))**ones_s[i]
        L *= (np.cos((2 * m_s[i] + 1) * theta)**(2))**zeroes_s[i]
        L = L / np.max(L)
        # L = L ** (sms[i])
        #plt.plot(theta, L, c='C1')
        #plt.show()

        arg_max = np.argmax(L)
        if arg_max == 0 or arg_max == len(theta) - 1:
            break
        max_L = np.max(L)
        value_plus = 1
        j_plus=0
        while value_plus > f:
            j_plus += 1
            try:
                value_plus = L[arg_max + j_plus]
            except:
                j_plus = length
                break

        value_minus = 1
        j_minus = 0
        while value_minus > max_L * f:
            j_minus -= 1
            try:
                value_minus = L[arg_max + j_minus]
            except:
                j_minus = -length
                break

        try:
            arg_plus = theta[arg_max + j_plus]
            arg_minus = theta[arg_max + j_minus]

        except:
            if j_plus == length:
                arg_minus = theta[arg_max + j_minus]
                arg_plus = 2 * np.max(theta) - arg_minus
            elif j_minus == -length:
                arg_plus = theta[arg_max + j_plus]
                arg_minus = 2 * np.min(theta) - arg_plus
            else:
                print('failed')

        l = arg_plus - arg_minus
        theta = np.linspace(arg_minus - spread * l, arg_plus + spread * l, length)
        #print(arg_minus, arg_plus)
        L=1
        L *= (np.sin((2 * m_s[i] + 1) * theta) ** (2)) ** ones_s[i]
        L *= (np.cos((2 * m_s[i] + 1) * theta) ** (2)) ** zeroes_s[i]
        L = L / np.max(L)
        #plt.plot(theta, L)
        #plt.show()
        theta_max_s[i] = theta[arg_max]
        error_s[i] = l

    # print(theta_max_s, error_s)
    return theta_max_s, error_s

def detect_sign(array):
    asign = np.sign(array)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    signchange[0]=0
    return signchange

def experimental_data(data, conf):
    bool_index = ~np.isnan(data)
    valids = np.sum(bool_index)
    if valids == 0:
        data_mean = np.nan
        data_std = np.nan
        conf_mean = np.nan
        conf_std = np.nan
    else:
        data_mean = np.nanmean(data)
        data_std = np.nanstd(data)
        conf_mean = np.nanmean(conf)
        conf_std = np.nanstd(conf)


    # print((data_mean, data_std), (conf_mean, conf_std), np.sum(bool_index))
    return (data_mean, data_std), (conf_mean, conf_std), np.sum(bool_index)