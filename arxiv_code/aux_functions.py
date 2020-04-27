import numpy as np

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

def classical_payoff(S0, sig, r, T, K):
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
    samples = 10000
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