import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, execute, Aer, QuantumRegister, ClassicalRegister
from qiskit.aqua.components.uncertainty_models import LogNormalDistribution
from aux_functions import log_normal, classical_payoff

dev = Aer.get_backend('qasm_simulator')
def extract_probability(qubits, counts, samples):
    """
    From the retuned sampling, extract probabilities of all states
    :param qubits: number of qubits of the circuit
    :param counts: Number of measurements extracted from the circuit
    :param samples: Number of measurements applied the circuit
    :return: probabilities of unary states
    """
    form = '{0:0%sb}' % str(qubits) # qubits?
    prob = []
    for i in range(2**qubits):
        prob.append(counts.get(form.format(i), 0)/samples)
    return prob

def qOr(QC, qubits):
    """
    Component for the comparator
    :param QC: Quantum circuit the comparator is added to
    :param qubits: Number of qubits
    :return: None, updates QC
    """

    QC.x(qubits[0])
    QC.x(qubits[1])
    QC.x(qubits[2])
    QC.ccx(qubits[0], qubits[1], qubits[2])
    QC.x(qubits[0])
    QC.x(qubits[1])

def ccry(QC, ctrls, targ, theta):
    """
    Component for the payoff calculator
    :param QC: Quantum circuit the payoff calculator is added to
    :param ctrls: Control qubits
    :param targ: Target qubit
    :param theta: Parameters of the payoff calculator
    :return: None, updates quantum circuit
    """

    QC.cu3(theta / 2, 0, 0, ctrls[1], targ)
    QC.cx(ctrls[0], ctrls[1])
    QC.cu3( -theta / 2, 0, 0, ctrls[1], targ)
    QC.cx(ctrls[0], ctrls[1])
    QC.cu3( theta / 2, 0, 0, ctrls[0], targ)
    

def load_quantum_sim(qu, S0, sig, r, T):
    """
    Function to create quantum circuit for the binary representation to return an approximate probability distribution.
        This function is thought to be the prelude of run_quantum_sim
    :param qu: Number of qubits
    :param S0: Initial price
    :param sig: Volatility
    :param r: Interest rate
    :param T: Maturity date
    :return: quantum circuit, backend, (values of prices, prob. distri. function), (mu, mean, variance)
    """
    mu = ((r - 0.5 * sig ** 2) * T + np.log(S0)) #parameters for the log_normal distribution
    sigma = sig * np.sqrt(T)
    mean = np.exp(mu + sigma ** 2 / 2)
    variance = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
    stddev = np.sqrt(variance)

    # lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
    low = np.maximum(0, mean - 3 * stddev)
    high = mean + 3 * stddev
    #S = np.linspace(low, high, samples)

    # construct circuit factory for uncertainty model
    uncertainty_model = LogNormalDistribution(qu, mu=mu, sigma=sigma, low=low, high=high)

    values = uncertainty_model.values
    pdf = uncertainty_model.probabilities

    qr = QuantumRegister(qu)
    cr = ClassicalRegister(qu)
    qc = QuantumCircuit(qr, cr)
    uncertainty_model.build(qc, qr)
    qc.measure(np.arange(0, qu), np.arange(0,qu))

    return qc, (values, pdf), (mu, mean, variance)

def run_quantum_sim(qu, qc, shots, basis_gates, noise_model):
    """
    Function to execute quantum circuit for the binary representation to return an approximate probability distribution.
        This function is thought to be preceded by load_quantum_sim
    :param qubits: Number of qubits
    :param qc: quantum circuit
    :param dev: backend
    :param shots: number of measurements
    :param basis_gates: native gates of the device
    :param noise_model: noise model
    :return: approximate probability distribution
    """
    job = execute(qc, dev, shots=shots, basis_gates=basis_gates, noise_model=noise_model)
    result = job.result()
    counts = result.get_counts(qc)

    prob_sim = extract_probability(qu, counts, shots)

    return prob_sim

def comparator(qc, q, K, high, low):
    """
    Circuit that codifies the comparator of the option
    :param qc: quantum circuit the comparator is added to
    :param q: number of qubits
    :param K: strike
    :param high: highest value for prices
    :param low: lowest value for prices
    :return: k, updates the circuit
    """

    k = int(np.floor(2 ** q * (K - low) / (high - low)))
    t = 2 ** q - k
    t_bin = np.binary_repr(t, q)
    if t_bin[-1] == '1':
        qc.cx(0, q)

    for i in range(q - 1):
        if t_bin[-2 - i] == '0':
            qc.ccx(1 + i, q + i, q + 1 + i)
        elif t_bin[-2 - i] == '1':
            qOr(qc, [1 + i, q + i, q + 1 + i])

    qc.cx(2 * q - 1, 2 * q)
    return k


def rotations(qc, q, k, u=0, error=0.05):
    """
    Circuit for the rotations of the payoff circuit
    :param qc: quantum circuit the rotations are added to
    :param q: number of qubits
    :param k: value k
    :param u: value u
    :param error: error of the approximation
    :return: value c
    """

    c = (2 * error) ** (1 / (2 * u + 2))
    g0 = 0.25 * np.pi - c
    qc.ry(2 * g0, 2 * q + 1)
    qc.cu3(4 * c * k / (k - 2 ** q + 1), 0, 0, 2 * q, 2 * q + 1)
    for _ in range(q):
        ccry(qc, [_, 2 * q], 2 * q + 1, 2 ** (2 + _) * c / (2 ** q - 1 - k))

    return c


def payoff_circuit(qc, q, K, high, low):
    """
    Setting all pieces for the payoff (comparatos + rotations) together
    :param qc: quantum circuit
    :param q: number of qubits
    :param K: strike
    :param high: highest value of prices
    :param low: lowest value of prices
    :return: k, c; updates qc
    """

    k = comparator(qc, q, K, high, low)
    c = rotations(qc, q, k)

    qc.measure([2*q+1], [0])

    return k, c

def load_payoff_quantum_sim(qu, S0, sig, r, T, K):
    """
    Function to create quantum circuit for the unary representation to return an approximate probability distribution.
        This function is thought to be the prelude of run_payoff_quantum_sim
    :param qu: Number of qubits
    :param S0: Initial price
    :param sig: Volatility
    :param r: Interest rate
    :param T: Maturity date
    :param K: strike
    :return: quantum circuit, backend, prices
    """

    mu = ((r - 0.5 * sig ** 2) * T + np.log(S0))  # parameters for the log_normal distribution
    sigma = sig * np.sqrt(T)
    mean = np.exp(mu + sigma ** 2 / 2)
    variance = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
    stddev = np.sqrt(variance)

    # lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
    low = np.maximum(0, mean - 3 * stddev)
    high = mean + 3 * stddev
    #S = np.linspace(low, high, samples)

    # construct circuit factory for uncertainty model
    uncertainty_model = LogNormalDistribution(qu, mu=mu, sigma=sigma, low=low, high=high)

    #values = uncertainty_model.values
    #pdf = uncertainty_model.probabilities

    qr = QuantumRegister(2*qu + 2)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    uncertainty_model.build(qc, qr)

    k, c = payoff_circuit(qc, qu, K, high, low)

    dev = Aer.get_backend('qasm_simulator')

    return c, k, high, low, qc, dev

def run_payoff_quantum_sim(qu, c, k, high, low, qc, dev, shots, basis_gates, noise_model):
    """
    Function to execute quantum circuit for the unary representation to return an approximate payoff.
        This function is thought to be preceded by load_quantum_sim
    :param qubits: Number of qubits
    :param qc: quantum circuit
    :param dev: backend
    :param shots: number of measurements
    :param basis_gates: native gates of the device
    :param noise_model: noise model
    :return: approximate payoff
    """
    job = execute(qc, dev, shots=shots, basis_gates = basis_gates, noise_model = noise_model)
    counts = job.result().get_counts(qc)

    prob = counts['1'] / (counts['1'] + counts['0'])
    qu_payoff = ((prob - 0.5 + c) * (2 ** qu - 1 - k) / 2 / c) * (high - low) / 2 ** qu

    return qu_payoff


def diff_qu_cl(qu_payoff_sim, cl_payoff):
    """
    Comparator of quantum and classical payoff
    :param qu_payoff_sim: quantum approximation for the payoff
    :param cl_payoff: classical payoff
    :return: Relative error
    """
    error = np.abs(100 * (cl_payoff - qu_payoff_sim) / cl_payoff)

    return error

def diffusion_operator(qubits):
    C = QuantumCircuit(2 * qubits+1)

    for q in range(qubits):
        C.x(q)
    C.h(qubits)
    C.mct([range(qubits)], qubits, ancilla_qubits=[range(qubits, 2 * (qubits - 1), 1)],mode='v-chain')  #THIS IS A VERY HEAVY GATE!!!!!
    for q in range(qubits):
        C.x(q)
    C.h(qubits)
    return C


def oracle_operator(qubits):
    C = QuantumCircuit(qubits + 1)
    C.z(qubits)

    return C


def load_Q_operator(qu, iterations, S0, sig, r, T, K):
    """
    Function to create quantum circuit for the unary representation to return an approximate probability distribution.
        This function is thought to be the prelude of run_payoff_quantum_sim
    :param qu: Number of qubits
    :param S0: Initial price
    :param sig: Volatility
    :param r: Interest rate
    :param T: Maturity date
    :param K: strike
    :return: quantum circuit, backend, prices
    """

    mu = ((r - 0.5 * sig ** 2) * T + np.log(S0))  # parameters for the log_normal distribution
    sigma = sig * np.sqrt(T)
    mean = np.exp(mu + sigma ** 2 / 2)
    variance = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
    stddev = np.sqrt(variance)

    # lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
    low = np.maximum(0, mean - 3 * stddev)
    high = mean + 3 * stddev
    #S = np.linspace(low, high, samples)

    # construct circuit factory for uncertainty model
    uncertainty_model = LogNormalDistribution(qu, mu=mu, sigma=sigma, low=low, high=high)

    #values = uncertainty_model.values
    #pdf = uncertainty_model.probabilities

    qr = QuantumRegister(2*qu + 2)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    uncertainty_model.build(qc, qr)

    k, c = payoff_circuit(qc, qu, K, high, low)

    prob_loading = uncertainty_model.build(qc, qr)
    prob_loading_inv = prob_loading.inv()
    payoff = payoff_circuit(qu, K, S)
    payoff_inv = payoff_circuit_inv(qu, K, S)

    diffusion = diffusion_operator(qu)
    oracle = oracle_operator(qu)

    qc = prob_loading + payoff
    qc_Q = oracle + payoff_inv + prob_loading_inv + diffusion + prob_loading + payoff
    for i in range(iterations):
        qc += qc_Q
    qc += measure_payoff(qu)

    return qc

    return c, k, high, low, qc, dev

def run_payoff_quantum_sim(qu, c, k, high, low, qc, dev, shots, basis_gates, noise_model):
    """
    Function to execute quantum circuit for the unary representation to return an approximate payoff.
        This function is thought to be preceded by load_quantum_sim
    :param qubits: Number of qubits
    :param qc: quantum circuit
    :param dev: backend
    :param shots: number of measurements
    :param basis_gates: native gates of the device
    :param noise_model: noise model
    :return: approximate payoff
    """
    job = execute(qc, dev, shots=shots, basis_gates = basis_gates, noise_model = noise_model)
    counts = job.result().get_counts(qc)

    prob = counts['1'] / (counts['1'] + counts['0'])
    qu_payoff = ((prob - 0.5 + c) * (2 ** qu - 1 - k) / 2 / c) * (high - low) / 2 ** qu

    return qu_payoff


def diff_qu_cl(qu_payoff_sim, cl_payoff):
    """
    Comparator of quantum and classical payoff
    :param qu_payoff_sim: quantum approximation for the payoff
    :param cl_payoff: classical payoff
    :return: Relative error
    """
    error = np.abs(100 * (cl_payoff - qu_payoff_sim) / cl_payoff)

    return error
