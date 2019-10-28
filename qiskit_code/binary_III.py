import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, execute, Aer, QuantumRegister, ClassicalRegister
from qiskit.aqua.components.uncertainty_models import LogNormalDistribution
from unary_III import log_normal, classical_payoff


def extract_probability(qubits, counts, samples):
    """
    From the retuned sampling, extract only probabilies of unary states
    """
    form = '{0:0%sb}' % str(qubits) # qubits?
    prob = []
    for i in range(2**qubits):
        prob.append(counts.get(form.format(i), 0)/samples)
    return prob

def qOr(QC, qubits):
    """
    Component for the comparator
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
    """
    QC.cu3(theta / 2, 0, 0, ctrls[1], targ)
    QC.cx(ctrls[0], ctrls[1])
    QC.cu3( -theta / 2, 0, 0, ctrls[1], targ)
    QC.cx(ctrls[0], ctrls[1])
    QC.cu3( theta / 2, 0, 0, ctrls[0], targ)
    

def binary_quantum_sim(qu, S0, sig, r, T, noise_objects):
    #samples = 2 ** qu

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
    shots = 10000

    dev = Aer.get_backend('qasm_simulator')
    noise_model, basis_gates, error = noise_objects
    job = execute(qc, dev, shots=shots, basis_gates=basis_gates, noise_model=noise_model)
    result = job.result()
    counts = result.get_counts(qc)

    prob_sim = extract_probability(qu, counts, shots)

    return (values, pdf, prob_sim), (mu, mean, variance)
    #cl_payoff = classical_payoff(values, pdf, K)

def binary_benchmark_sim(qu, S0, sig, r, T, noise_objects, err):
    (S, ln, prob_sim), (mu, mean, variance) = binary_quantum_sim(qu, S0, sig, r, T, noise_objects)
    Sp = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), 2 ** qu * 100)  # Generate a the exact target distribution to benchmark against quantum results
    lnp = log_normal(Sp, mu, sig, T)

    '''
    Plot the probability of each unary state on the corresponding option price and
    check against the exact probability drsitribution function for the target lognormal

    '''

    width = (S[1] - S[0]) / 1.2
    fig, ax = plt.subplots()
    rects2 = ax.bar(S, prob_sim, width, label='Quantum', alpha=0.8)
    ax.plot(Sp, (S[1] - S[0]) * lnp / (Sp[1] - Sp[0]), 'C1', label='Exact')
    ax.scatter(S, ln, s=500, color='C1', label='Exact', marker='_')
    #ax.vlines(K, 0, max(prob_sim), linestyles='dashed', label='K = {}'.format(K))  # Strike price marker
    plt.ylabel('Probability')
    plt.xlabel('Option price')
    plt.title('Option price distribution for {} qubits - {} error: {} - qasm_simulator'.format(qu, err, (noise_objects[2])))
    ax.legend()
    fig.tight_layout()
    text='binary/qubits.{}_S0.{}_sig.{}_r.{}_T.{}_error.{}_{}.png'.format(qu, S0, sig, r, T, noise_objects[2], err)
    fig.savefig(text)
    fig.savefig('binary/try.png')

def comparator(qc, q, K, high, low):
    '''
    Circuit that codifies the comparator of the option
    '''
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
    """
    k = comparator(qc, q, K, high, low)
    c = rotations(qc, q, k)

    qc.measure([2*q+1], [0])

    return k, c#We need to keep track of these variables

def payoff_quantum_sim(qu, S0, sig, r, T, K, noise_objects, printer=True):
    """
    Order for
    """
    #samples = 2 ** qu
    shots=10000

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
    noise_model, basis_gates, error = noise_objects
    job = execute(qc, dev, shots=shots, basis_gates = basis_gates, noise_model = noise_model)
    #result = job.result()
    counts = job.result().get_counts(qc)

    prob = counts['1'] / (counts['1'] + counts['0'])

    qu_payoff = ((prob - 0.5 + c) * (2 ** qu - 1 - k) / 2 / c) * (high - low) / 2 ** qu

    return qu_payoff


def binary_qu_cl(qu, S0, sig, r, T, K, noise_objects, cl_payoff):
    qu_payoff_sim = payoff_quantum_sim(qu, S0, sig, r, T, K, noise_objects)
    error = np.abs(100 * (cl_payoff - qu_payoff_sim) / cl_payoff)


    return qu_payoff_sim, error

