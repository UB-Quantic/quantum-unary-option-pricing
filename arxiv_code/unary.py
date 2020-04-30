import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, execute, Aer
from aux_functions import log_normal, classical_payoff, chernoff, find_next_k
from binary import extract_probability as extract_probability_all_samples

"""
This file provides all required functions for performing calculations in unary basis
"""

dev = Aer.get_backend('qasm_simulator')  # Get qasm backend for the simulation of the circuit. With this line, all different references can be removed

def rw_circuit(qubits, parameters):
    print(dev)
    """
    Circuit that emulates the probability sharing between neighbours seen usually
    in Brownian motion and stochastic systems
    :param qubits: number of qubits of the circuit
    :param parameters: parameters for performing the circuit
    :return: Quantum circuit
    """
    C = QuantumCircuit(qubits+1)
    if qubits%2==0:
        mid1 = int(qubits/2)
        mid0 = int(mid1-1)
        C.x(mid1)
        #SWAP-Ry gates
        #-------------------------------------------------
        C.cx(mid1, mid0)
        C.cu3(parameters[mid0], 0, 0, mid0, mid1)
        C.cx(mid1, mid0)
        #-------------------------------------------------
        for i in range(mid0):
            #SWAP-Ry gates
            #-------------------------------------------------
            C.cx(mid0-i, mid0-i-1)
            C.cu3(parameters[mid0-i-1], 0, 0, mid0-i-1, mid0-i)
            C.cx(mid0-i, mid0-i-1)
            #-------------------------------------------------
            C.cx(mid1+i, mid1+i+1)
            C.cu3(parameters[mid1+i], 0, 0, mid1+i+1, mid1+i)
            C.cx(mid1+i, mid1+i+1)
            #-------------------------------------------------
    else:
        mid = int((qubits-1)/2) #The random walk starts from the middle qubit
        C.x(mid)
        for i in range(mid):
            #SWAP-Ry gates
            #-------------------------------------------------
            C.cx(mid-i, mid-i-1)
            C.cu3(parameters[mid-i-1], 0, 0, mid-i-1, mid-i)
            C.cx(mid-i, mid-i-1)
            #-------------------------------------------------
            C.cx(mid+i, mid+i+1)
            C.cu3(parameters[mid+i], 0, 0, mid+i+1, mid+i)
            C.cx(mid+i, mid+i+1)
            #-------------------------------------------------
    return C

def rw_circuit_inv(qubits, parameters):
    """
    Circuit that emulates the probability sharing between neighbours seen usually
    in Brownian motion and stochastic systems, INVERTED!
    :param qubits: number of qubits of the circuit
    :param parameters: parameters for performing the circuit
    :return: Quantum circuit
    """
    C = QuantumCircuit(qubits+1)
    if qubits%2==0:
        mid1 = int(qubits/2)
        mid0 = int(mid1-1)
        for i in range(mid0 - 1, -1, -1):
            # SWAP-Ry gates
            # -------------------------------------------------
            C.cx(mid0 - i, mid0 - i - 1)
            C.cu3(-parameters[mid0 - i - 1], 0, 0, mid0 - i - 1, mid0 - i)
            C.cx(mid0 - i, mid0 - i - 1)
            # -------------------------------------------------
            C.cx(mid1 + i, mid1 + i + 1)
            C.cu3(-parameters[mid1 + i], 0, 0, mid1 + i + 1, mid1 + i)
            C.cx(mid1 + i, mid1 + i + 1)
            # -------------------------------------------------

        # SWAP-Ry gates
        # -------------------------------------------------
        C.cx(mid1, mid0)
        C.cu3(-parameters[mid0], 0, 0, mid0, mid1)
        C.cx(mid1, mid0)

        C.x(mid1)

    else:
        mid = int((qubits-1)/2) #The random walk starts from the middle qubit
        C.x(mid)
        for i in range(mid, -1, -1):
            #SWAP-Ry gates
            #-------------------------------------------------
            C.cx(mid-i, mid-i-1)
            C.cu3(-parameters[mid-i-1], 0, 0, mid-i-1, mid-i)
            C.cx(mid-i, mid-i-1)
            #-------------------------------------------------
            C.cx(mid+i, mid+i+1)
            C.cu3(-parameters[mid+i], 0, 0, mid+i+1, mid+i)
            C.cx(mid+i, mid+i+1)
            #-------------------------------------------------
    return C

def rw_parameters(qubits, pdf):
    """
    Solving for the exact angles for the random walk circuit that enables the loading of
    a desired probability distribution function
    :param qubits: number of qubits of the circuit
    :param pdf: probability distribution function to emulate
    :return: set of parameters
    """
    if qubits%2==0:
        mid = qubits // 2
    else:
        mid = (qubits-1)//2 #Important to keep track of the centre
    last = 1
    parameters = []
    for i in range(mid-1):
        angle = 2 * np.arctan(np.sqrt(pdf[i]/(pdf[i+1] * last)))
        parameters.append(angle)
        last = (np.cos(angle/2))**2 #The last solution is needed to solve the next one
    angle = 2 * np.arcsin(np.sqrt(pdf[mid-1]/last))
    parameters.append(angle)
    last = (np.cos(angle/2))**2
    for i in range(mid, qubits-1):
        angle = 2 * np.arccos(np.sqrt(pdf[i]/last))
        parameters.append(angle)
        last *= (np.sin(angle/2))**2
    return parameters

def measure_probability(qubits):
    """
    Circuit to sample the created probability distribution function
    :param qubits: number of qubits of the circuit
    :return: circuit + measurements
    """
    C = QuantumCircuit(qubits+1, qubits)
    C.measure(range(qubits),range(qubits)) #No measure on the ancilla qubit is necessary
    return C

def extract_probability(qubits, counts, samples):
    """
    From the retuned sampling, extract only probabilities of unary states
    :param qubits: number of qubits of the circuit
    :param counts: Number of measurements extracted from the circuit
    :param samples: Number of measurements applied the circuit
    :return: probabilities of unary states
    """
    form = '{0:0%sb}' % str(qubits) # qubits?
    prob = []
    for i in range(qubits):
        prob.append(counts.get(form.format(2**i), 0)/samples)
    return prob

def load_quantum_sim(qu, S0, sig, r, T):
    """
    Function to create quantum circuit for the unary representation to return an approximate probability distribution.
        This function is thought to be the prelude of run_quantum_sim
    :param qu: Number of qubits
    :param S0: Initial price
    :param sig: Volatility
    :param r: Interest rate
    :param T: Maturity date
    :return: quantum circuit, backend, (values of prices, prob. distri. function), (mu, mean, variance)
    """
    dev = Aer.get_backend('qasm_simulator') # Get qasm backend for the simulation of the circuit
    mu = (r - 0.5 * sig ** 2) * T + np.log(S0) # Define all the parameters to be used in the computation
    mean = np.exp(mu + 0.5 * T * sig ** 2) # Set the relevant zone of study and create the mapping between qubit and option price, and
                                            #generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
    values = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qu)
    pdf = log_normal(values, mu, sig * np.sqrt(T))
    lognormal_parameters = rw_parameters(qu, pdf) # Solve for the parameters needed to create the target lognormal distribution
    prob_loading = rw_circuit(qu, lognormal_parameters) # Build the probaility loading circuit with the adjusted parameters
    qc = prob_loading + measure_probability(qu) #Circuit to test the precision of the probability loading algorithm

    return qc, dev, (values, pdf), (mu, mean, variance)

def run_quantum_sim(qubits, qc, dev, shots, basis_gates, noise_model):
    """
    Function to execute quantum circuit for the unary representation to return an approximate probability distribution.
        This function is thought to be preceded by load_quantum_sim
    :param qubits: Number of qubits
    :param qc: quantum circuit
    :param dev: backend
    :param shots: number of measurements
    :param basis_gates: native gates of the device
    :param noise_model: noise model
    :return: approximate probability distribution
    """
    job_sim = execute(qc, dev, shots=shots, basis_gates=basis_gates, noise_model=noise_model)
    counts_sim = job_sim.result().get_counts(qc)

    prob_sim = extract_probability_all_samples(qubits, counts_sim, shots)

    return prob_sim

def test_inversion_probability(qu, S0, sig, r, T, shots, basis_gates, noise_model):
    dev = Aer.get_backend('qasm_simulator')  # Get qasm backend for the simulation of the circuit
    mu = (r - 0.5 * sig ** 2) * T + np.log(S0)  # Define all the parameters to be used in the computation
    mean = np.exp(
        mu + 0.5 * T * sig ** 2)  # Set the relevant zone of study and create the mapping between qubit and option price, and
    # generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
    values = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qu)
    pdf = log_normal(values, mu, sig * np.sqrt(T))
    lognormal_parameters = rw_parameters(qu,
                                         pdf)  # Solve for the parameters needed to create the target lognormal distribution
    prob_loading = rw_circuit(qu,
                              lognormal_parameters)  # Build the probaility loading circuit with the adjusted parameters
    prob_loading_inv = rw_circuit_inv(qu,
                              lognormal_parameters)  # Build the probaility loading circuit with the adjusted parameters

    qc = prob_loading + prob_loading_inv + measure_probability(qu)  # Circuit to test the precision of the probability loading algorithm

    job_sim = execute(qc, dev, shots=shots, basis_gates=basis_gates, noise_model=noise_model)
    counts_sim = job_sim.result().get_counts(qc)

    prob_sim = extract_probability_all_samples(qu, counts_sim, shots)

    return prob_sim


def unary_benchmark_sim(qu, S0, sig, r, T, noise_objects, err): # No hace nada
    """

    :param qu:
    :param S0:
    :param sig:
    :param r:
    :param T:
    :param noise_objects:
    :param err:
    :return:
    """
    (S, ln, prob_sim), (mu, mean, variance) = unary_quantum_sim(qu, S0, sig, r, T, noise_objects)

    '''
    Plot the probability of each unary state on the corresponding option price and
    check against the exact probability drsitribution function for the target lognormal

    '''
    fig, ax = plt.subplots()
    Sp = np.linspace(0, 2**qu, 2**qu)
    S_bar = np.array([2**i for i in range(qu)])
    ax.scatter(Sp, prob_sim, s=50, color='black', label='Samples', marker='.')
    un_samples = [prob_sim[s] for s in S_bar]
    ax.bar(S_bar, un_samples, width=S_bar/10, label='Desired samples', color='C0')
    ax.scatter(S_bar, ln, s=500, color='C1', label='Exact', marker='_')
    plt.ylabel('Probability')
    plt.xlabel('Option price')
    plt.title('Option price distribution for {} qubits - {} error: {} - qasm_simulator'.format(qu, err, (0.001*noise_objects[2])))
    ax.legend()
    ax.set(xscale='log')
    fig.tight_layout()
    fig.savefig('unary/qubits:{}_S0:{}_sig:{}_r:{}_T:{}_error:{}_{}_all:.png'.format(qu, S0, sig, r, T, noise_objects[2], err))
      
    '''
    Plot both only de desired and all probabilities
    
    '''
    Sp = np.linspace(max(mean-3*np.sqrt(variance),0), mean+3*np.sqrt(variance), 2**qu) #Generate a the exact target distribution to benchmark against quantum results
    lnp = log_normal(Sp, mu, sig, T)
    width = (S[1] - S[0]) / 1.2
    
    fig, ax = plt.subplots()
    ax.bar(S, un_samples, width, label='Quantum', alpha=0.8)
    ax.scatter(S, ln, s=500, color='C1', label='Exact', marker='_')
    ax.plot(Sp, lnp* (S[1]- S[0]) / (Sp[1] - Sp[0]), 'C1')
    plt.ylabel('Probability')
    plt.xlabel('Option price')
    plt.title('Option price distribution for {} qubits - {} error: {} - qasm_simulator'.format(qu, err, (0.001*noise_objects[2])))
    ax.legend()
    fig.tight_layout()
    fig.savefig('unary/qubits:{}_S0:{}_sig:{}_r:{}_T:{}_error:{}_{}.png'.format(qu, S0, sig, r, T, noise_objects[2], err))


def payoff_circuit(qubits, K, S):
    """
    Circuit that codifies the expected payoff of the option into the probability of a
    single ancilla qubit
    :param qubits: Number of qubits
    :param K: strike
    :param S: prices
    :return: quantum circuit
    """
    C = QuantumCircuit(qubits+1)
    for i in range(qubits): #Determine the first qubit's price that
        qK = i              #surpasses the strike price
        if K<S[i]:
            break
    C.barrier(range(qubits+1))
    for i in range(qK, qubits):                              #Control-RY rotations controled by states
        angle = 2 * np.arcsin(np.sqrt((S[i]-K)/(S[qubits-1]-K))) #with higher value than the strike
        C.cu3(angle, 0, 0, i, qubits)                        #targeting the ancilla qubit
    return C

def payoff_circuit_inv(qubits, K, S):
    """
    Circuit that codifies the expected payoff of the option into the probability of a
    single ancilla qubit, INVERTED!
    :param qubits: Number of qubits
    :param K: strike
    :param S: prices
    :return: quantum circuit
    """
    C = QuantumCircuit(qubits+1)
    for i in range(qubits): #Determine the first qubit's price that
        qK = i              #surpasses the strike price
        if K<S[i]:
            break
    C.barrier(range(qubits+1))
    for i in range(qK, qubits):                              #Control-RY rotations controled by states
        angle = 2 * np.arcsin(np.sqrt((S[i]-K)/(S[qubits-1]-K))) #with higher value than the strike
        C.cu3(-angle, 0, 0, i, qubits)                        #targeting the ancilla qubit
    return C

def measure_payoff(qubits):
    """
    Function to measure the expected payoff of the option into the probability of a
    single ancilla qubit
    :param qubits: number of qubits
    :return: circuit of measurements
    """
    C = QuantumCircuit(qubits+1, qubits+1)
    C.measure(range(qubits+1), range(qubits+1))
    return C

def load_payoff_quantum_sim(qu, S0, sig, r, T, K):
    """
    Function to create quantum circuit for the unary representation to return an approximate probability distribution.
        This function is thought to be the prelude of run_payoff_quantum_sim
    :param qu: Number of qubits
    :param S0: Initial price
    :param sig: Volatility
    :param r: Interest rate
    :param T: Maturity date
    :return: quantum circuit, backend, prices
    """

    dev = Aer.get_backend('qasm_simulator')  # Get qasm backend for the simulation of the circuit
    mu = (r - 0.5 * sig ** 2) * T + np.log(S0)  # Define all the parameters to be used in the computation
    mean = np.exp(mu + 0.5 * T * sig ** 2)  # Set the relevant zone of study and create the mapping between qubit and option price, and
                                # generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
    S = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qu)
    ln = log_normal(S, mu, sig * np.sqrt(T))
    lognormal_parameters = rw_parameters(qu, ln)  # Solve for the parameters needed to create the target lognormal distribution
    prob_loading = rw_circuit(qu, lognormal_parameters)  # Build the probaility loading circuit with the adjusted parameters
    qc = prob_loading + payoff_circuit(qu, K, S) + measure_payoff(qu)

    return qc, dev, S


def run_payoff_quantum_sim(qu, qc, dev, shots, S, K, basis_gates, noise_model):
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
    job_payoff_sim = execute(qc, dev, shots=shots, basis_gates=basis_gates, noise_model=noise_model) #Run the complete payoff expectation circuit through a simulator
                                                                        #and sample the ancilla qubit
    counts_payoff_sim = job_payoff_sim.result().get_counts(qc)
    ones=0
    zeroes=0
    for key in counts_payoff_sim.keys(): # Post-selection
        unary = 0
        for i in range(1,qu+1):
            unary+=int(key[i])
        if unary==1:
            if int(key[0])==0:
                zeroes+=counts_payoff_sim.get(key)
            else:
                ones+=counts_payoff_sim.get(key)
            
    qu_payoff_sim = ones * (S[qu - 1]-K) / (ones+zeroes)

    return qu_payoff_sim

def test_inversion_payoff(qu, S0, sig, r, T, K, shots, basis_gates, noise_model):
    dev = Aer.get_backend('qasm_simulator')  # Get qasm backend for the simulation of the circuit
    mu = (r - 0.5 * sig ** 2) * T + np.log(S0)  # Define all the parameters to be used in the computation
    mean = np.exp(
        mu + 0.5 * T * sig ** 2)  # Set the relevant zone of study and create the mapping between qubit and option price, and
    # generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
    S = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qu)
    ln = log_normal(S, mu, sig * np.sqrt(T))
    lognormal_parameters = rw_parameters(qu,
                                         ln)  # Solve for the parameters needed to create the target lognormal distribution
    prob_loading = rw_circuit(qu,
                              lognormal_parameters)  # Build the probaility loading circuit with the adjusted parameters
    prob_loading_inv = rw_circuit_inv(qu,
                              lognormal_parameters)

    payoff = payoff_circuit(qu, K, S)
    payoff_inv = payoff_circuit_inv(qu, K, S)
    qc = prob_loading + payoff + payoff_inv + prob_loading_inv + measure_payoff(qu)

    job_payoff_sim = execute(qc, dev, shots=shots, basis_gates=basis_gates,
                             noise_model=noise_model)  # Run the complete payoff expectation circuit through a simulator
    # and sample the ancilla qubit
    counts_payoff_sim = job_payoff_sim.result().get_counts(qc)


    return counts_payoff_sim

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
    C = QuantumCircuit(qubits+1)

    C.x(qubits)
    C.h(qubits)
    C.cx(0, qubits)
    C.h(qubits)
    C.x(qubits)
    return C

def oracle_operator(qubits):
    C = QuantumCircuit(qubits + 1)
    C.z(qubits)

    return C

def load_Q_operator(qu, iterations, S0, sig, r, T, K):

    mu = (r - 0.5 * sig ** 2) * T + np.log(S0)  # Define all the parameters to be used in the computation
    mean = np.exp(
        mu + 0.5 * T * sig ** 2)  # Set the relevant zone of study and create the mapping between qubit and option price, and
    # generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
    S = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qu)
    ln = log_normal(S, mu, sig * np.sqrt(T))
    lognormal_parameters = rw_parameters(qu,
                                         ln)  # Solve for the parameters needed to create the target lognormal distribution
    prob_loading = rw_circuit(qu,
                              lognormal_parameters)  # Build the probaility loading circuit with the adjusted parameters
    prob_loading_inv = rw_circuit_inv(qu,
                              lognormal_parameters)
    payoff = payoff_circuit(qu, K, S)
    payoff_inv = payoff_circuit_inv(qu, K, S)

    diffusion = diffusion_operator(qu)
    oracle = oracle_operator(qu)

    qc = prob_loading + payoff
    qc_Q = oracle + payoff_inv + prob_loading_inv + diffusion + prob_loading + payoff
    for i in range(iterations):
        qc += qc_Q
    qc += measure_payoff(qu)

    return qc # Es posible que estas dos funciones puedan unirse

def run_Q_operator(qu, qc, shots, basis_gates, noise_model):

    job_payoff_sim = execute(qc, dev, shots=shots, basis_gates=basis_gates,
                             noise_model=noise_model)  # Run the complete payoff expectation circuit through a simulator
    # and sample the ancilla qubit
    counts_payoff_sim = job_payoff_sim.result().get_counts(qc)
    ones = 0
    zeroes = 0
    for key in counts_payoff_sim.keys():  # Post-selection
        unary = 0
        for i in range(1, qu + 1):
            unary += int(key[i])
        if unary == 1:
            if int(key[0]) == 0:
                zeroes += counts_payoff_sim.get(key)
            else:
                ones += counts_payoff_sim.get(key)

    prob_1 = ones / (ones + zeroes)

    return prob_1


def IQAE(payoff_e, alpha, shots, qu, data, basis_gates, noise_model):
    S0, sig, r, T, K = data
    i = 0
    j = 0
    num_j = 0
    a = 0
    up=True
    theta_l, theta_h = 0, np.pi / 2
    N = int(np.ceil(np.pi * 0.25 / payoff_e))
    c = (1 - alpha) / N

    while theta_h - theta_l < 2 * payoff_e:
        i += 1
        j_, up = find_next_k(j, theta_l, theta_h, up)
        qc = load_Q_operator(qu, j_, S0, sig, r, T, K)
        a_ = run_Q_operator(qu, qc, shots, basis_gates, noise_model)
        if j == j_:
            a = (num_j * a + a_) / (num_j + 1)
            num_j += 1
        else:
            a = a_
            num_j = 0

        a_min, a_max = chernoff(a, c)

        J = 4 * j + 2

        J_theta_max = np.arccos(1 - 2 * a_max)
        J_theta_min = np.arccos(1 - 2 * a_min)
        if not up:
            J_theta_max, J_theta_min = - J_theta_min, - J_theta_max

        theta_l = 1 / J * (np.remainder(np.floor(J * theta_l), 2 * np.pi) + J_theta_min / J)
        theta_h = 1 / J * (np.remainder(np.floor(J * theta_h), 2 * np.pi) + J_theta_max / J)

        #add max number of iterations (N)

    a_l, a_h = np.sin(theta_l) ** 2, np.sin(theta_h) ** 2

    return a_l, a_h







