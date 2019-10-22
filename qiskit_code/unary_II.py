import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, execute, Aer
from qiskit import IBMQ, transpile
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_circuit_layout
from noise_mapping_II import unary_coupling

def log_normal(x, mu, sig, T):
    """
    Lognormal probability distribution function normalized for representation in finite intervals
    """
    dx = x[1]-x[0]
    log_norm = 1 / (x * sig * np.sqrt(2 * np.pi * T)) * np.exp(- np.power(np.log(x) - mu, 2.) / (2 * np.power(sig, 2.) * T))
    f = log_norm*dx/(np.sum(log_norm * dx))
    return f



def rw_circuit(qubits, parameters):
    """
    Circuit that emulates the probability sharing between neighbours seen usually
    in Brownian motion and stochastic systems
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


def rw_parameters(qubits, pdf):
    """
    Solving for the exact angles for the random walk circuit that enables the loading of
    a desired probability distribution function
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
    """
    C = QuantumCircuit(qubits+1, qubits)
    C.barrier(qubits)
    C.measure(range(qubits),range(qubits)) #No measure on the ancilla qubit is necessary
    return C


def extract_probability(qubits, counts, samples):
    """
    From the retuned sampling, extract only probabilies of unary states
    """
    form = '{0:0%sb}' % str(qubits) # qubits?
    prob = []
    for i in range(qubits):
        prob.append(counts.get(form.format(2**i), 0)/samples)
    return prob


def extract_probability_all_samples(qubits, counts, samples):
    """
    From the retuned sampling, extract only probabilies of unary states
    """
    form = '{0:0%sb}' % str(qubits) # qubits?
    prob = []
    for i in range(2**qubits):
        prob.append(counts.get(form.format(i), 0)/samples)
    return prob

def extract_probability_dev(qubits, counts, samples):
    form = '{0:0%sb}' % str(qubits) # qubits
    prob = []
    stop = []
    p=0
    for i in range(qubits):
        stop.append(form.format(2**i))
    for i in range(2**qubits):
        p+=counts.get(form.format(i), 0)/samples
        if form.format(i) in stop:
            prob.append(p)
            p=0
    return prob


def unary_quantum_sim(qu, S0, sig, r, T, noise_objects, all_samples):
    backend_sim = Aer.get_backend('qasm_simulator') # Get qasm backend for the simulation of the circuit
    mu = (r - 0.5 * sig ** 2) * T + np.log(S0) # Define all the parameters to be used in the computation
    mean = np.exp(mu + 0.5 * T * sig ** 2) # Set the relevant zone of study and create the mapping between qubit and option price, and
                                            #generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
    S = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qu)
    ln = log_normal(S, mu, sig, T)
    lognormal_parameters = rw_parameters(qu, ln) # Solve for the parameters needed to create the target lognormal distribution
    prob_loading = rw_circuit(qu, lognormal_parameters) # Build the probaility loading circuit with the adjusted parameters
    circ_prob = prob_loading + measure_probability(qu) #Circuit to test the precision of the probability loading algorithm
    shots = 2**qu
    noise_model, coupling_map, basis_gates, error = noise_objects
    job_sim = execute(circ_prob, backend_sim, shots=shots, basis_gates=basis_gates, noise_model=noise_model,coupling_map=coupling_map, optimization_level=3)
    result_sim = job_sim.result()  # Run the test circuit through the simulator and sample the results in order to get the estimated probabilities
    counts_sim = result_sim.get_counts(circ_prob)

    if all_samples:
        prob_sim = extract_probability_all_samples(qu, counts_sim, shots)
    else:
        prob_sim = extract_probability(qu, counts_sim, shots)
    print(sum(prob_sim))

    return (S, ln, prob_sim), (mu, mean, variance)


def unary_benchmark_sim(qu, S0, sig, r, T, noise_objects, err):
    (S, ln, prob_sim), (mu, mean, variance) = unary_quantum_sim(qu, S0, sig, r, T, noise_objects, all_samples)


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
    plt.title('Option price distribution for {} qubits - {} error: {} - qasm_simulator'.format(qu, err, (0.001*noise_objects[3])))
    ax.legend()
    ax.set(xscale='log')
    fig.tight_layout()
    fig.savefig('unary/qubits:{}_S0:{}_sig:{}_r:{}_T:{}_error:{}_{}_all:.png'.format(qu, S0, sig, r, T, noise_objects[3], err))
      
    '''
    Plot both only de desired and all probabilities
    
    '''
    Sp = np.linspace(max(mean-3*np.sqrt(variance),0), mean+3*np.sqrt(variance), 2**qu) #Generate a the exact target distribution to benchmark against quantum results
    lnp = log_normal(Sp, mu, sig, T)
    width = (S[1] - S[0]) / 1.2
    
    fig, ax = plt.subplots()
    ax.bar(S, prob_sim, width, label='Quantum', alpha=0.8)
    ax.scatter(S, ln, s=500, color='C1', label='Exact', marker='_')
    ax.plot(Sp, lnp* (S[1]- S[0]) / (Sp[1] - Sp[0]), 'C1')
    plt.ylabel('Probability')
    plt.xlabel('Option price')
    plt.title('Option price distribution for {} qubits - {} error: {} - qasm_simulator'.format(qu, err, (0.001*noise_objects[3])))
    ax.legend()
    fig.tight_layout()
    fig.savefig('unary/qubits:{}_S0:{}_sig:{}_r:{}_T:{}_error:{}_{}.png'.format(qu, S0, sig, r, T, noise_objects[3], err))


def payoff_circuit(qubits, K, S):
    '''
    Circuit that codifies the expected payoff of the option into the probability of a
    single ancilla qubit

    '''
    C = QuantumCircuit(qubits+1)
    for i in range(qubits): #Determine the first qubit's price that
        qK = i              #surpasses the strike price
        if K<S[i]:
            break
    C.barrier(range(qubits+1))
    for i in range(qK, qubits):                              #Control-RY rotations controled by states
        angle = 2 * np.arcsin(np.sqrt((S[i]-K)/(S[qubits-1]-K))) #with higher value than the strike
        C.cu3(((-1)**i)*angle, 0, 0, i, qubits)                        #targeting the ancilla qubit
    return C


def measure_payoff(qubits):
    '''
    Circuit that measures the ancilla qubit to sample the expected payoff of the option
    '''
    C = QuantumCircuit(qubits+1, 1)
    C.barrier(range(qubits+1))
    C.measure(qubits, 0)
    return C


def payoff_quantum_sim(qu, S0, sig, r, T, K, noise_objects):
    backend_sim = Aer.get_backend('qasm_simulator')  # Get qasm backend for the simulation of the circuit
    mu = (r - 0.5 * sig ** 2) * T + np.log(S0)  # Define all the parameters to be used in the computation
    samples = 2 ** qu
    mean = np.exp(mu + 0.5 * T * sig ** 2)  # Set the relevant zone of study and create the mapping between qubit and option price, and
                                # generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
    S = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qu)
    ln = log_normal(S, mu, sig, T)
    lognormal_parameters = rw_parameters(qu, ln)  # Solve for the parameters needed to create the target lognormal distribution
    prob_loading = rw_circuit(qu, lognormal_parameters)  # Build the probaility loading circuit with the adjusted parameters
    circ_payoff = prob_loading + payoff_circuit(qu, K, S) + measure_payoff(qu)
    noise_model, coupling_map, basis_gates, error = noise_objects
    job_payoff_sim = execute(circ_payoff, backend_sim, shots=samples, basis_gates=basis_gates, noise_model=noise_model,
                      coupling_map=coupling_map, optimization_level=3) #Run the complete payoff expectation circuit through a simulator
                                                                        #and sample the ancilla qubit
    result_payoff_sim = job_payoff_sim.result()
    counts_payoff_sim = result_payoff_sim.get_counts(circ_payoff)
    qu_payoff_sim = counts_payoff_sim.get('1') * (S[qu - 1]-K) / samples

    return qu_payoff_sim


def classical_payoff(qu, S0, sig, r, T, K):
    mu = (r - 0.5 * sig ** 2) * T + np.log(S0)  # Define all the parameters to be used in the computation
    samples = 2 ** qu
    mean = np.exp(
        mu + 0.5 * T * sig ** 2)  # Set the relevant zone of study and create the mapping between qubit and option price, and
    # generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
    Sp = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qu)
    lnp = log_normal(Sp, mu, sig, T)
    cl_payoff = 0
    for i in range(len(Sp)):
        if K < Sp[i]:
            cl_payoff += lnp[i] * (Sp[i] - K)

    return cl_payoff


def unary_qu_cl(qu, S0, sig, r, T, K, noise_objects, err):
    samples=2**qu
    qu_payoff_sim = payoff_quantum_sim(qu, S0, sig, r, T, K, noise_objects)
    cl_payoff = classical_payoff(qu, S0, sig, r, T, K)
    error = np.abs(100 * (cl_payoff - qu_payoff_sim) * 2 / (cl_payoff + qu_payoff_sim))
    print('With precision {}'.format(samples))
    print('Classical Payoff: {}'.format(cl_payoff))
    print('')
    print('With {} qubits and {} samples with {} error: {}'.format(qu, samples, err, (0.001*noise_objects[3])))
    print('Quantum Payoff: {}'.format(qu_payoff_sim))
    print('')
    print('Percentage off: {}%'.format(error))

    return qu_payoff_sim, cl_payoff, error

