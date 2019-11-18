import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, execute, BasicAer

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

def unary_quantum_sim(qu, S0, sig, r, T):
    backend_sim = BasicAer.get_backend('qasm_simulator') # Get qasm backend for the simulation of the circuit
    mu = (r - 0.5 * sig ** 2) * T + np.log(S0) # Define all the parameters to be used in the computation
    mean = np.exp(mu + 0.5 * T * sig ** 2) # Set the relevant zone of study and create the mapping between qubit and option price, and
                                            #generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
    S = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qu)
    ln = log_normal(S, mu, sig, T)
    lognormal_parameters = rw_parameters(qu, ln) # Solve for the parameters needed to create the target lognormal distribution
    prob_loading = rw_circuit(qu, lognormal_parameters) # Build the probaility loading circuit with the adjusted parameters
    circ_prob = prob_loading + measure_probability(qu) #Circuit to test the precision of the probability loading algorithm
    shots = 10000
    job_sim = execute(circ_prob, backend_sim, shots=shots)
    counts_sim = job_sim.result().get_counts(circ_prob)

    prob_sim = extract_probability(qu, counts_sim, shots)

    return (S, ln, prob_sim), (mu, mean, variance)


def unary_benchmark_sim(qu, S0, sig, r, T):
    (S, ln, prob_sim), (mu, mean, variance) = unary_quantum_sim(qu, S0, sig, r, T)

    '''
    Plot only the desired probabilities
    
    '''
    Sp = np.linspace(max(mean-3*np.sqrt(variance),0), mean+3*np.sqrt(variance), 10000) #Generate a the exact target distribution to benchmark against quantum results
    lnp = log_normal(Sp, mu, sig, T)
    width = (S[1] - S[0]) / 1.2
    
    fig, ax = plt.subplots()
    ax.bar(S, prob_sim, width, label='Quantum', alpha=0.8)
    ax.scatter(S, ln, s=500, color='C1', label='Exact', marker='_')
    ax.plot(Sp, lnp* (S[1]- S[0]) / (Sp[1] - Sp[0]), 'C1')
    plt.ylabel('Probability')
    plt.xlabel('Option price')
    plt.title('Option price distribution for {} qubits - unary'.format(qu))
    ax.legend()
    fig.tight_layout()
    fig.savefig('qubits_{}_prob_2.png'.format(qu))


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
        C.cu3(angle, 0, 0, i, qubits)                        #targeting the ancilla qubit
    return C

def measure_payoff(qubits):
    C = QuantumCircuit(qubits+1, 1)
    #C.barrier(range(qubits+1))
    C.measure(qubits, 0)
    return C
'''
def measure_payoff(qubits):
    
    C = QuantumCircuit(qubits+1, qubits+1)
    C.measure(range(qubits+1), range(qubits+1))
    return C
'''

'''
def payoff_quantum_sim(qu, S0, sig, r, T, K, noise_objects):
    backend_sim = Aer.get_backend('qasm_simulator')  # Get qasm backend for the simulation of the circuit
    mu = (r - 0.5 * sig ** 2) * T + np.log(S0)  # Define all the parameters to be used in the computation
    samples = 10000
    mean = np.exp(mu + 0.5 * T * sig ** 2)  # Set the relevant zone of study and create the mapping between qubit and option price, and
                                # generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
    S = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qu)
    ln = log_normal(S, mu, sig, T)
    lognormal_parameters = rw_parameters(qu, ln)  # Solve for the parameters needed to create the target lognormal distribution
    prob_loading = rw_circuit(qu, lognormal_parameters)  # Build the probaility loading circuit with the adjusted parameters
    circ_payoff = prob_loading + payoff_circuit(qu, K, S) + measure_payoff(qu)
    noise_model, basis_gates, error = noise_objects
    job_payoff_sim = execute(circ_payoff, backend_sim, shots=samples, basis_gates=basis_gates, noise_model=noise_model) #Run the complete payoff expectation circuit through a simulator
                                                                        #and sample the ancilla qubit
    #result_payoff_sim = job_payoff_sim.result()
    #counts_payoff_sim = result_payoff_sim.get_counts(circ_payoff)
    counts_payoff_sim = job_payoff_sim.result().get_counts(circ_payoff)
    ones=0
    zeroes=0
    #keys = counts_payoff_sim.keys()
    for key in counts_payoff_sim.keys():
        unary = 0
        for i in range(1,qu+1):
            unary+=int(key[i])
        if unary==1:
            if int(key[0])==0:
                zeroes+=counts_payoff_sim.get(key)
            else:
                ones+=counts_payoff_sim.get(key)
            
    qu_payoff_sim = ones * (S[qu - 1]-K) / (ones+zeroes)
    #print(ones+zeroes)

    return qu_payoff_sim

'''
def payoff_quantum_sim(qu, S0, sig, r, T, K):
    backend_sim = BasicAer.get_backend('qasm_simulator')  # Get qasm backend for the simulation of the circuit
    mu = (r - 0.5 * sig ** 2) * T + np.log(S0)  # Define all the parameters to be used in the computation
    samples = 10000
    mean = np.exp(mu + 0.5 * T * sig ** 2)  # Set the relevant zone of study and create the mapping between qubit and option price, and
                                # generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
    S = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qu)
    ln = log_normal(S, mu, sig, T)
    lognormal_parameters = rw_parameters(qu, ln)  # Solve for the parameters needed to create the target lognormal distribution
    prob_loading = rw_circuit(qu, lognormal_parameters)  # Build the probaility loading circuit with the adjusted parameters
    circ_payoff = prob_loading + payoff_circuit(qu, K, S) + measure_payoff(qu)
    job_payoff_sim = execute(circ_payoff, backend_sim, shots=samples) #Run the complete payoff expectation circuit through a simulator
                                                                        #and sample the ancilla qubit
    result_payoff_sim = job_payoff_sim.result()
    counts_payoff_sim = result_payoff_sim.get_counts(circ_payoff)
    qu_payoff_sim = counts_payoff_sim.get('1') * (S[qu - 1]-K) / samples

    return qu_payoff_sim


def classical_payoff(S0, sig, r, T, K):
    mu = (r - 0.5 * sig ** 2) * T + np.log(S0)  # Define all the parameters to be used in the computation
    samples = 10000
    mean = np.exp(mu + 0.5 * T * sig ** 2)  # Set the relevant zone of study and create the mapping between qubit and option price, and
    # generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
    Sp = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), samples)
    lnp = log_normal(Sp, mu, sig, T)
    cl_payoff = 0
    for i in range(len(Sp)):
        if K < Sp[i]:
            cl_payoff += lnp[i] * (Sp[i] - K)

    return cl_payoff


def unary_qu_cl(qu, S0, sig, r, T, K, cl_payoff):

    qu_payoff_sim = payoff_quantum_sim(qu, S0, sig, r, T, K)
    error = np.abs(100 * (cl_payoff - qu_payoff_sim) / cl_payoff)

    return qu_payoff_sim, error

