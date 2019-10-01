import matplotlib.pyplot as plt
import numpy as np
from math import pi
import qcgpu
from QuantumState.QuantumState import QCircuit
from scipy.optimize import minimize
import os

#Creation and normalization of an array with gaussian probabilities
def _gauss(x, mu, sigma):
    dx = x[1]-x[0]
    g = (1/(np.sqrt(2*pi*sigma**2)))*np.exp(-((x-mu)**2)/(2*sigma**2))
    f = g*dx/(np.sum(g * dx))
    return f


# Variational circuit that simulates a quantum random walk of a particle
def _RWcircuit_gpu(qubits, parameters):
    if qubits % 2 == 0: raise ValueError('Try odd number of qubits')
    C = qcgpu.State(qubits)
    mid = int((qubits - 1) / 2)
    C.x(mid)
    p = 0
    for layer in range(mid):
        # Population distribution to open states
        C.cu(mid - layer, mid - layer - 1, parameters[p], 0, 0)
        p += 1
        C.cx(mid - layer - 1, mid - layer)
        C.cu(mid + layer, mid + layer + 1, parameters[p], 0, 0)
        p += 1
        C.cx(mid + layer + 1, mid + layer)
        for i in reversed(range(layer)):
            # Probabilistic SWAPs for already occupied states
            C.cx(mid - i, mid - i - 1)
            C.cu(mid - i - 1, mid - i, parameters[p], 0, 0)
            p += 1
            C.cx(mid - i, mid - i - 1)

            C.cx(mid + i, mid + i + 1)
            C.cu(mid + i + 1, mid + i, parameters[p], 0, 0)
            p += 1
            C.cx(mid + i, mid + i + 1)
    return C


def _RWcircuit_cpu(qubits, parameters):
    C = QCircuit(qubits)
    mid = int((qubits - 1) / 2)
    C.X(mid)
    p = 0
    for layer in range(mid):
        # Population distribution to open states
        C.CU3(mid - layer, mid - layer - 1, [parameters[p], 0, 0])
        p += 1
        C.Cx(mid - layer - 1, mid - layer)
        C.CU3(mid + layer, mid + layer + 1, [parameters[p], 0, 0])
        p += 1
        C.Cx(mid + layer + 1, mid + layer)
        for i in reversed(range(layer)):
            # Probabilistic SWAPs for already occupied states
            C.Cx(mid - i, mid - i - 1)
            C.CU3(mid - i - 1, mid - i, [parameters[p], 0, 0])
            p += 1
            C.Cx(mid - i, mid - i - 1)

            C.Cx(mid + i, mid + i + 1)
            C.CU3(mid + i + 1, mid + i, [parameters[p], 0, 0])
            p += 1
            C.Cx(mid + i, mid + i + 1)
    return C


#Parameter creation for the first step of the variational algorithm
def _parameters(qubits):
    a = int((qubits**2 - 2 * qubits + 1)/2)
    p = 2*pi*np.random.random((a,))
    return p


#Cost funtion
def _cost_function(par, qubits, ga, gpu):
    if gpu:
        circ = _RWcircuit_gpu(qubits, par)
        cost = 0
        psi = circ.amplitudes()
        #prob = circ.probabilities()
        for i in range(qubits):
            cost+=((ga[i]) - (np.abs(psi[int(2**i)]))**2)**2
            #cost+=((ga[i]) - (prob[int(2**i)]))**2
        return cost

    else:
        circ = _RWcircuit_cpu(qubits, par)
        cost = 0
        psi = circ.psi
        for i in range(qubits):
            cost += ((ga[i]) - (np.abs(psi[int(2 ** i)])) ** 2) ** 2
        return cost


#Probability distribution of the circuit's final state
def _result(par, qubits, gpu):
    if gpu:
        circ = _RWcircuit_gpu(qubits, par)
        psi = circ.amplitudes()
        sol = []
        for i in range(qubits):
            sol.append(np.abs(psi[int(2**i)])**2)
            #sol.append(prob[int(2**i)])
        return np.array(sol)

    else:
        circ = _RWcircuit_cpu(qubits, par)
        psi = circ.psi
        sol = []
        for i in range(qubits):
            sol.append(np.abs(psi[int(2**i)])**2)
        return np.array(sol)


def _name_folder(qu, S0, sig, gpu):
    name = 'unary_representation/data/qu:{}_S0:{}_sigma:{}'.format(qu, S0, sig)
    if gpu:
        name += '_gpu'

    return name


def fit(qu, S0, sig, method='SLSQP', gpu=False):
    foldname = _name_folder(qu, S0, sig, gpu)
    if foldname.replace('unary_representation/data/', '') in os.listdir('unary_representation/data') and 'parameters_{}.txt'.format(method) in os.listdir(foldname):
            param = np.loadtxt(foldname + '/parameters_{}.txt'.format(method))

    else:
        S = S_(S0, sig, qu)
        g = _gauss(S, S0, sig)
        para = _parameters(qu)
        solution = minimize(_cost_function, para, args=(qu, g, gpu), method=method, options={'disp': True})
        param = solution.x
        _create_folder(_name_folder(qu, S0, sig, gpu))
        np.savetxt(_name_folder(qu, S0, sig, gpu) + '/parameters_{}.txt'.format(method), param)

    return param

def paint_fit(qu, S0, sig, K, method='SLSQP', gpu=False):
    S = np.linspace(S0 - 3 * sig, S0 + 3 * sig, qu)
    width = 6 * sig / qu / 1.2
    Sp = Sp_(S0, sig, qu)
    fp = _gauss(Sp, S0, sig)
    optimal_parameters = fit(qu, S0, sig, gpu=gpu, method=method)
    r = _result(optimal_parameters, qu, gpu)
    fig, ax = plt.subplots()
    # rects1 = ax.bar(S-width/2, g, width, label='Exact')
    ax.bar(S, r, width, label='Quantum')
    ax.plot(Sp, max(r) * fp / max(fp), 'C1', label='Exact')

    ax.vlines(K, 0, max(r), linestyles='dashed', label='K = {}'.format(K))
    plt.ylabel('Probability')
    plt.xlabel('Option price')
    plt.title('Option price unary distribution for {} qubits.'.format(qu))
    ax.legend()

    fig.tight_layout()
    fig.savefig( _name_folder(qu, S0, sig, gpu) + '/hist_{}.png'.format(method))


def _create_folder(directory):
    """
    Auxiliar function for creating directories with name directory

    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
        
        
def S_(S0, sig, qu):
    return np.linspace(S0-3*sig, S0+3*sig, qu)


def Sp_(S0, sig, qu):
    return np.linspace(S0-3*sig, S0+3*sig, 2**qu)