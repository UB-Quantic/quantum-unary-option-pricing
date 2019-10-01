import matplotlib.pyplot as plt
import numpy as np
from math import pi
import qcgpu
from QuantumState.QuantumState import QCircuit
from scipy.optimize import minimize
import os

def _VCircuit_gpu(qubits, parameters, layers):
    if qubits%2==1: raise ValueError('Try even number of qubits')
    C = qcgpu.State(qubits)
    p = 0
    for layer in range(layers):
        for i in range(qubits):
            C.u(i, parameters[p], 0, 0)
            p+=1
        for i in range(0, qubits, 2):
            C.apply_controlled_gate(qcgpu.gate.z(),i, i+1)
        for i in range(qubits):
            C.u(i, parameters[p], 0, 0)
            p+=1
        C.apply_controlled_gate(qcgpu.gate.z(), 0, qubits-1)
        for i in range(1, qubits-1, 2):
            C.apply_controlled_gate(qcgpu.gate.z(), i, i+1)
    for i in range(qubits):
        C.u(i, parameters[p], 0, 0)
        p+=1
    return C


def _VCircuit_cpu(qubits, parameters, layers):
    if qubits % 2 == 1: raise ValueError('Try even number of qubits')

    C = QCircuit(qubits)
    p = 0
    for layer in range(layers):
        for i in range(qubits):
            C.Ry(i, parameters[p])
            p += 1
        for i in range(0, qubits, 2):
            C.Cz(i, i + 1)
        for i in range(qubits):
            C.Ry(i, parameters[p])
            p += 1
        C.Cz(0, qubits - 1)
        for i in range(1, qubits - 1, 2):
            C.Cz(i, i + 1)
    for i in range(qubits):
        C.Ry(i, parameters[p])
        p += 1
    return C


def _parameters(qubits, layers):
    num = 2 * qubits * (layers + 1)
    p = 2*pi*np.random.random((num,))
    return p


def _HO(x, p, psi, sig, mu):
    H = ((x-mu)*psi + 2*p*sig**2)**2
    return H


def _cost_function(parameters, qubits, layers, sig, S0, gpu):
    if gpu:
        circ = _VCircuit_gpu(qubits, parameters, layers)
        cost = 0
        psi = circ.amplitudes()

    else:
        circ = _VCircuit_cpu(qubits, parameters, layers)
        cost = 0
        psi = circ.psi

    x = S_(S0, sig, qubits)
    dx = x[1]-x[0]
    p = np.gradient(psi, dx)
    for i in range(2**qubits):
        cost+=_HO(x[i], p[i], psi[i], sig, S0)
    return cost


def _result(parameters, qubits, layers, gpu):
    if gpu:
        circ = _VCircuit_gpu(qubits, parameters, layers)
        psi = circ.amplitudes()
    else:
        circ = _VCircuit_cpu(qubits, parameters, layers)
        psi = circ.psi
    sol = np.abs(psi)**2
    return np.array(sol)


def fit(qubits, S0, sig, layers, to = 1e-5, meth='L-BFGS-B', gpu=False): #, maxi = 10000
    foldname = _name_folder(qubits, S0, sig, gpu, layers)
    if gpu:
        gpu_string = 'gpu'
    else:
        gpu_string = 'cpu'

    foldname2 = foldname.replace('binary_representation/data/', '')
    if foldname2[:foldname2.find('/',0)] in os.listdir('binary_representation/data') and \
            'parameters_{}.txt'.format(meth) in os.listdir(foldname) and \
            gpu_string in foldname:
        param = np.loadtxt(foldname + '/parameters_{}.txt'.format(meth))
        print('Already computed')

    else:
        print('Not yet computed')
        S = S_(S0, sig, qubits)
        #g = gauss(S, S0, sig)
        para = _parameters(qubits, layers)
        solution = minimize(_cost_function, para, args=(qubits, layers, sig, S0, gpu), method=meth, tol=to) #, options={'maxiter':maxi}
        param = solution.x
        _create_folder(_name_folder(qubits, S0, sig, gpu, layers))
        np.savetxt(_name_folder(qubits, S0, sig, gpu, layers) + '/parameters_{}.txt'.format(meth), param)
    return param


def paint_fit(qu, S0, sig, layers, K, method, gpu=False):
    S = S_(S0, sig, qu)
    width = 6 * sig / 2**qu / 1.2
    fp = _gauss(S, S0, sig)
    optimal_parameters = fit(qu, S0, sig, layers, gpu=gpu, meth=method)
    r = _result(optimal_parameters, qu, layers, gpu=gpu)
    fig, ax = plt.subplots()
    # rects1 = ax.bar(S-width/2, g, width, label='Exact')
    ax.bar(S, r, width, label='Quantum')
    ax.plot(S, max(r) * fp / max(fp), 'C1', label='Exact')

    ax.vlines(K, 0, max(r), linestyles='dashed', label='K = {}'.format(K))
    plt.ylabel('Probability')
    plt.xlabel('Option price')
    plt.title('Option price unary distribution for {} qubits.'.format(qu))
    ax.legend()

    fig.tight_layout()
    _create_folder(_name_folder(qu, S0, sig, gpu, layers).replace('data', 'hist'))
    fig.savefig(_name_folder(qu, S0, sig, gpu, layers).replace('data', 'hist') + '/{}.png'.format(method))


def _gauss(x, mu, sigma):
    g = (1/(np.sqrt(2*pi*sigma**2)))*np.exp(-((x-mu)**2)/(2*sigma**2))
    dx = x[1]-x[0]
    f = g*dx/(np.sum(g * dx))
    return f


def _create_folder(directory):
    """
    Auxiliar function for creating directories with name directory

    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def _name_folder(qu, S0, sig, gpu, layers):
    name = 'binary_representation/data/qu:{}_S0:{}_sigma:{}'.format(qu, S0, sig)
    if gpu:
        name += '/gpu'
    else:
        name += '/cpu'

    name += '/layers:{}'.format(layers)

    return name


def S_(S0, sig, qu):
    return np.linspace(S0-3*sig, S0+3*sig, int(2**qu))