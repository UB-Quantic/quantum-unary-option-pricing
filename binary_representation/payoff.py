import qcgpu
from QuantumState.QuantumState import QCircuit
from binary_representation.fit_gaussian import fit
import numpy as np


def _VCircuit_gpu(qubits, parameters, layers):
    if qubits%2==1: raise ValueError('Try even number of qubits')
    C = qcgpu.State(2*qubits + 2)
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

    C = QCircuit(2*qubits + 2)
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


def _qOR_cpu(QC, qubits):
    QC.X(qubits[0])
    QC.X(qubits[1])
    QC.X(qubits[2])
    QC.multi_Cx(qubits[:2], qubits[2])
    QC.X(qubits[0])
    QC.X(qubits[1])


def _qOR_gpu(QC, qubits):
    QC.x(qubits[0])
    QC.x(qubits[1])
    QC.x(qubits[2])
    QC.toffoli(qubits[0], qubits[1], qubits[2])
    QC.x(qubits[0])
    QC.x(qubits[1])


def _comparator_cpu(QC, n, K, X_max, X_min):
    if QC.num_qubits < 2*n + 1:
        raise ValueError('Not enough qubits')
    if K < X_min or K > X_max:
        raise ValueError('Not good Strike')
    k = int(np.ceil(2**n * (K - X_min) / (X_max - X_min)))
    t = 2**n - k
    t_bin = np.binary_repr(t, n)
    if t_bin[-1] == '1':
        QC.Cx(0, n)

    for i in range(n - 1):
        if t_bin[-2 - i] == '0':
            QC.multi_Cx([1 + i, n + i], n + 1 + i)
        elif t_bin[-2 - i] == '1':
            _qOR_cpu(QC, [1 + i, n + i, n + 1 + i])

    QC.Cx(2*n - 1, 2*n)
    return k

def _comparator_gpu(QC, n, K, X_max, X_min):
    if QC.num_qubits < 2*n + 1:
        raise ValueError('Not enough qubits')
    if K < X_min or K > X_max:
        raise ValueError('Not good Strike')
    k = int(np.ceil(2**n * (K - X_min) / (X_max - X_min)))
    t = 2**n - k
    t_bin = np.binary_repr(t, n)
    if t_bin[-1] == '1':
        QC.cx(0, n)

    for i in range(n - 1):
        if t_bin[-2 - i] == '0':
            QC.toffoli(1 + i, n + i, n + 1 + i)
        elif t_bin[-2 - i] == '1':
            _qOR_gpu(QC, [1 + i, n + i, n + 1 + i])

    QC.cx(2*n - 1, 2*n)
    return k

def _rotations_cpu(QC, n, g0, c, k):
    if QC.num_qubits < 2*n + 2:
        raise ValueError('Not enough qubits')
    QC.Ry(2*n + 1, 2 * g0)
    QC.CU3(2*n, 2*n + 1, [4 * c * k / (k - 2**n + 1), 0, 0])
    for _ in range(n):
        QC.multi_Ry([_, 2 * n], 2*n + 1, 2**(2 + _) * c/ (2**n - 1 - k))

def _rotations_gpu(QC, n, g0, c, k):
    if QC.num_qubits < 2*n + 2:
        raise ValueError('Not enough qubits')
    QC.u(2*n + 1, 2 * g0, 0 , 0)
    QC.cu3(2*n, 2*n + 1, [4 * c * k / (k - 2**n + 1), 0, 0])
    for _ in range(n):
        _CCRy(QC, [_, 2 * n], 2 * n + 1, 2 ** (2 + _) * c / (2 ** n - 1 - k))

def _CCRy(QC, ctrls, targ, theta):
    QC.cu3(ctrls[1], targ, theta / 2, 0, 0)
    QC.cx(ctrls[0], ctrls[1])
    QC.cu3(ctrls[1], targ, -theta / 2, 0, 0)
    QC.cx(ctrls[0], ctrls[1])
    QC.cu3(ctrls[0], targ, theta / 2, 0, 0)

def _expected_payoff(psi, c, X_max, K):
    prob = np.sum(np.abs(psi)**2)

    return prob, (prob - 0.5 + c)*(X_max - K) / 2 / c


def quantum_payoff(qubits, S0, sig, K, layers, method, gpu, error=0.1):
    parameters = fit(qubits, S0, sig, layers, meth=method, gpu=gpu)
    X_min = S0 - 3 * sig
    X_max = S0 + 3 * sig
    c = np.sqrt(2) * error ** (1 / (2 * qubits + 2))
    g0 = 0.25 * np.pi - c
    if gpu:
        C = _VCircuit_gpu(qubits, parameters, layers)
        probs = np.abs(C.amplitudes()[:2**qubits])**2
        k = _comparator_gpu(C, qubits, K, X_max, X_min)
        _rotations_gpu(C, qubits, g0, c, k)
        psi = C.amplitudes()
    else:
        C = _VCircuit_cpu(qubits, parameters, layers)  #Hay un fallo en algún punto de este código porque no da lo que tiene que dar
        probs = np.abs(C.psi[:2**qubits]) ** 2   #Hay un fallo en algún punto de este código porque no da lo que tiene que dar
        k = _comparator_cpu(C, qubits, K, X_max, X_min)   #Hay un fallo en algún punto de este código porque no da lo que tiene que dar
        _rotations_cpu(C, qubits, g0, c, k)  #Hay un fallo en algún punto de este código porque no da lo que tiene que dar
        psi = C.psi

    prob, payoff = _expected_payoff(psi, c, X_max, K)

    return payoff, probs

def classical_payoff(probs, S0, sig, K):
    X = np.linspace(S0 - 3 * sig, S0 + 3 * sig, len(probs))
    payoff = 0
    for x, p in zip(X, probs):
        if x > K:
            payoff += (x - K) * p

    return payoff


