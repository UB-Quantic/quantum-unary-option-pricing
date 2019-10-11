from unary_representation.fit_gaussian import fit, S_, Sp_, _gauss
import numpy as np
import qcgpu
from QuantumState.QuantumState import QCircuit

# Variational circuit that simulates a quantum random walk of a particle
def _RWcircuit_gpu(qubits, parameters):
    if qubits % 2 == 0: raise ValueError('Try odd number of qubits')
    C = qcgpu.State(qubits + 1)
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
    C = QCircuit(qubits + 1)
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



def _PayoffCircuit(qubits, S0, sig, K, gpu):
    parameters = fit(qubits, S0, sig, gpu=gpu)
    S=S_(S0, sig, qubits)
    if gpu:
        C = _RWcircuit_gpu(qubits, parameters)
        for i in range(qubits):
            bK = i
            if K < S[i]:
                break
            # Control-RY rotations controled by states with higher value than strike
            # targeting the ancilla qubit
        for i in range(bK, qubits):
            angle = 2 * np.arcsin(np.sqrt((S[i] - K) / S[qubits - 1]))
            C.cu(i, qubits, angle, 0, 0)
        return C
    else:
        C = _RWcircuit_cpu(qubits, parameters)
        for i in range(qubits):
            bK = i
            if K < S[i]:
                break
            # Control-RY rotations controled by states with higher value than strike
            # targeting the ancilla qubit
        for i in range(bK, qubits):
            angle = 2 * np.arcsin(np.sqrt((S[i] - K) / S[qubits - 1]))
            C.CU3(i, qubits, [angle, 0, 0])
        return C


def Payoff(qubits, S0, sig, K, gpu):
    C = _PayoffCircuit(qubits, S0, sig, K, gpu)
    if gpu:
        psi = C.amplitudes()
    else:
        psi = C.psi
    CPayoff = 0
    Sp = Sp_(S0, sig, qubits)
    fp = _gauss(Sp, S0, sig)
    for i in range(2**qubits):
        if K<Sp[i]:
            CPayoff += fp[i]*(Sp[i]-K)
    #The quantum pay-off is computes adding the probabilities
    #of finding the ancilla qubit on the state 1
    S = S_(S0, sig, qubits)
    QPayoff = 0
    for i in range(2**qubits, 2**(qubits+1)):
        QPayoff += S[qubits-1]*(np.abs(psi[i])**2)
    return CPayoff, QPayoff
