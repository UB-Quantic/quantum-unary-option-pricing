import itertools
import numpy as np
from qiskit.providers.aer.noise.errors import  ReadoutError, pauli_error, thermal_relaxation_error
from qiskit.providers.aer.noise import NoiseModel
from qiskit import IBMQ


def unary_coupling(qubits):
    entangling1 = [[i, i + 1] for i in range(qubits - 1)]
    entangling2 = [[i + 1, i] for i in range(qubits - 1)]
    entangling3 = [[i, qubits] for i in range(qubits)]

    coupling_map_unary = entangling1 + entangling2 + entangling3
    return coupling_map_unary


def binary_coupling(qubits):
    entangling1 = list(itertools.combinations(list(range(3,-1, -1)), r=2))
    entangling1 = [list(e) for e in entangling1]
    entangling2 = []
    for i in range(qubits - 1):
        entangling2 += [[i + 1, qubits + i], [i + 1, qubits + i + 1], [qubits + i, qubits + i + 1]]
    entangling3 = []
    for i in range(qubits):
        entangling3 += [[i, 2*qubits], [i, 2*qubits + 1]]
    entangling4 = [[0, qubits], [2*qubits - 1, 2*qubits], [2*qubits, 2*qubits + 1]]

    coupling_map_binary = entangling1 + entangling2 + entangling3 + entangling4
    return coupling_map_binary

def melbourne_coupling():
    provider = IBMQ.get_provider(group='open')
    device = provider.get_backend('ibmq_16_melbourne')
    #properties = device.properties()
    coupling_map = device.configuration().coupling_map
    return coupling_map

def thermal_map(noise_thermal, qubits, unary=True):
    # Thermal error
    # T1 and T2 values for qubits 0-16
    T1s = np.random.normal(50e3, 10e3, qubits)  # Sampled from normal distribution mean 50 microsec
    T2s = np.random.normal(70e3, 10e3, qubits)  # Sampled from normal distribution mean 50 microsec

    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(qubits)])

    # Instruction times (in nanoseconds)
    time_u1 = 0  # virtual gate
    time_u2 = 50  # (single X90 pulse)
    time_u3 = 100  # (two X90 pulses)
    time_cx = 300
    time_reset = 1000  # 1 microsecond
    time_measure = 1000  # 1 microsecond

    # QuantumError objects
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                    for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                      for t1, t2 in zip(T1s, T2s)]
    errors_u1 = [thermal_relaxation_error(t1, t2, time_u1)
                 for t1, t2 in zip(T1s, T2s)]
    errors_u2 = [thermal_relaxation_error(t1, t2, time_u2)
                 for t1, t2 in zip(T1s, T2s)]
    errors_u3 = [thermal_relaxation_error(t1, t2, time_u3)
                 for t1, t2 in zip(T1s, T2s)]

    for j in range(qubits):
        noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
        noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
        noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
        noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
        noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])

    time_cx = 300
    if unary:
        coupling = unary_coupling(qubits)
    else:
        coupling = binary_coupling(qubits)
    TA = np.random.multivariate_normal([50e3, 70e3], [[10e3, 0],[0, 10e3]], len(coupling))  # Sampled from normal distribution mean 50 microsec
    TB = np.random.multivariate_normal([50e3, 70e3], [[10e3, 0],[0, 10e3]], len(coupling))

    errors_cx = [thermal_relaxation_error(ta[0], ta[1], time_cx).expand(
        thermal_relaxation_error(tb[0], tb[1], time_cx)) for ta, tb in zip(TA, TB)]
    for k in range(len(coupling)):
        noise_thermal.add_quantum_error(errors_cx[k], "cx", coupling[k])

    return noise_thermal


def noise_model_unary(qubits, t):
    noise_model = NoiseModel()#basis_gates=['id', 'u2', 'u3', 'cx'])
    flip_error = 0.001 * t
    phase_error = 0.002 * t
    cnot_error = 0.01 * t
    p1given0 = 0.1 * t
    p0given1=0.05 * t
    identity = pauli_error([('I', 1)])
    bit_flip = pauli_error([('X', flip_error), ('I', 1 - flip_error)])
    phase_flip = pauli_error([('Z', phase_error), ('I', 1 - phase_error)])
    bitphase_flip = bit_flip.compose(phase_flip)
    cnot_flip = pauli_error([('X', cnot_error), ('I', 1 - cnot_error)])
    cnot_phase = pauli_error([('Z', cnot_error), ('I', 1 - cnot_error)])
    cnot_error = cnot_flip.compose(cnot_phase)
    cnot_error = identity.tensor(cnot_error)
    coupling_map = unary_coupling(qubits)
    for c in coupling_map:
        noise_model.add_quantum_error(cnot_error, ['cx'], c, warnings=False)

    noise_model.add_all_qubit_quantum_error(bitphase_flip, ["u1", "u2", "u3"])

    measure_error = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
    noise_model.add_all_qubit_readout_error(measure_error, ["u1", "u2", "u3"])

    return noise_model

def noise_model_unary_measure(qubits, t):
    noise_model = NoiseModel()#basis_gates=['id', 'u2', 'u3', 'cx'])
    p1given0 = 0.001 * t
    p0given1 = 0.001 * t

    measure_error = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
    noise_model.add_all_qubit_readout_error(measure_error, ["u1", "u2", "u3"])

    return noise_model

def noise_model_unary_phaseflip(qubits, t):
    noise_model = NoiseModel()#basis_gates=['id', 'u2', 'u3', 'cx'])
    phase_error = 0.001 * t
    identity = pauli_error([('I', 1)])
    phase_flip = pauli_error([('Z', phase_error), ('I', 1 - phase_error)])
    cnot_phase = pauli_error([('Z', phase_error), ('I', 1 - phase_error)])
    cnot_error = identity.tensor(cnot_phase)
    coupling_map = unary_coupling(qubits)
    for c in coupling_map:
        noise_model.add_quantum_error(cnot_error, ['cx'], c, warnings=False)

    noise_model.add_all_qubit_quantum_error(phase_flip, ["u1", "u2", "u3"])

    return noise_model

def noise_model_unary_bitflip(qubits, t):
    noise_model = NoiseModel()#basis_gates=['id', 'u2', 'u3', 'cx'])
    flip_error = 0.001 * t
    identity = pauli_error([('I', 1)])
    bit_flip = pauli_error([('X', flip_error), ('I', 1 - flip_error)])
    cnot_flip = pauli_error([('X', flip_error), ('I', 1 - flip_error)])
    cnot_error = identity.tensor(cnot_flip)
    coupling_map = unary_coupling(qubits)
    for c in coupling_map:
        noise_model.add_quantum_error(cnot_error, ['cx'], c, warnings=False)

    noise_model.add_all_qubit_quantum_error(bit_flip, ["u1", "u2", "u3"])

    return noise_model

def noise_model_binary(qubits, t):
    noise_model = NoiseModel()  # basis_gates=['id', 'u2', 'u3', 'cx'])
    flip_error = 0.001 * t
    phase_error = 0.002 * t
    cnot_error = 0.01 * t
    p1given0 = 0.1 * t
    p0given1 = 0.05 * t
    identity = pauli_error([('I', 1)])
    bit_flip = pauli_error([('X', flip_error), ('I', 1 - flip_error)])
    phase_flip = pauli_error([('Z', phase_error), ('I', 1 - phase_error)])
    bitphase_flip = bit_flip.compose(phase_flip)
    cnot_flip = pauli_error([('X', cnot_error), ('I', 1 - cnot_error)])
    cnot_phase = pauli_error([('Z', cnot_error), ('I', 1 - cnot_error)])
    cnot_error = cnot_flip.compose(cnot_phase)
    cnot_error = identity.tensor(cnot_error)
    coupling_map = binary_coupling(qubits)
    for c in coupling_map:
        noise_model.add_quantum_error(cnot_error, ['cx'], c, warnings=False)

    noise_model.add_all_qubit_quantum_error(bitphase_flip, ["u1", "u2", "u3"])

    measure_error = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
    noise_model.add_all_qubit_readout_error(measure_error, ["u1", "u2", "u3"])

    return noise_model

def noise_model_binary_measure(qubits, t):
    noise_model = NoiseModel()#basis_gates=['id', 'u2', 'u3', 'cx'])
    p1given0 = 0.001 * t
    p0given1 = 0.001 * t

    measure_error = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
    noise_model.add_all_qubit_readout_error(measure_error, ["u1", "u2", "u3"])

    return noise_model

def noise_model_binary_phaseflip(qubits, t):
    noise_model = NoiseModel()#basis_gates=['id', 'u2', 'u3', 'cx'])
    phase_error = 0.001 * t
    identity = pauli_error([('I', 1)])
    phase_flip = pauli_error([('Z', phase_error), ('I', 1 - phase_error)])
    cnot_phase = pauli_error([('Z', phase_error), ('I', 1 - phase_error)])
    cnot_error = identity.tensor(cnot_phase)
    coupling_map = binary_coupling(qubits)
    for c in coupling_map:
        noise_model.add_quantum_error(cnot_error, ['cx'], c, warnings=False)

    noise_model.add_all_qubit_quantum_error(phase_flip, ["u1", "u2", "u3"])

    return noise_model

def noise_model_binary_bitflip(qubits, t):
    noise_model = NoiseModel()#basis_gates=['id', 'u2', 'u3', 'cx'])
    flip_error = 0.001 * t
    identity = pauli_error([('I', 1)])
    bit_flip = pauli_error([('X', flip_error), ('I', 1 - flip_error)])
    cnot_flip = pauli_error([('X', flip_error), ('I', 1 - flip_error)])
    cnot_error = identity.tensor(cnot_flip)
    coupling_map = binary_coupling(qubits)
    for c in coupling_map:
        noise_model.add_quantum_error(cnot_error, ['cx'], c, warnings=False)

    noise_model.add_all_qubit_quantum_error(bit_flip, ["u1", "u2", "u3"])

    return noise_model

