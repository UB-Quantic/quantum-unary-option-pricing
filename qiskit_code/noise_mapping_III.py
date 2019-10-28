from qiskit.providers.aer.noise.errors import pauli_error
from qiskit.providers.aer.noise import NoiseModel

def noise_model_total(qubits, t):
    noise_model = NoiseModel()#basis_gates=['id', 'u2', 'u3', 'cx'])
    flip_error = 0.00005 * t
    phase_error = 0.00005 * t
    cnot_error = 0.0001 * t
    #p_measure = 0.0005 * t
    bit_flip = pauli_error([('X', flip_error), ('I', 1 - flip_error)])
    phase_flip = pauli_error([('Z', phase_error), ('I', 1 - phase_error)])
    bitphase_flip = bit_flip.compose(phase_flip)
    cnot_flip = pauli_error([('X', cnot_error), ('I', 1 - cnot_error)])
    cnot_phase = pauli_error([('Z', cnot_error), ('I', 1 - cnot_error)])
    cnot_error = cnot_flip.compose(cnot_phase)
    cnot_error = cnot_error.tensor(cnot_error)
    noise_model.add_all_qubit_quantum_error(cnot_error, ['cx'], warnings=False)
    noise_model.add_all_qubit_quantum_error(bitphase_flip, ["u1", "u2", "u3"])
    #measure_error = pauli_error([('X',p_measure), ('I', 1 - p_measure)])
    #noise_model.add_all_qubit_quantum_error(measure_error, "measure")

    return noise_model

def noise_model_measure(qubits, t):
    noise_model = NoiseModel()#basis_gates=['id', 'u2', 'u3', 'cx'])
    p_measure = 0.0005 * t
    measure_error = pauli_error([('X',p_measure), ('I', 1 - p_measure)])
    noise_model.add_all_qubit_quantum_error(measure_error, "measure")

    return noise_model

def noise_model_phaseflip(qubits, t):
    noise_model = NoiseModel()#basis_gates=['id', 'u2', 'u3', 'cx'])
    phase_error = 0.00005 * t
    cnot_error = 0.0001 * t
    phase_flip = pauli_error([('Z', phase_error), ('I', 1 - phase_error)])
    cnot_phase = pauli_error([('Z', cnot_error), ('I', 1 - cnot_error)])
    cnot_error = cnot_phase.tensor(cnot_phase)
    noise_model.add_all_qubit_quantum_error(cnot_error, ['cx'], warnings=False)
    noise_model.add_all_qubit_quantum_error(phase_flip, ["u1", "u2", "u3"])

    return noise_model

def noise_model_bitflip(qubits, t):
    noise_model = NoiseModel()#basis_gates=['id', 'u2', 'u3', 'cx'])
    flip_error = 0.00005 * t
    cnot_error = 0.0001 * t
    bit_flip = pauli_error([('X', flip_error), ('I', 1 - flip_error)])
    cnot_flip = pauli_error([('X', cnot_error), ('I', 1 - cnot_error)])
    cnot_error = cnot_flip.tensor(cnot_flip)
    noise_model.add_all_qubit_quantum_error(cnot_error, ['cx'], warnings=False)
    noise_model.add_all_qubit_quantum_error(bit_flip, ["u1", "u2", "u3"])

    return noise_model


