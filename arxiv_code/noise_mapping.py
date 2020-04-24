from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, pauli_error, thermal_relaxation_error
from numpy import array

"""
This file
"""

def noise_model_measure(error, measure=True):
    noise_model = NoiseModel()#basis_gates=['id', 'u2', 'u3', 'cx'])
    p_measure = 10 * error
    measure_error = pauli_error([('X',p_measure), ('I', 1 - p_measure)])
    noise_model.add_all_qubit_readout_error(measure_error, "measure")

    return noise_model

def noise_model_phase(error, measure=True, thermal=True):
    noise_model = NoiseModel()#basis_gates=['id', 'u2', 'u3', 'cx'])
    phase_error = error
    cz_error = 2 * error
    phase_flip = pauli_error([('Z', phase_error), ('I', 1 - phase_error)])
    cz_error = pauli_error([('Z', cz_error), ('I', 1 - cz_error)])
    cz_error = cz_error.tensor(cz_error)
    noise_model.add_all_qubit_quantum_error(cz_error, ['cx'], warnings=False)
    noise_model.add_all_qubit_quantum_error(phase_flip, ["u1", "u2", "u3"])

    if measure:
        measure_error = 10 * error
        measure_error = array([[1 - measure_error, measure_error], [measure_error, 1 - measure_error]])
        noise_model.add_all_qubit_readout_error(measure_error)

    if thermal:
        thermal = thermal_relaxation_error(1.5, 1.2, error)
        cthermal = thermal_relaxation_error(1.5, 1.2, 2 * error)
        cthermal = cthermal.tensor(cthermal)
        noise_model.add_all_qubit_quantum_error(cthermal, ['cx'], warnings=False)
        noise_model.add_all_qubit_quantum_error(thermal, ["u1", "u2", "u3"], warnings=False)

    return noise_model

def noise_model_bit(error, measure=True, thermal=True):
    noise_model = NoiseModel()#basis_gates=['id', 'u2', 'u3', 'cx'])
    flip_error = error
    cnot_error = 2 * error
    bit_flip = pauli_error([('X', flip_error), ('I', 1 - flip_error)])
    cnot_flip = pauli_error([('X', cnot_error), ('I', 1 - cnot_error)])
    cnot_error = cnot_flip.tensor(cnot_flip)
    noise_model.add_all_qubit_quantum_error(cnot_error, ['cx'], warnings=False)
    noise_model.add_all_qubit_quantum_error(bit_flip, ["u1", "u2", "u3"])

    if measure:
        measure_error = 10 * error
        measure_error = array([[1 - measure_error, measure_error], [measure_error, 1 - measure_error]])
        noise_model.add_all_qubit_readout_error(measure_error)

    if thermal:
        thermal = thermal_relaxation_error(1.5, 1.2, error)
        cthermal = thermal_relaxation_error(1.5, 1.2, 2 * error)
        cthermal = cthermal.tensor(cthermal)
        noise_model.add_all_qubit_quantum_error(cthermal, ['cx'], warnings=False)
        noise_model.add_all_qubit_quantum_error(thermal, ["u1", "u2", "u3"], warnings=False)

    return noise_model


def noise_model_bitphase(error, measure=False, thermal=True):
    noise_model = NoiseModel()  # basis_gates=['id', 'u2', 'u3', 'cx'])
    cnot_error = 2 * error
    bit_flip = pauli_error([('X', error), ('I', 1 - error)])
    phase_flip = pauli_error([('Z', error), ('I', 1 - error)])
    bitphase_flip = bit_flip.compose(phase_flip)
    cnot_flip = pauli_error([('X', cnot_error), ('I', 1 - cnot_error)])
    cnot_phase = pauli_error([('Z', cnot_error), ('I', 1 - cnot_error)])
    cnot_error = cnot_flip.compose(cnot_phase)
    cnot_error = cnot_error.tensor(cnot_error)
    noise_model.add_all_qubit_quantum_error(cnot_error, ['cx'], warnings=False)
    noise_model.add_all_qubit_quantum_error(bitphase_flip, ["u1", "u2", "u3"])

    if measure:
        measure_error = 10 * error
        measure_error = array([[1 - measure_error, measure_error], [measure_error, 1 - measure_error]])
        noise_model.add_all_qubit_readout_error(measure_error)

    if thermal:
        thermal = thermal_relaxation_error(1.5, 1.2, error)
        cthermal = thermal_relaxation_error(1.5, 1.2, 2 * error)
        cthermal = cthermal.tensor(cthermal)
        noise_model.add_all_qubit_quantum_error(cthermal, ['cx'], warnings=False)
        noise_model.add_all_qubit_quantum_error(thermal, ["u1", "u2", "u3"], warnings=False)

    return noise_model

def noise_model_depolarizing(error, measure=False, thermal=True):
    noise_model = NoiseModel()  # basis_gates=['id', 'u2', 'u3', 'cx'])
    depolarizing = depolarizing_error(error, 1)
    cdepol_error = depolarizing_error(2 * error, 2)
    noise_model.add_all_qubit_quantum_error(cdepol_error, ['cx'], warnings=False)
    noise_model.add_all_qubit_quantum_error(depolarizing, ["u1", "u2", "u3"])

    if measure:
        measure_error = 10 * error
        measure_error = array([[1 - measure_error, measure_error], [measure_error, 1 - measure_error]])
        noise_model.add_all_qubit_readout_error(measure_error)

    if thermal:
        thermal = thermal_relaxation_error(1.5, 1.2, error)
        cthermal = thermal_relaxation_error(1.5, 1.2, 2 * error)
        cthermal = cthermal.tensor(cthermal)
        noise_model.add_all_qubit_quantum_error(cthermal, ['cx'], warnings=False)
        noise_model.add_all_qubit_quantum_error(thermal, ["u1", "u2", "u3"], warnings=False)

    return noise_model

def noise_model_thermal(error, measure=False):
    noise_model = NoiseModel()  # basis_gates=['id', 'u2', 'u3', 'cx'])
    thermal = thermal_relaxation_error(1.5, 1.2, error)
    cthermal = thermal_relaxation_error(1.5, 1.2, 2 * error)
    cthermal = cthermal.tensor(cthermal)
    noise_model.add_all_qubit_quantum_error(cthermal, ['cx'], warnings=False)
    noise_model.add_all_qubit_quantum_error(thermal, ["u1", "u2", "u3"])

    if measure:
        measure_error = 10 * error
        measure_error = array([[1 - measure_error, measure_error], [measure_error, 1 - measure_error]])
        noise_model.add_all_qubit_readout_error(measure_error)

    return noise_model

