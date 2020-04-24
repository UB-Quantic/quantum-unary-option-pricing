import numpy as np
import binary as bin
import unary as un

from aux_functions import *
from noise_mapping import *
from time import time
import os
import matplotlib.pyplot as plt


def name_folder_data(data):
    string = 'results/S0(%s)_sig(%s)_r(%s)_T(%s)_K(%s)' % data
    return string
class errors:
    def __init__(self, data, max_gate_error, steps):
        self.data = data
        self.S0, self.sig, self.r, self.T, self.K = self.data
        self.cl_payoff = classical_payoff(self.S0, self.sig, self.r, self.T, self.K)
        # CNOT error is 2 * gate_error
        # Measurement error is 10 * gate_error
        self.max_gate_error = max_gate_error
        self.steps = steps
        self.error_steps = np.linspace(0, max_gate_error, steps)
        self.list_errors = ['bitflip', 'bitflip_m', 'phaseflip', 'phaseflip', 'bitphaseflip', 'bitphaseflip',
                            'depolarizing', 'depolarizing_m', 'measurement']
        try:
            os.makedirs(name_folder_data(self.data))
        except:
            pass


    def compute_save_errors_binary(self, bins, error_name, repeats, measure_error=False):
        qubits = int(np.log2(bins))
        results = np.zeros((len(self.error_steps), repeats))
        if error_name == 'bitflip':
            noise = noise_model_bit
            if measure_error:
                error_name += '_m'
        elif error_name == 'phaseflip':
            noise = noise_model_phase
            if measure_error:
                error_name += '_m'
        elif error_name == 'bitphaseflip':
            noise = noise_model_bitphase
            if measure_error:
                error_name += '_m'
        elif error_name == 'depolarizing':
            noise = noise_model_depolarizing
            if measure_error:
                error_name += '_m'
        elif error_name == 'measurement':
            noise = noise_model_measurement
        else:
            raise NameError('Error not indexed')

        for i, error in enumerate(self.error_steps):
            print('\n')
            print(i)
            noise_model = noise(error, measure=measure_error)
            basis_gates = noise_model.basis_gates
            c, k, high, low, qc, dev, shots = bin.load_payoff_quantum_sim(qubits, self.S0, self.sig, self.r, self.T, self.K)
            for r in range(repeats):
                qu_payoff_sim = bin.run_payoff_quantum_sim(qubits, c, k, high, low, qc, dev, shots, basis_gates, noise_model)
                diff = bin.diff_qu_cl(qu_payoff_sim, self.cl_payoff)
                results[i, r] = diff
        try:
            os.makedirs(name_folder_data(self.data) + '/%s_bins/binary/'%bins)
        except:
            pass
        np.savetxt(name_folder_data(self.data) + '/%s_bins/binary/'%bins + error_name + '_gate(%s)_steps(%s)_repeats(%s).npz'%(self.max_gate_error, self.steps, repeats), results)


    def compute_save_errors_unary(self, bins, error_name, repeats, measure_error=False):
        qubits = bins
        results = np.zeros((len(self.error_steps), repeats))
        if error_name == 'bitflip':
            noise = noise_model_bit
            if measure_error:
                error_name += '_m'
        elif error_name == 'phaseflip':
            noise = noise_model_phase
            if measure_error:
                error_name += '_m'
        elif error_name == 'bitphaseflip':
            noise = noise_model_bitphase
            if measure_error:
                error_name += '_m'
        elif error_name == 'depolarizing':
            noise = noise_model_depolarizing
            if measure_error:
                error_name += '_m'
        elif error_name == 'measurement':
            noise = noise_model_measurement
        else:
            raise NameError('Error not indexed')

        for i, error in enumerate(self.error_steps):
            print('\n')
            print(i)
            noise_model = noise(error, measure=measure_error)
            basis_gates = noise_model.basis_gates
            qc, dev, shots, S = un.load_payoff_quantum_sim(qubits, self.S0, self.sig, self.r, self.T, self.K)
            for r in range(repeats):
                qu_payoff_sim = un.run_payoff_quantum_sim(qubits, qc, dev, shots, S, self.K, basis_gates, noise_model)
                diff = un.diff_qu_cl(qu_payoff_sim, self.cl_payoff)
                results[i, r] = diff
        try:
            os.makedirs(name_folder_data(self.data) + '/%s_bins/unary/' % bins)
        except:
            pass
        np.savetxt(name_folder_data(self.data) + '/%s_bins/unary/'%bins +
                   error_name + '_gate(%s)_steps(%s)_repeats(%s).npz'%(self.max_gate_error, self.steps, repeats), results)

    def paint(self, bins, error_name, repeats, bounds=0.15, measure_error=False):
        if measure_error:
            error_name += '_m'

        matrix_unary = np.loadtxt(name_folder_data(self.data) + '/%s_bins/unary/'%bins
                            + error_name + '_gate(%s)_steps(%s)_repeats(%s).npz'%(self.max_gate_error, self.steps, repeats))
        matrix_unary = np.sort(matrix_unary, axis=1)
        mins_unary = matrix_unary[:, int(bounds * (self.steps))]
        maxs_unary = matrix_unary[:, int((1 - bounds) * (self.steps))]
        means_unary = np.mean(matrix_unary, axis=1)

        matrix_binary = np.loadtxt(name_folder_data(self.data) + '/%s_bins/binary/' % bins
                                  + error_name + '_gate(%s)_steps(%s)_repeats(%s).npz' % (
                                  self.max_gate_error, self.steps, repeats))
        matrix_binary = np.sort(matrix_binary, axis=1)
        mins_binary = matrix_binary[:, int(bounds * (self.steps))]
        maxs_binary = matrix_binary[:, int((1 - bounds) * (self.steps))]
        means_binary = np.mean(matrix_binary, axis=1)

        fig, ax = plt.subplots()
        ax.scatter(self.error_steps, means_unary, s=20, color='C0', label='unary', marker='x')
        ax.scatter(self.error_steps, means_binary, s=20, color='C1', label='binary', marker='+')
        ax.fill_between(self.error_steps, maxs_unary, mins_unary, alpha=0.2, facecolor='C0')
        ax.fill_between(self.error_steps, maxs_binary, mins_binary, alpha=0.2, facecolor='C1')
        plt.ylabel('percentage off classical value (%)')
        plt.xlabel('single-qubit gate error (%)')
        # plt.title('Expected payoff with increasing bitflip error - qasm_simulator')
        ax.legend()
        fig.tight_layout()
        fig.savefig(name_folder_data(self.data) + '/%s_bins/' % bins
                                  + error_name + '_gate(%s)_steps(%s)_repeats(%s).pdf' % (
                                  self.max_gate_error, self.steps, repeats))


