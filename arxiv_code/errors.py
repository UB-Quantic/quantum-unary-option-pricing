import numpy as np
import binary as bin
import unary as un

from aux_functions import *
from noise_mapping import *
from time import time
import os
import matplotlib.pyplot as plt
from matplotlib import colors
from qiskit import execute
from qiskit import Aer

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.aqua.components.uncertainty_models import LogNormalDistribution
"""
This file creates the Python class defining all quantities to perform 
"""


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
                            'depolarizing', 'depolarizing_m', 'thermal', 'thermal_m', 'measurement']
        try:
            os.makedirs(name_folder_data(self.data))
        except:
            pass

    def select_error(self, error_name, measure_error=False, thermal_error=False):
        if 'bitflip' in error_name:
            noise = noise_model_bit
            if measure_error:
                error_name += '_m'
            if thermal_error:
                error_name += '_t'
        elif 'phaseflip' in error_name:
            noise = noise_model_phase
            if measure_error:
                error_name += '_m'
            if thermal_error:
                error_name += '_t'
        elif 'bitphaseflip' in error_name:
            noise = noise_model_bitphase
            if measure_error:
                error_name += '_m'
            if thermal_error:
                error_name += '_t'
        elif 'depolarizing' in error_name:
            noise = noise_model_depolarizing
            if measure_error:
                error_name += '_m'
            if thermal_error:
                error_name += '_t'
        elif error_name == 'thermal':
            noise = noise_model_thermal
            if measure_error:
                error_name += '_m'
        elif error_name == 'measurement':
            noise = noise_model_measure
        else:
            raise NameError('Error not indexed')

        return noise

    def change_name(self, error_name, measure_error, thermal_error):
        if measure_error and '_m' not in error_name:
            error_name += '_m'
        if thermal_error and '_t' not in error_name:
            error_name += '_t'

        return error_name


    def compute_save_errors_binary(self, bins, error_name, repeats, measure_error=False, thermal_error=False, shots=10000):
        qubits = int(np.log2(bins))
        results = np.zeros((len(self.error_steps), repeats))
        noise = self.select_error(error_name, measure_error=measure_error, thermal_error=thermal_error)
        error_name = self.change_name(error_name, measure_error, thermal_error)

        for i, error in enumerate(self.error_steps):
            print(i)
            noise_model = noise(error, measure=measure_error, thermal=thermal_error)
            basis_gates = noise_model.basis_gates
            c, k, high, low, qc = bin.load_payoff_quantum_sim(qubits, self.S0, self.sig, self.r, self.T, self.K)
            for r in range(repeats):
                qu_payoff_sim = bin.run_payoff_quantum_sim(qubits, c, k, high, low, qc, shots, basis_gates, noise_model)
                diff = bin.diff_qu_cl(qu_payoff_sim, self.cl_payoff)
                results[i, r] = diff
        try:
            os.makedirs(name_folder_data(self.data) + '/%s_bins/binary/'%bins)
        except:
            pass
        np.savetxt(name_folder_data(self.data) + '/%s_bins/binary/'%bins + error_name + '_gate(%s)_steps(%s)_repeats(%s).npz'%(self.max_gate_error, self.steps, repeats), results)


    def compute_save_errors_unary(self, bins, error_name, repeats, measure_error=False, thermal_error=False, shots=10000):
        qubits = bins
        results = np.zeros((len(self.error_steps), repeats))
        noise = self.select_error(error_name, measure_error=measure_error, thermal_error=thermal_error)
        error_name = self.change_name(error_name, measure_error, thermal_error)

        for i, error in enumerate(self.error_steps):
            print(i)
            noise_model = noise(error, measure=measure_error, thermal=thermal_error)
            basis_gates = noise_model.basis_gates
            qc, S = un.load_payoff_quantum_sim(qubits, self.S0, self.sig, self.r, self.T, self.K)
            for r in range(repeats):
                qu_payoff_sim = un.run_payoff_quantum_sim(qubits, qc, shots, S, self.K, basis_gates, noise_model)
                diff = un.diff_qu_cl(qu_payoff_sim, self.cl_payoff)
                results[i, r] = diff
        try:
            os.makedirs(name_folder_data(self.data) + '/%s_bins/unary/' % bins)
        except:
            pass
        np.savetxt(name_folder_data(self.data) + '/%s_bins/unary/'%bins +
                   error_name + '_gate(%s)_steps(%s)_repeats(%s).npz'%(self.max_gate_error, self.steps, repeats), results)

    def paint_errors(self, bins, error_name, repeats, bounds=0.15, measure_error=False, thermal_error=False):
        error_name = self.change_name(error_name, measure_error, thermal_error)

        matrix_unary = np.loadtxt(name_folder_data(self.data) + '/%s_bins/unary/'%bins
                            + error_name + '_gate(%s)_steps(%s)_repeats(%s).npz'%(self.max_gate_error, self.steps, repeats))
        matrix_unary = np.sort(matrix_unary, axis=1)
        mins_unary = matrix_unary[:, int(bounds * (repeats))]
        maxs_unary = matrix_unary[:, int(-(bounds) * (repeats)-1)]
        means_unary = np.mean(matrix_unary, axis=1)
        matrix_binary = np.loadtxt(name_folder_data(self.data) + '/%s_bins/binary/' % bins
                                  + error_name + '_gate(%s)_steps(%s)_repeats(%s).npz' % (
                                  self.max_gate_error, self.steps, repeats))
        matrix_binary = np.sort(matrix_binary, axis=1)
        mins_binary = matrix_binary[:, int(bounds * (repeats))]
        maxs_binary = matrix_binary[:, int(-(bounds) * (repeats)-1)]
        means_binary = np.mean(matrix_binary, axis=1)

        fig, ax = plt.subplots()
        ax.scatter(100 * self.error_steps, means_unary, s=20, color='C0', label='unary', marker='x')
        ax.scatter(100 * self.error_steps, means_binary, s=20, color='C1', label='binary', marker='+')
        ax.fill_between(100 * self.error_steps, maxs_unary, mins_unary, alpha=0.2, facecolor='C0')
        ax.fill_between(100 * self.error_steps, maxs_binary, mins_binary, alpha=0.2, facecolor='C1')
        plt.ylabel('percentage off classical value (%)')
        plt.xlabel('single-qubit gate error (%)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(name_folder_data(self.data) + '/%s_bins/' % bins
                                  + error_name + '_gate(%s)_steps(%s)_repeats(%s).pdf' % (
                                  self.max_gate_error, self.steps, repeats))


    def compute_save_outcomes_binary(self, bins, error_name, error_value, repeats, measure_error=False, thermal_error=False, shots=10000):
        error_name = self.change_name(error_name, measure_error, thermal_error)
        qubits = int(np.log2(bins))
        try:
            os.makedirs(name_folder_data(self.data) + '/%s_bins/binary/probs' % bins)
        except:
            pass
        noise = self.select_error(error_name, measure_error=measure_error, thermal_error=thermal_error)
        noise_model = noise(error_value, measure=measure_error, thermal=thermal_error)
        basis_gates = noise_model.basis_gates
        qc, (values, pdf), (mu, mean, variance) = bin.load_quantum_sim(qubits, self.S0, self.sig, self.r, self.T)
        probs = np.zeros((len(values), repeats))
        for r in range(repeats):
            probs[:, r] = bin.run_quantum_sim(qubits, qc, shots, basis_gates, noise_model)

        np.savetxt(name_folder_data(self.data) + '/%s_bins/binary/probs/'%bins +
                   error_name + '_gate(%s)_repeats(%s).npz'%(error_value, repeats), probs)

        return probs

    def compute_save_outcomes_unary(self, bins, error_name, error_value, repeats, measure_error=False, thermal_error=False, shots=10000):
        error_name = self.change_name(error_name, measure_error, thermal_error)
        qubits = bins
        try:
            os.makedirs(name_folder_data(self.data) + '/%s_bins/unary/probs' % bins)
        except:
            pass
        noise = self.select_error(error_name, measure_error=measure_error, thermal_error=thermal_error)
        noise_model = noise(error_value, measure=measure_error, thermal=thermal_error)
        basis_gates = noise_model.basis_gates
        qc, (values, pdf), (mu, mean, variance) = un.load_quantum_sim(qubits, self.S0, self.sig, self.r, self.T)
        probs = np.zeros((len(values), repeats))
        for r in range(repeats):
            res = un.run_quantum_sim(qubits, qc, shots, basis_gates, noise_model)
            probs[:, r] = [res[2**i] for i in range(qubits)]
            probs[:, r] /= np.sum(probs[:, r])

        np.savetxt(name_folder_data(self.data) + '/%s_bins/unary/probs/'%bins +
                    error_name + '_gate(%s)_repeats(%s).npz'%(error_value, repeats), probs)

        return probs

    def paint_outcomes(self, bins, error_name, error_value, repeats, bounds=0.15, measure_error=False, thermal_error=False):
        error_name = self.change_name(error_name, measure_error, thermal_error)
        print(error_name)
        matrix_unary = np.loadtxt(name_folder_data(self.data) + '/%s_bins/unary/probs/'%bins +
                            error_name + '_gate(%s)_repeats(%s).npz'%(error_value, repeats))
        matrix_unary = np.sort(matrix_unary, axis=1)
        mins_unary = matrix_unary[:, int(bounds * (repeats))]
        maxs_unary = matrix_unary[:, int(-(bounds) * (repeats)-1)]
        means_unary = np.mean(matrix_unary, axis=1)
        matrix_binary = np.loadtxt(name_folder_data(self.data) + '/%s_bins/binary/probs/'%bins +
                            error_name + '_gate(%s)_repeats(%s).npz'%(error_value, repeats))
        matrix_binary = np.sort(matrix_binary, axis=1)
        mins_binary = matrix_binary[:, int(bounds * (repeats))]
        maxs_binary = matrix_binary[:, int(-(bounds) * (repeats)-1)]
        means_binary = np.mean(matrix_binary, axis=1)

        qc, (values, pdf), (mu, mean, variance) = un.load_quantum_sim(bins, self.S0, self.sig, self.r, self.T)

        width = (values[1] - values[0]) / 1.3
        exact_values = np.linspace(np.min(values), np.max(values), bins * 100)
        mu = (self.r - 0.5 * self.sig ** 2) * self.T + np.log(self.S0)
        exact_pdf = log_normal(exact_values, mu, self.sig * np.sqrt(self.T))
        exact_pdf = exact_pdf * pdf[0] / exact_pdf[0]
        fig, ax = plt.subplots()

        ax.bar(values + width / 4, maxs_binary, width / 2, alpha=0.3, color='C1')
        ax.bar(values + width / 4, mins_binary, width / 2, alpha=.75, color='C1', label='binary')
        ax.bar(values - width / 4, maxs_unary, width / 2, alpha=0.3, color='C0')
        ax.bar(values - width / 4, mins_unary, width / 2, alpha=.75, color='C0', label='unary')
        ax.plot(exact_values, exact_pdf, color='black', label='PDF')
        ax.scatter(values - width / 4, means_unary, s=20, color='C0', marker='x', zorder=10)
        ax.scatter(values + width / 4, means_binary, s=20, color='C1', marker='x', zorder=10)
        ax.scatter(values, pdf, s=1250, color='black', marker='_', zorder=9)
        plt.ylabel('Probability')
        plt.xlabel('Option price')
        # plt.title('Option price distribution for {} qubits - {} error: {} - qasm_simulator'.format(qu, err, (noise_objects[2])))
        ax.legend()
        fig.tight_layout()
        fig.savefig(name_folder_data(self.data) + '/%s_bins/' % bins
                    + error_name + '_gate(%s)_repeats(%s)_probs.pdf' % (
                        error_value, repeats))

    def compute_save_KL_binary(self, bins, error_name, repeats, measure_error=False, thermal_error=False, shots=10000):
        error_name = self.change_name(error_name, measure_error, thermal_error)

        divergences = np.zeros((len(self.error_steps), repeats))
        for i, error in enumerate(self.error_steps):
            try:
                probs = np.loadtxt(name_folder_data(self.data) + '/%s_bins/binary/probs/' % bins +
                           error_name + '_gate(%s)_repeats(%s).npz' % (error, repeats))

            except:
                probs = self.compute_save_outcomes_binary(bins, error_name, error, repeats,
                                                          measure_error=measure_error, thermal_error=thermal_error, shots=shots)

            qc, (values, pdf), (mu, mean, variance) = un.load_quantum_sim(bins, self.S0, self.sig, self.r, self.T)
            for j in range(probs.shape[1]):
                divergences[i, j] = KL(probs[:, j], pdf)

            np.savetxt(name_folder_data(self.data) + '/%s_bins/binary/probs/' % bins +
                error_name + '_gate(%s)_repeats(%s)_div.npz' % (self.max_gate_error, repeats), divergences)

    def compute_save_KL_unary(self, bins, error_name, repeats, measure_error=False, thermal_error=False, shots=10000):
        error_name = self.change_name(error_name, measure_error, thermal_error)

        divergences = np.zeros((len(self.error_steps), repeats))
        for i, error in enumerate(self.error_steps):
            try:

                probs = np.loadtxt(name_folder_data(self.data) + '/%s_bins/unary/probs/' % bins +
                                   error_name + '_gate(%s)_repeats(%s).npz' % (error, repeats))

            except:
                probs = self.compute_save_outcomes_unary(bins, error_name, error, repeats,
                                                          measure_error=measure_error, thermal_error=thermal_error, shots=shots)

            qc, (values, pdf), (mu, mean, variance) = un.load_quantum_sim(bins, self.S0, self.sig, self.r, self.T)
            for j in range(probs.shape[1]):
                divergences[i, j] = KL(probs[:, j], pdf)

            np.savetxt(name_folder_data(self.data) + '/%s_bins/unary/probs/' % bins +
                   error_name + '_gate(%s)_repeats(%s)_div.npz' % (self.max_gate_error, repeats), divergences)

    def paint_divergences(self, bins, error_name, repeats, bounds=0.15, measure_error=False, thermal_error=False):
        error_name = self.change_name(error_name, measure_error, thermal_error)

        matrix_unary = np.loadtxt(name_folder_data(self.data) + '/%s_bins/unary/probs/'%bins
                            + error_name + '_gate(%s)_repeats(%s)_div.npz' % (self.max_gate_error, repeats))
        matrix_binary = np.loadtxt(name_folder_data(self.data) + '/%s_bins/binary/probs/' % bins
                                         + error_name + '_gate(%s)_repeats(%s)_div.npz' % (self.max_gate_error, repeats))
        matrix_unary = np.sort(matrix_unary, axis=1)
        mins_unary = matrix_unary[:, int(bounds * (repeats))]
        maxs_unary = matrix_unary[:, int(-(bounds) * (repeats)-1)]
        means_unary = np.mean(matrix_unary, axis=1)
        matrix_binary = np.sort(matrix_binary, axis=1)
        mins_binary = matrix_binary[:, int(bounds * (repeats))]
        maxs_binary = matrix_binary[:, int(-(bounds) * (repeats)-1)]
        means_binary = np.mean(matrix_binary, axis=1)

        fig, ax = plt.subplots()
        ax.scatter(100 * self.error_steps, means_unary, s=20, color='C0', label='unary', marker='x')
        ax.scatter(100 * self.error_steps, means_binary, s=20, color='C1', label='binary', marker='+')
        ax.fill_between(100 * self.error_steps, maxs_unary, mins_unary, alpha=0.2, facecolor='C0')
        ax.fill_between(100 * self.error_steps, maxs_binary, mins_binary, alpha=0.2, facecolor='C1')
        plt.ylabel('KL Divergence')
        plt.xlabel('single-qubit gate error (%)')
        plt.yscale('log')
        ax.legend()
        fig.tight_layout()
        fig.savefig(name_folder_data(self.data) + '/%s_bins/' % bins
                                  + error_name + '_gate(%s)_steps(%s)_repeats(%s)_div.pdf' % (
                                  self.max_gate_error, self.steps, repeats))

    def unary_iqae(self, bins, error_name, error_value, payoff_e=0.001, alpha=0.1, measure_error=False, thermal_error=False, shots=100):
        qu = bins
        noise = self.select_error(error_name, measure_error=measure_error, thermal_error=thermal_error)
        noise_model = noise(error_value, measure=measure_error, thermal=thermal_error)
        basis_gates = noise_model.basis_gates
        a_l, a_h = un.IQAE(payoff_e, alpha, shots, qu, self.data, basis_gates, noise_model)
        qc, S = un.load_payoff_quantum_sim(qu, self.S0, self.sig, self.r, self.T, self.K)
        payoff_l = un.get_payoff_from_prob(a_l * np.pi, qu, S, self.K)
        payoff_h = un.get_payoff_from_prob(a_h * np.pi, qu, S, self.K)

        print(payoff_l, payoff_h, self.cl_payoff)

    def test_binary_Q(self, qu): # Ready to delete
        print(qu)
        S0, sig, r, T, K = self.S0, self.sig, self.r, self.T, self.K
        iterations = 1
        iterations = int(iterations)
        mu = ((r - 0.5 * sig ** 2) * T + np.log(S0))  # parameters for the log_normal distribution
        sigma = sig * np.sqrt(T)
        mean = np.exp(mu + sigma ** 2 / 2)
        variance = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
        stddev = np.sqrt(variance)

        # lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
        low = np.maximum(0, mean - 3 * stddev)
        high = mean + 3 * stddev
        # S = np.linspace(low, high, samples)

        # construct circuit factory for uncertainty model
        uncertainty_model = LogNormalDistribution(qu, mu=mu, sigma=sigma, low=low, high=high)

        values = uncertainty_model.values
        pdf = uncertainty_model.probabilities

        qr = QuantumRegister(2 * qu + 2)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        uncertainty_model.build(qc, qr)
        bin.payoff_circuit(qc, qu, K, high, low, measure=False)
        bin.oracle_operator(qc, qu)
        bin.payoff_circuit_inv(qc, qu, K, high, low, measure=False)
        uncertainty_model.build_inverse(qc, qr)
        job_payoff_sim = execute(qc, Aer.get_backend('statevector_simulator'))# Run the complete payoff expectation circuit through a simulator
        statevector_1 = job_payoff_sim.result().data()['statevector']
        bin.diffusion_operator(qc, qu)
        job_payoff_sim = execute(qc, Aer.get_backend(
            'statevector_simulator'))  # Run the complete payoff expectation circuit through a simulator
        statevector_2 = job_payoff_sim.result().data()['statevector']
        print('algorithm')
        for i, (s1, s2) in enumerate(zip(statevector_1, statevector_2)):
            print(np.binary_repr(i, 2*qu+2), s1, s2)

    def test_unary_mlae(self, bins, error_name, error_value, bin_error, measure_error=False, thermal_error=False, shots=100, mode='eis'): # Ready to delete
        mode = mode.lower()
        if 'lis' in mode:
            M = np.ceil(1 / bin_error)
            m_s = np.arange(0, M, 1)
        elif 'eis' in mode:
            M = np.ceil(2 * np.log2(1 / bin_error) - 1)
            print(M)
            m_s = np.hstack((0, 2 ** np.arange(0, M)))
            print(m_s)
        qu = bins
        noise = self.select_error(error_name, measure_error=measure_error, thermal_error=thermal_error)
        noise_model = noise(error_value, measure=measure_error, thermal=thermal_error)
        basis_gates = noise_model.basis_gates
        qc = un.load_Q_operator(qu, 0, self.S0, self.sig, self.r, self.T, self.K)
        ones, zeroes = un.run_Q_operator(qu, qc, shots, basis_gates, noise_model)

        a = ones / (ones + zeroes)
        theta = np.arcsin(np.sqrt(a))
        i = 0
        print(np.sin((2 * i + 1) * theta) ** 2, ones / (ones + zeroes))
        for i in range(1, 10):
            qc = un.load_Q_operator(qu, i, self.S0, self.sig, self.r, self.T, self.K)
            ones, zeroes = un.run_Q_operator(qu, qc, shots, basis_gates, noise_model)
            print(np.sin((2 * i + 1) * theta) ** 2, ones / (ones + zeroes))

    def test_binary_mlae(self, bins, error_name, error_value, bin_error, measure_error=False, thermal_error=False, shots=100, mode='eis'): # Ready to delete
        mode = mode.lower()
        if 'lis' in mode:
            M = np.ceil(1 / bin_error)
            m_s = np.arange(0, M, 1)
        elif 'eis' in mode:
            M = np.ceil(2 * np.log2(1 / bin_error) - 1)
            print(M)
            m_s = np.hstack((0, 2 ** np.arange(0, M)))
            print(m_s)
        qu = int(np.log2(bins))
        noise = self.select_error(error_name, measure_error=measure_error, thermal_error=thermal_error)
        noise_model = noise(error_value, measure=measure_error, thermal=thermal_error)
        basis_gates = noise_model.basis_gates
        qc, (k, c, high, low) = bin.load_Q_operator(qu, 0, self.S0, self.sig, self.r, self.T, self.K)
        print(qc.num_qubits)
        ones, zeroes = bin.run_Q_operator(qc, shots, basis_gates, noise_model)

        a = ones / (ones + zeroes)
        theta = np.arcsin(np.sqrt(a))
        i = 0
        print(np.sin((2 * i + 1) * theta) ** 2, ones / (ones + zeroes))
        for i in range(1, 10):
            qc, r = bin.load_Q_operator(qu, i, self.S0, self.sig, self.r, self.T, self.K)

            ones, zeroes = bin.run_Q_operator(qc, shots, basis_gates, noise_model)
            print(np.sin((2 * i + 1) * theta) ** 2, ones / (ones + zeroes))



    def unary_mlae(self, bins, error_name, error_value, M, measure_error=False, thermal_error=False, shots=100):
        m_s = np.arange(0, M, 1)
        qu = bins
        noise = self.select_error(error_name, measure_error=measure_error, thermal_error=thermal_error)
        noise_model = noise(error_value, measure=measure_error, thermal=thermal_error)
        basis_gates = noise_model.basis_gates

        a_s, error_s = un.MLAE(qu, self.data, m_s, shots, basis_gates, noise_model)

        qc, S = un.load_payoff_quantum_sim(qu, self.S0, self.sig, self.r, self.T, self.K)
        payoff_qu = un.get_payoff_from_prob(a_s, qu, S, self.K)
        print('last')
        print(bin.diff_qu_cl(payoff_qu, self.cl_payoff), 100 * np.abs(error_s / self.cl_payoff))

    def binary_mlae(self, bins, error_name, error_value, M=4, measure_error=False, thermal_error=False, shots=100):
        m_s = np.arange(0, M+1, 1)
        qu = int(np.log2(bins))
        noise = self.select_error(error_name, measure_error=measure_error, thermal_error=thermal_error)
        noise_model = noise(error_value, measure=measure_error, thermal=thermal_error)
        basis_gates = noise_model.basis_gates

        a_s, error_s, (k, c, high, low) = bin.MLAE(qu, self.data, m_s, shots, basis_gates, noise_model)

        #qc, S = bin.load_payoff_quantum_sim(qu, self.S0, self.sig, self.r, self.T, self.K)
        payoff_qu = bin.get_payoff_from_prob(a_s, qu, c, k, high, low)
        error_qu = bin.get_payoff_error_from_prob_error(error_s, qu, c, k, high, low)
        # algo no funciona bien en este error_qu, ¿pero qué?
        print(bin.diff_qu_cl(payoff_qu, self.cl_payoff), 100 * np.abs(error_qu / self.cl_payoff))

    def compute_save_errors_binary_amplitude_estimation(self, bins, error_name, repeats, M=4, measure_error=False, thermal_error=False,
                                   shots=500, u=0, error=0.05):
        qubits = int(np.log2(bins))
        error_name = self.change_name(error_name, measure_error, thermal_error)
        noise = self.select_error(error_name, measure_error=measure_error, thermal_error=thermal_error)
        m_s = np.arange(0, M + 1, 1)
        (values, pdf) = bin.get_pdf(qubits, self.S0, self.sig, self.r, self.T)[1]
        print(values, pdf)
        c = (2 * error) ** (1 / (2 * u + 2))
        k = int(np.floor(2 ** qubits * (self.K - np.min(values)) / (np.max(values) - np.min(values))))
        print(k)
        # payoff = np.sum((pdf * (values - self.K))[values >= self.K)])

        optimal_prob = 1/2 - c + 2 * c * 2**qubits / (2**qubits - k) * np.sum((pdf * (values - values[k]))[values >= self.K])
        # ((prob - 0.5 + c) * (2 ** qu - 1 - k) / 2 / c) * (high - low) / 2 ** qu
        # Podría tomarse como optimal prob el último resultado con el circuito sin errores

        error_payoff = np.empty((len(m_s), self.steps, repeats))
        confidence = np.empty_like(error_payoff)

        for i, error in enumerate(self.error_steps):
            print(error)
            circuits = [[]]*len(m_s)
            noise_model = noise(error, measure=measure_error, thermal=thermal_error)
            basis_gates = noise_model.basis_gates
            for j, m in enumerate(m_s):
                qc = bin.load_Q_operator(qubits, m, self.S0, self.sig, self.r, self.T, self.K)[0]
                circuits[j] = qc

            for r in range(repeats):
                ones_s = [[]]*len(m_s)
                zeroes_s = [[]] * len(m_s)
                for j, m in enumerate(m_s):
                    ones, zeroes = bin.run_Q_operator(circuits[j], shots, basis_gates, noise_model)
                    ones_s[j] = int(ones)
                    zeroes_s[j] = int(zeroes)
                theta = np.linspace(0, np.pi / 2)
                theta_max_s, error_theta_s = max_likelihood(theta, m_s, ones_s, zeroes_s)
                #print(theta_max_s, error_theta_s)
                a_s, error_s = np.sin(theta_max_s) ** 2, np.abs(np.sin(2 * theta_max_s) * error_theta_s)
                #print(a_s, error_s)
                error_payoff[:, i, r] = a_s
                confidence[:, i, r] =np.abs(error_s) / 2

        try:
            os.makedirs(name_folder_data(self.data) + '/%s_bins/binary/amplitude_estimation' % bins)
        except:
            pass
        for j, m in enumerate(m_s):
            np.savetxt(name_folder_data(
                self.data) + '/%s_bins/binary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s).npz' % (
                       self.max_gate_error, self.steps, repeats, m), error_payoff[j])
            np.savetxt(name_folder_data(
                self.data) + '/%s_bins/binary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s)_confidence.npz' % (
                       self.max_gate_error, self.steps, repeats, m), confidence[j])

    def compute_save_errors_unary_amplitude_estimation(self, bins, error_name, repeats, M=4, measure_error=False, thermal_error=False,
                                   shots=500):
        qubits = int(bins)
        error_name = self.change_name(error_name, measure_error, thermal_error)
        noise = self.select_error(error_name, measure_error=measure_error, thermal_error=thermal_error)
        m_s = np.arange(0, M + 1, 1)
        (values, pdf), (mu, mean, variance)= un.get_pdf(qubits, self.S0, self.sig, self.r, self.T)

        optimal_prob = np.sum(pdf[values >= self.K] * (values[values >= self.K] - self.K)) / (np.max(values) - self.K)
        #print(np.sqrt(np.arcsin(optimal_prob)))
        error_payoff = np.empty((len(m_s), self.steps, repeats))
        confidence = np.empty_like(error_payoff)

        for i, error in enumerate(self.error_steps):
            print(error)
            circuits = [[]]*len(m_s)
            noise_model = noise(error, measure=measure_error, thermal=thermal_error)
            basis_gates = noise_model.basis_gates
            for j, m in enumerate(m_s):
                qc = un.load_Q_operator(qubits, m, self.S0, self.sig, self.r, self.T, self.K)
                circuits[j] = qc

            for r in range(repeats):
                ones_s = [[]]*len(m_s)
                zeroes_s = [[]] * len(m_s)
                for j, m in enumerate(m_s):
                    ones, zeroes = un.run_Q_operator(circuits[j], shots, basis_gates, noise_model)
                    ones_s[j] = int(ones)
                    zeroes_s[j] = int(zeroes)
                theta = np.linspace(0, np.pi / 2)
                theta_max_s, error_theta_s = max_likelihood(theta, m_s, ones_s, zeroes_s)
                #print(theta_max_s, error_theta_s)
                a_s, error_s = np.sin(theta_max_s) ** 2, np.abs(np.sin(2 * theta_max_s) * error_theta_s)
                #print(a_s, error_s)
                error_payoff[:, i, r] = a_s
                confidence[:, i, r] = np.abs(error_s) / 2

        try:
            os.makedirs(name_folder_data(self.data) + '/%s_bins/unary/amplitude_estimation' % bins)
        except:
            pass
        for j, m in enumerate(m_s):
            np.savetxt(name_folder_data(
                self.data) + '/%s_bins/unary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s).npz' % (
                       self.max_gate_error, self.steps, repeats, m), error_payoff[j])
            np.savetxt(name_folder_data(
                self.data) + '/%s_bins/unary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s)_confidence.npz' % (
                       self.max_gate_error, self.steps, repeats, m), confidence[j])

    def error_emplitude_estimation(self, bins, error_name, repeats, M=4, measure_error=False, thermal_error=False, error=0.05, u=0):

        (values, pdf), (mu, mean, variance) = un.get_pdf(bins, self.S0, self.sig, self.r, self.T)
        a_un = np.sum(pdf[values >= self.K] * (values[values >= self.K] - self.K)) / (np.max(values) - self.K)
        print(a_un)

        (values, pdf) = bin.get_pdf(int(np.log2(bins)), self.S0, self.sig, self.r, self.T)[1]
        c = (2 * error) ** (1 / (2 * u + 2))
        k = (2 ** int(np.log2(bins)) * (self.K - np.min(values)) / (np.max(values) - np.min(values)))
        # a_bin = 1/2 - c + 2 * c * 2**int(np.log2(bins)) / (2**int(np.log2(bins)) - k) * np.sum((pdf * (values - self.K))[values >= self.K])
        a_bin = 0.308

        fig, ax = plt.subplots()
        un_data = np.empty(M+1)
        un_conf = np.empty(M + 1)
        bin_data = np.empty(M + 1)
        bin_conf = np.empty(M + 1)
        for m in range(M+1):
            un_data_ = np.loadtxt(name_folder_data(
                    self.data) + '/%s_bins/unary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s).npz' % (
                           self.max_gate_error, self.steps, repeats, m))[0]
            un_conf_ = np.loadtxt(name_folder_data(
                self.data) + '/%s_bins/unary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s)_confidence.npz' % (
                                      self.max_gate_error, self.steps, repeats, m))[0]
            bin_data_ = np.loadtxt(name_folder_data(
                self.data) + '/%s_bins/binary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s).npz' % (
                                     self.max_gate_error, self.steps, repeats, m))[0]
            bin_conf_ = np.loadtxt(name_folder_data(
                self.data) + '/%s_bins/binary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s)_confidence.npz' % (
                                       self.max_gate_error, self.steps, repeats, m))[0]

            un_data[m], un_conf[m] = errors_experiment(un_data_, un_conf_)
            bin_data[m], bin_conf[m] = errors_experiment(bin_data_, bin_conf_)

        ax.scatter(np.arange(M+1), un_data, c='C0', marker='x', label='unary', zorder=10)
        ax.fill_between(np.arange(M+1), un_data - un_conf, un_data + un_conf, color='C0', alpha=0.3)
        ax.plot([0, M], [a_un, a_un], c='blue', ls='--')
        #ax.text(.5, a_un * (1.05), r'Optimal $a$ - unary')

        ax.scatter(np.arange(M + 1), bin_data, c='C1', marker='+', label='binary', zorder=10)
        ax.fill_between(np.arange(M + 1), bin_data - bin_conf, bin_data + bin_conf, color='C1', alpha=0.3, zorder=10)
        ax.plot([0, M], [a_bin, a_bin], c='orangered', ls='--')
        #ax.text(.5, a_bin * (0.95), r'Optimal $a$ - binary')
        ax.set(xlabel='AE iterations', ylabel=r'$a$', xticks=np.arange(0, M + 1), xticklabels=np.arange(0, M + 1))

        fig.savefig(name_folder_data(
            self.data) + '/%s_bins/' % bins + error_name + '_amplitude_estimation_perfect_circuit_results.pdf')

        un_error = np.abs(un_data - a_un)
        bin_error = np.abs(bin_data - a_bin)

        fig, bx = plt.subplots()
        bx.errorbar(np.arange(M + 1) + 0.1, un_error, un_conf, c='blue', alpha=0.7, capsize=10, ls=' ')
        bx.scatter(np.arange(M + 1) + 0.1, un_error, c='C0', label='unary', marker='x', zorder=3, s=100)
        bx.errorbar(np.arange(M + 1) - 0.1, bin_error, bin_conf, c='orangered', alpha=0.7, capsize=10, ls=' ')
        bx.scatter(np.arange(M + 1) - 0.1, bin_error, c='C1', label='binary', marker='+', zorder=3, s=100)

        shots=500
        error_bound = 1 / np.sqrt(shots * np.cumsum(1 + 2 * np.cumsum(np.arange(M+1))))

        bx.plot(np.arange(M + 1), error_bound, c='black', label='Classical sampling')
        bx.set(xlabel='AE iterations', xticks=np.arange(0, M+1), xticklabels=np.arange(0, M+1),
               ylabel=r'$\Delta a $', ylim=[0.0001,0.05])
        plt.yscale('log')
        bx.legend()

        fig.savefig(name_folder_data(
                self.data) + '/%s_bins/' % bins + error_name + '_amplitude_estimation_perfect_circuit_error.pdf')



    def paint_amplitude_estimation_unary(self, bins, error_name, repeats, M=4, measure_error=False, thermal_error=False):
        (values, pdf), (mu, mean, variance) = un.get_pdf(bins, self.S0, self.sig, self.r, self.T)
        a_un = np.sum(pdf[values >= self.K] * (values[values >= self.K] - self.K)) / (np.max(values) - self.K)
        error_name = self.change_name(error_name, measure_error, thermal_error)
        #noise = self.select_error(error_name, measure_error=measure_error, thermal_error=thermal_error)
        m_s = np.arange(0, M + 1, 1)
        valids = np.empty((M+1, self.steps))
        fig_0, ax_0 = plt.subplots()
        fig_1, ax_1 = plt.subplots()
        fig_2, ax_2 = plt.subplots(figsize=(10,4))
        ############ Unary
        for j, m in enumerate(m_s):
            a = np.loadtxt(name_folder_data(
                self.data) + '/%s_bins/unary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s).npz' % (
                       self.max_gate_error, self.steps, repeats, m+2))
            confidences= np.loadtxt(name_folder_data(
                self.data) + '/%s_bins/unary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s)_confidence.npz' % (
                                             self.max_gate_error, self.steps, repeats, m+2))

            data = np.empty((self.steps, 2))
            conf = np.empty((self.steps, 2))
            for _ in range(self.steps):
                data[_], conf[_], valids[j, _] = experimental_data(np.abs(a[_] - a_un), confidences[_])

            ax_0.scatter(100*self.error_steps, 100*data[:,0], color='C%s' % (2+j), label=r'M=%s' % m, marker='x')
            ax_0.fill_between(100*self.error_steps, 100*(data[:,0] - data[:, 1]), 100*(data[:,0] + data[:, 1]), color='C%s' % (2+j), alpha=0.3)


            ax_1.errorbar(100*self.error_steps, conf[:, 0], yerr=conf[:, 1], color='C%s' % (2 + j), alpha=0.7, capsize=10, ls=' ')
            ax_1.scatter(100*self.error_steps, conf[:, 0], color='C%s' % (2 + j), marker='x', zorder=3, s=100, label=r'M=%s' % m)
            #ax_1.errorbar(self.error_steps, conf[:, 0], yerr = conf[:, 1], color='C%s' % j, label=r'M=%s' % m, marker='x')


            '''ax_1.scatter(self.error_steps, conf[:, 0], color='C%s' % j, label=r'M=%s' % m, marker='x', zorder=10)
            ax_1.bar(self.error_steps, conf[:, 0] + conf[:, 1], width=self.max_gate_error / self.steps, color='C%s' % j, alpha=0.3, zorder=j)
            ax_1.bar(self.error_steps, conf[:, 0] - conf[:, 1], width=self.max_gate_error / self.steps, color='C%s' % j, alpha=0.7, zorder=j)'''

        ax_0.set(xlabel='single-qubit gate error (%)', ylabel='percentage off optimal $a$ (%)', ylim=[0, 8])
        ax_1.set(xlabel='single-qubit gate error (%)', ylabel='$\Delta a$', ylim=[0.001, 0.1], yscale='log')
        ax_0.legend(loc='upper left')
        ax_1.legend()
        fig_0.tight_layout()
        fig_1.tight_layout()
        fig_0.savefig(name_folder_data(
                self.data) + '/%s_bins/unary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_data.pdf' % (
                            self.max_gate_error, self.steps, repeats))
        fig_1.savefig(name_folder_data(
            self.data) + '/%s_bins/unary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_sample_error.pdf' % (
                          self.max_gate_error, self.steps, repeats))

        norm = colors.Normalize(vmin=0., vmax=1.)
        cmap = plt.get_cmap('Greys')


        hist = ax_2.imshow(valids / np.max(valids), cmap=cmap, norm=norm)
        ax_2.set(xticks=np.arange(0, self.steps+1, 2), xticklabels = np.round(100*self.max_gate_error * np.arange(0, self.steps+1, 2) / (self.steps-1), 2),
                 xlabel = 'single-qubit gate error (%)',
                 yticks=np.arange(M + 1), yticklabels = np.arange(M + 1), ylabel='AE iterations')
        plt.colorbar(hist, orientation='horizontal', fraction=.15)
        fig_2.tight_layout()
        fig_2.savefig(name_folder_data(
            self.data) + '/%s_bins/unary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_sample_valids.pdf' % (
                          self.max_gate_error, self.steps, repeats))

    def paint_amplitude_estimation_binary(self, bins, error_name, repeats, M=4, measure_error=False,
                                         thermal_error=False, error=0.05, u=0):

        (values, pdf) = bin.get_pdf(int(np.log2(bins)), self.S0, self.sig, self.r, self.T)[1]
        c = (2 * error) ** (1 / (2 * u + 2))
        k = (2 ** int(np.log2(bins)) * (self.K - np.min(values)) / (np.max(values) - np.min(values)))
        # a_bin = 1/2 - c + 2 * c * 2**int(np.log2(bins)) / (2**int(np.log2(bins)) - k) * np.sum((pdf * (values - self.K))[values >= self.K])
        a_bin = 0.308
        error_name = self.change_name(error_name, measure_error, thermal_error)
        #noise = self.select_error(error_name, measure_error=measure_error, thermal_error=thermal_error)
        m_s = np.arange(0, M + 1, 1)
        valids = np.empty((M+1, self.steps))
        fig_0, ax_0 = plt.subplots()
        fig_1, ax_1 = plt.subplots()
        fig_2, ax_2 = plt.subplots(figsize=(10,4))
        ############ Binary
        for j, m in enumerate(m_s):
            a = np.loadtxt(name_folder_data(
                self.data) + '/%s_bins/binary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s).npz' % (
                       self.max_gate_error, self.steps, repeats, m))
            confidences= np.loadtxt(name_folder_data(
                self.data) + '/%s_bins/binary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s)_confidence.npz' % (
                                             self.max_gate_error, self.steps, repeats, m))

            data = np.empty((self.steps, 2))
            conf = np.empty((self.steps, 2))
            for _ in range(self.steps):
                data[_], conf[_], valids[j, _] = experimental_data(np.abs(a[_] - a_bin), confidences[_])

            ax_0.scatter(100*self.error_steps, 100*data[:,0], color='C%s' % (2+j), label=r'M=%s' % m, marker='x')
            ax_0.fill_between(100*self.error_steps, 100*(data[:,0] - data[:, 1]), 100*(data[:,0] + data[:, 1]), color='C%s' % (2+j), alpha=0.3)

            #ax_0.plot([self.error_steps[0], self.error_steps[-1]], [a_bin, a_bin], color='black', ls='--', zorder=10)

            ax_1.errorbar(100*self.error_steps, conf[:, 0], yerr=conf[:, 1], color='C%s' % (2 + j), alpha=0.7, capsize=10,
                          ls=' ')
            ax_1.scatter(100*self.error_steps, conf[:, 0], color='C%s' % (2 + j), marker='x', zorder=3, s=100,
                         label=r'M=%s' % m)

        ax_0.set(xlabel='single-qubit gate error (%)', ylabel='percentage off optimal $a$ (%)', ylim=[0, 20])
        ax_1.set(xlabel='single-qubit gate error (%)', ylabel='$\Delta a$', ylim=[0.001, 0.1], yscale='log')
        ax_0.legend(loc='upper left')
        ax_1.legend()
        fig_0.tight_layout()
        fig_1.tight_layout()
        fig_0.savefig(name_folder_data(
                self.data) + '/%s_bins/binary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_data.pdf' % (
                            self.max_gate_error, self.steps, repeats))
        fig_1.savefig(name_folder_data(
            self.data) + '/%s_bins/binary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_sample_error.pdf' % (
                          self.max_gate_error, self.steps, repeats))

        cmap = plt.get_cmap('Greys')
        norm = colors.Normalize(vmin=0., vmax=1.)

        hist = ax_2.imshow(valids / np.max(valids), cmap=cmap, norm=norm)
        ax_2.set(xticks=np.arange(0, self.steps+1, 2), xticklabels = np.round(100*self.max_gate_error * np.arange(0, self.steps+1, 2) / (self.steps-1), 2),
                 xlabel = 'single-qubit gate error (%) ',
                 yticks=np.arange(M + 1), yticklabels = np.arange(M + 1), ylabel='AE iterations')
        plt.colorbar(hist, orientation='horizontal', fraction=.15)
        fig_2.tight_layout()
        fig_2.savefig(name_folder_data(
            self.data) + '/%s_bins/binary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_sample_valids.pdf' % (
                          self.max_gate_error, self.steps, repeats))











