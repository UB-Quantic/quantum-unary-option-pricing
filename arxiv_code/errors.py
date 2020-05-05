import numpy as np
import binary as bin
import unary as un

from aux_functions import *
from noise_mapping import *
from time import time
import os
import matplotlib.pyplot as plt
from qiskit import execute
from qiskit import Aer
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
            c, k, high, low, qc, dev = bin.load_payoff_quantum_sim(qubits, self.S0, self.sig, self.r, self.T, self.K)
            for r in range(repeats):
                qu_payoff_sim = bin.run_payoff_quantum_sim(qubits, c, k, high, low, qc, dev, shots, basis_gates, noise_model)
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
        qc, dev, (values, pdf), (mu, mean, variance) = bin.load_quantum_sim(qubits, self.S0, self.sig, self.r, self.T)
        probs = np.zeros((len(values), repeats))
        for r in range(repeats):
            probs[:, r] = bin.run_quantum_sim(qubits, qc, dev, shots, basis_gates, noise_model)

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
        qc, dev, (values, pdf), (mu, mean, variance) = un.load_quantum_sim(qubits, self.S0, self.sig, self.r, self.T)
        probs = np.zeros((len(values), repeats))
        for r in range(repeats):
            res = un.run_quantum_sim(qubits, qc, dev, shots, basis_gates, noise_model)
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

        qc, dev, (values, pdf), (mu, mean, variance) = un.load_quantum_sim(bins, self.S0, self.sig, self.r, self.T)

        width = (values[1] - values[0]) / 1.3
        exact_values = np.linspace(np.min(values), np.max(values), bins * 100)
        mu = (self.r - 0.5 * self.sig ** 2) * self.T + np.log(self.S0)
        from aux_functions import log_normal
        exact_pdf = log_normal(exact_values, mu, self.sig * np.sqrt(self.T))
        exact_pdf = exact_pdf * pdf[0] / exact_pdf[0]
        fig, ax = plt.subplots()

        ax.bar(values + width / 4, maxs_binary, width / 2, alpha=0.3, color='C1')
        ax.bar(values + width / 4, mins_binary, width / 2, alpha=.75, color='C1', label='binary')
        ax.bar(values - width / 4, maxs_unary, width / 2, alpha=0.3, color='C0')
        ax.bar(values - width / 4, mins_unary, width / 2, alpha=.75, color='C0', label='unary')
        ax.plot(exact_values, exact_pdf, color='black')
        ax.scatter(values - width / 4, means_unary, s=20, color='C0', marker='x', zorder=10)
        ax.scatter(values + width / 4, means_binary, s=20, color='C1', marker='x', zorder=10)
        ax.scatter(values, pdf, s=1250, color='black', label='PDF', marker='_', zorder=9)
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

            qc, dev, (values, pdf), (mu, mean, variance) = un.load_quantum_sim(bins, self.S0, self.sig, self.r, self.T)
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

            qc, dev, (values, pdf), (mu, mean, variance) = un.load_quantum_sim(bins, self.S0, self.sig, self.r, self.T)
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

    def test_unary_Q(self, qu):
        S0, sig, r, T, K = self.S0, self.sig, self.r, self.T, self.K
        iterations = 1
        iterations = int(iterations)
        mu = (r - 0.5 * sig ** 2) * T + np.log(S0)  # Define all the parameters to be used in the computation
        mean = np.exp(
            mu + 0.5 * T * sig ** 2)  # Set the relevant zone of study and create the mapping between qubit and option price, and
        # generate the target lognormal distribution within the interval
        variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
        S = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qu)
        ln = log_normal(S, mu, sig * np.sqrt(T))
        lognormal_parameters = un.rw_parameters(qu,
                                             ln)  # Solve for the parameters needed to create the target lognormal distribution
        prob_loading = un.rw_circuit(qu,
                                  lognormal_parameters)  # Build the probaility loading circuit with the adjusted parameters
        prob_loading_inv = un.rw_circuit_inv(qu,
                                  lognormal_parameters)
        payoff = un.payoff_circuit(qu, K, S)
        payoff_inv = un.payoff_circuit_inv(qu, K, S)

        diffusion = un.diffusion_operator(qu)
        oracle = un.oracle_operator(qu)

        qc = prob_loading + payoff
        job_payoff_sim = execute(qc, Aer.get_backend('statevector_simulator'))# Run the complete payoff expectation circuit through a simulator
        statevector = job_payoff_sim.result().data()['statevector']
        print('algorithm')
        for i in range(qu):
            print(statevector[2 ** i], statevector[2 ** qu + 2 ** i])

        job_payoff_sim = execute(qc + diffusion, Aer.get_backend(
            'statevector_simulator'))  # Run the complete payoff expectation circuit through a simulator
        statevector = job_payoff_sim.result().data()['statevector']
        print('diffusion')
        for i in range(qu):
            print(statevector[2 ** i], statevector[2 ** qu + 2 ** i])


        '''qc_Q = oracle + payoff_inv + prob_loading_inv + diffusion + prob_loading + payoff
        for i in range(iterations):
            qc += qc_Q'''

    def test_unary_mlae(self, bins, error_name, error_value, bin_error, measure_error=False, thermal_error=False, shots=100, mode='eis'):
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



    def unary_mlae(self, bins, error_name, error_value, bin_error, measure_error=False, thermal_error=False, shots=100, mode='eis'):
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

        a = un.MLAE(qu, self.data, m_s, shots, basis_gates, noise_model)
        print('theta', np.arcsin(np.sqrt(a)))
        print('a', a)

        qc, S = un.load_payoff_quantum_sim(qu, self.S0, self.sig, self.r, self.T, self.K)
        print(S, S[qu - 1])
        a_exact = self.cl_payoff / (S[qu - 1] - self.K)
        print('theta_exact', np.arcsin(np.sqrt(a_exact)))
        print('a exact', a_exact)
        payoff_qu = un.get_payoff_from_prob(a, qu, S, self.K)

        print(a, payoff_qu, self.cl_payoff)











