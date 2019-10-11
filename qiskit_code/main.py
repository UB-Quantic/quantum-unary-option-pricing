from binary import binary_benchmark_sim, payoff_quantum_sim, binary_quantum_sim
from noise_mapping import noise_model_unary, unary_coupling, binary_coupling, noise_model_binary
from unary import unary_benchmark_sim, unary_qu_cl


#provider = IBMQ.load_account()

qu = 10 #Number of qubits for the probability distribution (ancilla not included)

S0 = 2
K = 1.9
sig = 0.4

r = 0.05
T = 0.1

error = 1

"""
noise_model=noise_model_unary(qu, error)
basis_gates=noise_model.basis_gates
coupling_map = unary_coupling(qu)
noise_objects = noise_model, coupling_map, basis_gates, error
print('Done')

unary_benchmark_sim(qu, S0, sig, r, T, noise_objects, all_samples=True)
print('Done')

#unary_qu_cl(qu, S0, sig, r, T, K, noise_objects)
"""

qu=3


noise_model=noise_model_binary(qu, error)
basis_gates=noise_model.basis_gates
coupling_map = binary_coupling(qu)
noise_objects = noise_model, coupling_map, basis_gates, error

binary_benchmark_sim(qu, S0, sig, r, T, noise_objects)

