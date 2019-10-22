from binary import binary_benchmark_sim, binary_qu_cl
from noise_mapping_II import noise_model_unary_measure, noise_model_unary_phaseflip, noise_model_unary_bitflip, unary_coupling, binary_coupling, noise_model_binary_measure, noise_model_binary_phaseflip, noise_model_binary_bitflip
from unary import unary_benchmark_sim, unary_qu_cl
import numpy as np
import matplotlib.pyplot as plt

#provider = IBMQ.load_account()
'''
Collective parameters for the lognormal distribution

'''
S0 = 2
K = 1.9
sig = 0.4

r = 0.05
T = 0.1

'''
Unary benchmarking with error

'''
 #Number of qubits for the probability distribution (ancilla not included)

unary_bitflip_payoff = []
unary_phaseflip_payoff = []
unary_measurement_payoff = []
binary_bitflip_payoff = []
binary_phaseflip_payoff = []
binary_measurement_payoff = []
percentage_error = []

for error in range(0,50,50):
      #------------------------------------------------------
      #--------------------Unary-----------------------------
      #------------------------------------------------------
      qu = 16
      err = 'bitflip'
      noise_model=noise_model_unary_bitflip(qu, error)
      basis_gates=noise_model.basis_gates
      coupling_map = unary_coupling(qu)
      noise_objects = noise_model, coupling_map, basis_gates, error
      print('Unary noise Done')
      unary_benchmark_sim(qu, S0, sig, r, T, noise_objects, err)
      print('Unary benchmark Done')
      qu_payoff, cl_payoff, diff = unary_qu_cl(qu, S0, sig, r, T, K, noise_objects)
      print('Unary payoff Done')
      percentage_error.append(error*0.001)
      unary_bitflip_payoff.append(qu_payoff)
      #------------------------------------------------------
      err = 'phaseflip'
      noise_model=noise_model_unary_phaseflip(qu, error)
      basis_gates=noise_model.basis_gates
      coupling_map = unary_coupling(qu)
      noise_objects = noise_model, coupling_map, basis_gates, error
      print('Unary noise Done')
      unary_benchmark_sim(qu, S0, sig, r, T, noise_objects, err)
      print('Unary benchmark Done')
      qu_payoff, cl_payoff, diff = unary_qu_cl(qu, S0, sig, r, T, K, noise_objects)
      print('Unary payoff Done')
      unary_phaseflip_payoff.append(qu_payoff)
      #------------------------------------------------------
      err = 'measurement'
      noise_model=noise_model_unary_measure(qu, error)
      basis_gates=noise_model.basis_gates
      coupling_map = unary_coupling(qu)
      noise_objects = noise_model, coupling_map, basis_gates, error
      print('Unary noise Done')
      unary_benchmark_sim(qu, S0, sig, r, T, noise_objects, err)
      print('Unary benchmark Done')
      qu_payoff, cl_payoff, diff = unary_qu_cl(qu, S0, sig, r, T, K, noise_objects)
      print('Unary payoff Done')
      unary_measurement_payoff.append(qu_payoff)
      #------------------------------------------------------
      #--------------------Binary----------------------------
      #------------------------------------------------------
      qu=4
      noise_model=noise_model_binary_bitflip(qu, error)
      basis_gates=noise_model.basis_gates
      coupling_map = binary_coupling(qu)
      noise_objects = noise_model, coupling_map, basis_gates, error
      print('Binary noise Done')
      binary_benchmark_sim(qu, S0, sig, r, T, noise_objects)
      print('Binary benchmark Done')
      qu_payoff, cl_payoff, diff = binary_qu_cl(qu, S0, sig, r, T, K, noise_objects)
      print('Binary payoff Done')
      binary_bitflip_payoff.append(qu_payoff)
      #------------------------------------------------------
      noise_model=noise_model_binary_phaseflip(qu, error)
      basis_gates=noise_model.basis_gates
      coupling_map = binary_coupling(qu)
      noise_objects = noise_model, coupling_map, basis_gates, error
      print('Binary noise Done')
      binary_benchmark_sim(qu, S0, sig, r, T, noise_objects)
      print('Binary benchmark Done')
      qu_payoff, cl_payoff, diff = binary_qu_cl(qu, S0, sig, r, T, K, noise_objects)
      print('Binary payoff Done')
      binary_phaseflip_payoff.append(qu_payoff)
      #------------------------------------------------------
      noise_model=noise_model_binary_measure(qu, error)
      basis_gates=noise_model.basis_gates
      coupling_map = binary_coupling(qu)
      noise_objects = noise_model, coupling_map, basis_gates, error
      print('Binary noise Done')
      binary_benchmark_sim(qu, S0, sig, r, T, noise_objects)
      print('Binary benchmark Done')
      qu_payoff, cl_payoff, diff = binary_qu_cl(qu, S0, sig, r, T, K, noise_objects)
      print('Binary payoff Done')
      binary_measurement_payoff.append(qu_payoff)
      
#------------------------------------------------------      
fig, ax = plt.subplots()
ax.scatter(percentage_error, unary_bitflip_payoff, s=50, color='C0', label='Unary', marker='x')
ax.scatter(percentage_error, binary_bitflip_payoff, s=50, color='C1', label='Binary', marker='+')
ax.axhline(cl_payoff, color='C3', label='Classical')
plt.ylabel('Expected payoff')
plt.xlabel('Device error')
plt.title('Expected payoff with increasing bitflip error - qasm_simulator')
ax.legend()
fig.tight_layout()
fig.savefig('unary/K:{}_S0:{}_sig:{}_r:{}_T:{}_bitflip_payoff.png'.format(K, S0, sig, r, T))
#------------------------------------------------------
fig, ax = plt.subplots()
ax.scatter(percentage_error, np.abs(100 * (cl_payoff - unary_bitflip_payoff) * 2 / (cl_payoff + unary_bitflip_payoff)), s=50, color='C0', label='Unary', marker='x')
ax.scatter(percentage_error, np.abs(100 * (cl_payoff - binary_bitflip_payoff) * 2 / (cl_payoff + binary_bitflip_payoff)), s=50, color='C1', label='Binary', marker='+')
plt.ylabel('Percentage off')
plt.xlabel('Device error')
plt.title('Percentage off classical expected payoff with increasing bitflip error - qasm_simulator')
ax.legend()
fig.tight_layout()
fig.savefig('unary/K:{}_S0:{}_sig:{}_r:{}_T:{}_bitflip_percentage.png'.format(K, S0, sig, r, T))
#------------------------------------------------------
#------------------------------------------------------      
fig, ax = plt.subplots()
ax.scatter(percentage_error, unary_phaseflip_payoff, s=50, color='C0', label='Unary', marker='x')
ax.scatter(percentage_error, binary_phaseflip_payoff, s=50, color='C1', label='Binary', marker='+')
ax.axhline(cl_payoff, color='C3', label='Classical')
plt.ylabel('Expected payoff')
plt.xlabel('Device error')
plt.title('Expected payoff with increasing phaseflip error - qasm_simulator')
ax.legend()
fig.tight_layout()
fig.savefig('unary/K:{}_S0:{}_sig:{}_r:{}_T:{}_phaseflip_payoff.png'.format(K, S0, sig, r, T))
#------------------------------------------------------
fig, ax = plt.subplots()
ax.scatter(percentage_error, np.abs(100 * (cl_payoff - unary_phaseflip_payoff) * 2 / (cl_payoff + unary_phaseflip_payoff)), s=50, color='C0', label='Unary', marker='x')
ax.scatter(percentage_error, np.abs(100 * (cl_payoff - binary_phaseflip_payoff) * 2 / (cl_payoff + binary_phaseflip_payoff)), s=50, color='C1', label='Binary', marker='+')
plt.ylabel('Percentage off')
plt.xlabel('Device error')
plt.title('Percentage off classical expected payoff with increasing phaseflip error - qasm_simulator')
ax.legend()
fig.tight_layout()
fig.savefig('unary/K:{}_S0:{}_sig:{}_r:{}_T:{}_phaseflip_percentage.png'.format(K, S0, sig, r, T))
#------------------------------------------------------
#------------------------------------------------------      
fig, ax = plt.subplots()
ax.scatter(percentage_error, unary_measurement_payoff, s=50, color='C0', label='Unary', marker='x')
ax.scatter(percentage_error, binary_measurement_payoff, s=50, color='C1', label='Binary', marker='+')
ax.axhline(cl_payoff, color='C3', label='Classical')
plt.ylabel('Expected payoff')
plt.xlabel('Device error')
plt.title('Expected payoff with increasing measurement error - qasm_simulator')
ax.legend()
fig.tight_layout()
fig.savefig('unary/K:{}_S0:{}_sig:{}_r:{}_T:{}_measurement_payoff.png'.format(K, S0, sig, r, T))
#------------------------------------------------------
fig, ax = plt.subplots()
ax.scatter(percentage_error, np.abs(100 * (cl_payoff - unary_measurement_payoff) * 2 / (cl_payoff + unary_measurement_payoff)), s=50, color='C0', label='Unary', marker='x')
ax.scatter(percentage_error, np.abs(100 * (cl_payoff - binary_measurement_payoff) * 2 / (cl_payoff + binary_measurement_payoff)), s=50, color='C1', label='Binary', marker='+')
plt.ylabel('Percentage off')
plt.xlabel('Device error')
plt.title('Percentage off classical expected payoff with increasing measurement error - qasm_simulator')
ax.legend()
fig.tight_layout()
fig.savefig('unary/K:{}_S0:{}_sig:{}_r:{}_T:{}_measurement_percentage.png'.format(K, S0, sig, r, T))
#------------------------------------------------------







