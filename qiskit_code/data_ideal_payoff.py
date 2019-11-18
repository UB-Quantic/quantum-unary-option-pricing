from binary_ideal_tests_try import binary_qu_cl, binary_benchmark_sim
from unary_ideal_tests import unary_qu_cl, classical_payoff, unary_benchmark_sim

#provider = IBMQ.load_account()
'''
Collective parameters for the lognormal distribution

'''
S0 = 2
K = 1.9
sig = 0.4

r = 0.05
T = 0.25

unary_qubits = 16
binary_qubits = 4

instances = 10

unary_payoff = []
binary_payoff = []
unary_err = []
binary_err = []

cl_payoff = classical_payoff(S0, sig, r, T, K)

for i in range(instances):
      #------------------------------------------------------
      qu = unary_qubits
      qu_payoff, diff = unary_qu_cl(qu, S0, sig, r, T, K, cl_payoff)
      unary_payoff.append(qu_payoff)
      unary_err.append(diff)
      #------------------------------------------------------
      qu = binary_qubits
      qu_payoff, diff = binary_qu_cl(qu, S0, sig, r, T, K, cl_payoff)
      binary_payoff.append(qu_payoff)
      binary_err.append(diff)

unary_payoff_mean = 0
unary_err_mean = 0
binary_payoff_mean = 0
binary_err_mean = 0
for i in range(instances):
      unary_payoff_mean += unary_payoff[i]/instances
      unary_err_mean += unary_err[i]/instances
      binary_payoff_mean += binary_payoff[i]/instances
      binary_err_mean += binary_err[i]/instances
      
print('With precision {}'.format(10000))
print('Classical Payoff: {}'.format(cl_payoff))
print('')
print('')
print('Unary')
print('')
print('With {} qubits and {} samples averaged over {} instances'.format(unary_qubits, 10000, instances))
print('Quantum Payoff: {}'.format(unary_payoff_mean))
print('')
print('Percentage off: {}%'.format(unary_err_mean))
print('')
print('')
print('Binary')
print('')
print('With {} qubits and {} samples averaged over {} instances'.format(binary_qubits, 10000, instances))
print('Quantum Payoff: {}'.format(binary_payoff_mean))
print('')
print('Percentage off: {}%'.format(binary_err_mean))

#unary_benchmark_sim(8, S0, sig, r, T)
#binary_benchmark_sim(3, S0, sig, r, T)

#unary_benchmark_sim(16, S0, sig, r, T)
#binary_benchmark_sim(4, S0, sig, r, T)


