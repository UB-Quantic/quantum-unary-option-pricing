from errors import errors
import unary as un
un.rw_circuit(8, [0,0,0,0,0,0])
S0 = 2
K = 1.9
sig = 0.4
r = 0.05
T = 0.1
data = (S0, sig, r, T, K)

bins = 8
max_error_gate = 0.005
error_name = 'measurement'
repeats = 100
measure = False
thermal = False
steps = 101
Err = errors(data, max_error_gate, steps)
probs = Err.test_inversion(bins, error_name, 0, measure_error=False, thermal_error=False, shots=10000)
print(probs)
'''print('binary')
Err.compute_save_errors_binary(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)
print('unary')
Err.compute_save_errors_unary(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)
print('paint errors')
Err.paint_errors(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)

print('KL unary')
Err.compute_save_KL_unary(8, error_name, repeats, measure_error=measure, thermal_error=thermal)
print('KL binary')
Err.compute_save_KL_binary(8, error_name, repeats, measure_error=measure, thermal_error=thermal)'''

'''
print('paint outcomes')
Err.paint_outcomes(bins, error_name, max_error_gate, repeats, measure_error=measure, thermal_error=thermal)
print('paint divergences')
Err.paint_divergences(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)'''


