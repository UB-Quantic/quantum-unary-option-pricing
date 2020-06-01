from errors import errors
import unary as un


S0 = 2
K = 1.9
sig = 0.4
r = 0.0501
T = 0.1
data = (S0, sig, r, T, K)

bins = 8
max_error_gate = 0.001
error_name = 'depolarizing'
repeats = 10
measure = True
thermal = True
steps = 6
Err = errors(data, max_error_gate, steps)
'''print('binary')
Err.compute_save_errors_binary(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)
print('unary')
Err.compute_save_errors_unary(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)
print('paint errors')
Err.paint_errors(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)'''
'''print('KL unary')
Err.compute_save_KL_unary(8, error_name, repeats, measure_error=measure, thermal_error=thermal)
print('KL binary')
Err.compute_save_KL_binary(8, error_name, repeats, measure_error=measure, thermal_error=thermal)

print('paint outcomes')
Err.paint_outcomes(bins, error_name, max_error_gate, repeats, measure_error=measure, thermal_error=thermal)
print('paint divergences')

Err.paint_divergences(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)'''
#Err.compute_save_errors_unary_amplitude_estimation(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)
#Err.compute_save_errors_binary_amplitude_estimation(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)
Err.error_emplitude_estimation(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)
Err.paint_amplitude_estimation_binary(bins, error_name, repeats, M=4, measure_error=measure, thermal_error=thermal)
Err.paint_amplitude_estimation_unary(bins, error_name, repeats, M=4, measure_error=measure, thermal_error=thermal)
#Err.compute_save_errors_binary(bins, error_name, repeats)
#Err.unary_mlae(bins, error_name, error_value, bin_error)
#Err.test_binary_Q(6)
# res = Err.unary_mlae(bins, error_name, 0.00, 4, shots=100)




