from errors import errors
import unary as un


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
steps = 51
# Crear objeto de errores
Err = errors(data, max_error_gate, steps)


print('binary')
#Calcular errores al payoff en binario
Err.compute_save_errors_binary(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)
print('unary')
#Calcular errores al payoff en unario
Err.compute_save_errors_unary(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)


print('paint errors')
#Pintar los errores, figuras 14 del paper
Err.paint_errors(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)


print('KL unary')
#Calcular errores en la distribución de probabilidad en unario
Err.compute_save_KL_unary(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)
print('KL binary')
#Calcular errores en la distribución de probabilidad en binario
Err.compute_save_KL_binary(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)

print('paint outcomes')
#Pintar distribuciones de probabilidad en unario y binario, figuras 11 del paper
Err.paint_outcomes(bins, error_name, 0.0, repeats, measure_error=measure, thermal_error=thermal)
Err.paint_outcomes(bins, error_name, 0.001, repeats, measure_error=measure, thermal_error=thermal)
print('paint divergences')
#Pintar divergencias KL de probabilidad en unario y binario, figura 12 del paper
Err.paint_divergences(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)

print('AE unary')
#Amplitude Estimation unario
Err.compute_save_errors_unary_amplitude_estimation(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)
print('AE binary')
#Amplitude Estimation binario
Err.compute_save_errors_binary_amplitude_estimation(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)

print('paint AE')
#Pintar amplitude estimation unario y binario, figuras 15 del paper
Err.error_emplitude_estimation(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)
#Pintar errores en el payoff binario y unario, figuras 16 y 17
Err.paint_amplitude_estimation_binary(bins, error_name, repeats, M=4, measure_error=measure, thermal_error=thermal)
Err.paint_amplitude_estimation_unary(bins, error_name, repeats, M=4, measure_error=measure, thermal_error=thermal)





