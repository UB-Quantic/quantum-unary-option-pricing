from errors import errors
import unary as un
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--bins", default=8, type=int)
parser.add_argument("--error_c", default=0.05, type=float)

# 8 bins: 0.05
# 16 bins: 0.0065
# En adelante: 0.005
args = vars(parser.parse_args())

S0 = 2
K = 1.9
sig = 0.4
r = 0.05
T = 0.1
data = (S0, sig, r, T, K)

bins = args.get('bins')
error_c = args.get('error_c')
bins_list = [7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20] # Ready for painting distinct bins

max_error_gate = 0.005
error_name = 'depolarizing'
repeats = 100
measure = True
thermal = False
steps = 51
# Create error objecterror=
Err = errors(data, max_error_gate, steps)
#Err.paint_AE_unary_run_bins(bins_list, error_name, repeats, M=4, step=3, measure_error=measure, thermal_error=thermal)
#Err.paint_cl_payoff(100)
#print('binary')
# Compute payoff errors in binary
# Err.compute_save_errors_binary(bins, error_name, repeats, measure_error=measure, thermal_error=thermal, error_c=error_c)
#print('unary')
# Compute payoff errors in unary
#Err.compute_save_errors_unary(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)


#print('paint errors')
#Paint errors corresponding to Fig. 14
Err.paint_errors(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)


#print('KL unary')
#Compute errors in probability distribution, unary
#Err.compute_save_KL_unary(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)
#print('KL binary')
#Compute errors in probability distribution, binary
#Err.compute_save_KL_binary(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)

#print('paint outcomes')
#Paint probability distributions, figures 11
#Err.paint_outcomes(bins, error_name, 0.0, repeats, measure_error=measure, thermal_error=thermal)
#Err.paint_outcomes(bins, error_name, 0.001, repeats, measure_error=measure, thermal_error=thermal)
Err.paint_outcomes(bins, error_name, 0.005, repeats, measure_error=measure, thermal_error=thermal)
#print('paint divergences')
#Paint KL divergences in probability distribution, figure 12
Err.paint_divergences(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)

print('AE unary')
#Amplitude Estimation unary
#Err.compute_save_errors_unary_amplitude_estimation(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)
#Err.error_emplitude_estimation_unary(bins, error_name, repeats, M=4, measure_error=measure, error=error_c)
#print('AE binary')
#Amplitude Estimation binary

#Err.compute_save_errors_binary_amplitude_estimation(bins, error_name, repeats, measure_error=measure, thermal_error=thermal, error_c=error_c)
Err.error_emplitude_estimation_binary(bins, error_name, repeats, M=4, measure_error=measure, error=error_c)

print('paint AE')
#Unary and Binary Amplitude Estimation, figures 15
Err.error_emplitude_estimation(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)
#Unary and Binary payoff errors, figures 16, 17
Err.paint_amplitude_estimation_unary(bins, error_name, repeats, M=4, measure_error=measure, thermal_error=thermal)
Err.paint_amplitude_estimation_binary(bins, error_name, repeats, M=4, measure_error=measure, thermal_error=thermal)

'''Err.paint_AE_unary_run_bins(bins_list, error_name, repeats, M=4, step=3,
                                measure_error=measure, thermal_error=thermal, shots=500, alpha=0.05)'''
#Err.paint_amplitude_estimation(bins, error_name, repeats, M=4, measure_error=measure, thermal_error=thermal)






