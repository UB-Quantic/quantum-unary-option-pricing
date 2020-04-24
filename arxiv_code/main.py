from errors import errors

S0 = 2
K = 1.9
sig = 0.4
r = 0.05
T = 0.1
data = (S0, sig, r, T, K)

max_error_gate = 0.005

steps = 101

Err = errors(data, max_error_gate, steps)
bins = 8
error_name = 'bitflip'
repeats = 100
measure=False

Err.compute_save_errors_unary(bins, error_name, repeats)
Err.compute_save_errors_binary(bins, error_name, repeats)
Err.paint(bins, error_name, repeats)