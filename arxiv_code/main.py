from errors import errors

S0 = 2
K = 1.9
sig = 0.4

r = 0.05
T = 0.1

data = (S0, sig, r, T, K)

max_error_gate = 0.005

steps = 11

Err = errors(data, max_error_gate, steps)
bins = 8
error_name = 'bitflip'
repeats = 10
measure=False
Err.paint(bins, error_name, repeats)