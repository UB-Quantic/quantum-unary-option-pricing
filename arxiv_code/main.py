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
error_name = 'bitflip'
repeats = 100
measure = False
thermal = False
steps = 11
Err = errors(data, max_error_gate, steps)
res = Err.test_unary_Q(bins)
print(res)


