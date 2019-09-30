from classical_simulations.MC_aux import MC, painting_MC
from unary_representation.fit_gaussian import fit, paint_fit
from unary_representation.payoff import Payoff

qu = 9
S0 = 125
sigma = 20
K = 125
gpu=False
paint_fit(qu, S0, sigma, K, method='Powell', gpu=gpu)
CP, QP = Payoff(qu, S0, sigma, K, gpu)

print(CP, QP)
