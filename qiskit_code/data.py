from binary_III import binary_qu_cl
from noise_mapping_III import noise_model_measure, noise_model_phaseflip, noise_model_bitflip, noise_model_total
from unary_III import unary_qu_cl, classical_payoff

#provider = IBMQ.load_account()
'''
Collective parameters for the lognormal distribution

'''
S0 = 2
K = 1.9
sig = 0.4

r = 0.05
T = 0.1

bitflip_error = []
phaseflip_error = []
measurement_error = []
noise_error = []

cl_payoff = classical_payoff(S0, sig, r, T, K)

for error in range(100):
      bitflip_error.append(error*0.00005)
      phaseflip_error.append(error*0.00005)
      measurement_error.append(error*0.0005)
      noise_error.append(error*0.00005)
      
      unary_bitflip = []
      unary_phaseflip = []
      unary_measurement = []
      unary_noise = []
      
      binary_bitflip = []
      binary_phaseflip = []
      binary_measurement = []
      binary_noise = []
      
      for i in range(100):
            #------------------------------------------------------
            qu = 8
            noise_model=noise_model_bitflip(qu, error)
            basis_gates=noise_model.basis_gates
            noise_objects = noise_model, basis_gates, error
            qu_payoff, diff = unary_qu_cl(qu, S0, sig, r, T, K, noise_objects, cl_payoff)
            unary_bitflip.append(diff)
            #------------------------------------------------------
            noise_model=noise_model_phaseflip(qu, error)
            basis_gates=noise_model.basis_gates
            noise_objects = noise_model, basis_gates, error
            qu_payoff, diff = unary_qu_cl(qu, S0, sig, r, T, K, noise_objects, cl_payoff)
            unary_phaseflip.append(diff)
            #------------------------------------------------------
            noise_model=noise_model_measure(qu, error)
            basis_gates=noise_model.basis_gates
            noise_objects = noise_model, basis_gates, error
            qu_payoff, diff = unary_qu_cl(qu, S0, sig, r, T, K, noise_objects, cl_payoff)
            unary_measurement.append(diff)
            #------------------------------------------------------
            noise_model=noise_model_total(qu, error)
            basis_gates=noise_model.basis_gates
            noise_objects = noise_model, basis_gates, error
            qu_payoff, diff = unary_qu_cl(qu, S0, sig, r, T, K, noise_objects, cl_payoff)
            unary_noise.append(diff)
      
            #------------------------------------------------------
            qu = 3
            noise_model=noise_model_bitflip(qu, error)
            basis_gates=noise_model.basis_gates
            noise_objects = noise_model, basis_gates, error
            qu_payoff, diff = binary_qu_cl(qu, S0, sig, r, T, K, noise_objects, cl_payoff)
            binary_bitflip.append(diff)
            #------------------------------------------------------
            noise_model=noise_model_phaseflip(qu, error)
            basis_gates=noise_model.basis_gates
            noise_objects = noise_model, basis_gates, error
            qu_payoff, diff = binary_qu_cl(qu, S0, sig, r, T, K, noise_objects, cl_payoff)
            binary_phaseflip.append(diff)
            #------------------------------------------------------
            noise_model=noise_model_measure(qu, error)
            basis_gates=noise_model.basis_gates
            noise_objects = noise_model, basis_gates, error
            qu_payoff, diff = binary_qu_cl(qu, S0, sig, r, T, K, noise_objects, cl_payoff)
            binary_measurement.append(diff)
            #------------------------------------------------------
            noise_model=noise_model_total(qu, error)
            basis_gates=noise_model.basis_gates
            noise_objects = noise_model, basis_gates, error
            qu_payoff, diff = unary_qu_cl(qu, S0, sig, r, T, K, noise_objects, cl_payoff)
            binary_noise.append(diff)
            
      with open("results/unary_rep_bitflip_{}_error.txt".format(error), "w") as file:
            file.write(str(unary_bitflip))
      with open("results/unary_rep_phaseflip_{}_error.txt".format(error), "w") as file:
            file.write(str(unary_phaseflip))
      with open("results/unary_rep_measurement_{}_error.txt".format(error), "w") as file:
            file.write(str(unary_measurement))
      with open("results/unary_rep_noise_{}_error.txt".format(error), "w") as file:
            file.write(str(unary_noise))
            
      with open("results/binary_rep_bitflip_{}_error.txt".format(error), "w") as file:
            file.write(str(binary_bitflip))
      with open("results/binary_rep_phaseflip_{}_error.txt".format(error), "w") as file:
            file.write(str(binary_phaseflip))
      with open("results/binary_rep_measurement_{}_error.txt".format(error), "w") as file:
            file.write(str(binary_measurement))
      with open("results/binary_rep_noise_{}_error.txt".format(error), "w") as file:
            file.write(str(binary_noise))
            