import matplotlib.pyplot as plt

#provider = IBMQ.load_account()
'''
Collective parameters for the lognormal distribution

'''
S0 = 2
K = 1.9
sig = 0.4

r = 0.05
T = 0.1

unary_qubits = 8
binary_qubits = 3

p_out = 0.10
 
unary_bitflip_top = []
unary_bitflip_bottom = []
unary_bitflip_mean = []

unary_phaseflip_top = []
unary_phaseflip_bottom = []
unary_phaseflip_mean = []

unary_measurement_top = []
unary_measurement_bottom = []
unary_measurement_mean = []

unary_noise_top = []
unary_noise_bottom = []
unary_noise_mean = []

binary_bitflip_top = []
binary_bitflip_bottom = []
binary_bitflip_mean = []

binary_phaseflip_top = []
binary_phaseflip_bottom = []
binary_phaseflip_mean = []

binary_measurement_top = []
binary_measurement_bottom = []
binary_measurement_mean = []

binary_noise_top = []
binary_noise_bottom = []
binary_noise_mean = []

unary_bitflip = []
unary_phaseflip = []
unary_measurement = []
unary_noise = []

binary_bitflip = []
binary_phaseflip = []
binary_measurement = []
binary_noise = []

bitflip_error = []
phaseflip_error = []
measurement_error = []
noise_error = []

for error in range(100):
      bitflip_error.append(100*error*0.00005)
      phaseflip_error.append(100*error*0.00005)
      measurement_error.append(100*error*0.0005)
      noise_error.append(100*error*0.00005)
      with open("results/unary_rep_bitflip_{}_error.txt".format(error), "r") as file:
            for line in file:
                  unary_bitflip = eval(line)
      with open("results/unary_rep_phaseflip_{}_error.txt".format(error), "r") as file:
            for line in file:
                  unary_phaseflip = eval(line)
      with open("results/unary_rep_measurement_{}_error.txt".format(error), "r") as file:
            for line in file:
                  unary_measurement = eval(line)
      with open("results/unary_rep_noise_{}_error.txt".format(error), "r") as file:
            for line in file:
                  unary_noise = eval(line)
            
      with open("results/binary_rep_bitflip_{}_error.txt".format(error), "r") as file:
            for line in file:
                  binary_bitflip = eval(line)
      with open("results/binary_rep_phaseflip_{}_error.txt".format(error), "r") as file:
            for line in file:
                  binary_phaseflip = eval(line)
      with open("results/binary_rep_measurement_{}_error.txt".format(error), "r") as file:
            for line in file:
                  binary_measurement = eval(line)
      with open("results/binary_rep_noise_{}_error.txt".format(error), "r") as file:
            for line in file:
                  binary_noise = eval(line)
            
      unary_bitflip.sort()
      mean = 0
      for i in unary_bitflip:
            mean += i
      unary_bitflip_mean.append(mean/len(unary_bitflip))
      unary_bitflip_top.append(unary_bitflip[int(p_out*len(unary_bitflip))])
      unary_bitflip_bottom.append(unary_bitflip[len(unary_bitflip)-1-int(p_out*len(unary_bitflip))])
      
      unary_phaseflip.sort()
      mean = 0
      for i in unary_phaseflip:
            mean += i
      unary_phaseflip_mean.append(mean/len(unary_phaseflip))
      unary_phaseflip_top.append(unary_phaseflip[int(p_out*len(unary_phaseflip))])
      unary_phaseflip_bottom.append(unary_phaseflip[len(unary_phaseflip)-1-int(p_out*len(unary_phaseflip))])
      
      unary_measurement.sort()
      mean = 0
      for i in unary_measurement:
            mean += i
      unary_measurement_mean.append(mean/len(unary_measurement))
      unary_measurement_top.append(unary_measurement[int(p_out*len(unary_measurement))])
      unary_measurement_bottom.append(unary_measurement[len(unary_measurement)-1-int(p_out*len(unary_measurement))])
      
      unary_noise.sort()
      mean = 0
      for i in unary_noise:
            mean += i
      unary_noise_mean.append(mean/len(unary_noise))
      unary_noise_top.append(unary_noise[int(p_out*len(unary_noise))])
      unary_noise_bottom.append(unary_noise[len(unary_noise)-1-int(p_out*len(unary_noise))])
      
      binary_bitflip.sort()
      mean = 0
      for i in binary_bitflip:
            mean += i
      binary_bitflip_mean.append(mean/len(binary_bitflip))
      binary_bitflip_top.append(binary_bitflip[int(p_out*len(binary_bitflip))])
      binary_bitflip_bottom.append(binary_bitflip[len(binary_bitflip)-1-int(p_out*len(binary_bitflip))])
      
      binary_phaseflip.sort()
      mean = 0
      for i in binary_phaseflip:
            mean += i
      binary_phaseflip_mean.append(mean/len(binary_phaseflip))
      binary_phaseflip_top.append(binary_phaseflip[int(p_out*len(binary_phaseflip))])
      binary_phaseflip_bottom.append(binary_phaseflip[len(binary_phaseflip)-1-int(p_out*len(binary_phaseflip))])
      
      binary_measurement.sort()
      mean = 0
      for i in binary_measurement:
            mean += i
      binary_measurement_mean.append(mean/len(binary_measurement))
      binary_measurement_top.append(binary_measurement[int(p_out*len(binary_measurement))])
      binary_measurement_bottom.append(binary_measurement[len(binary_measurement)-1-int(p_out*len(binary_measurement))])
      
      binary_noise.sort()
      mean = 0
      for i in binary_noise:
            mean += i
      binary_noise_mean.append(mean/len(binary_noise))
      binary_noise_top.append(binary_noise[int(p_out*len(binary_noise))])
      binary_noise_bottom.append(binary_noise[len(binary_noise)-1-int(p_out*len(binary_noise))])
      

fig, ax = plt.subplots()
ax.scatter(bitflip_error, unary_bitflip_mean, s=10, color='C0', label='unary', marker='x')
ax.scatter(bitflip_error, binary_bitflip_mean, s=10, color='C1', label='binary', marker='+')
ax.fill_between(bitflip_error, unary_bitflip_top, unary_bitflip_bottom, alpha=0.2, facecolor='C0')
ax.fill_between(bitflip_error, binary_bitflip_top, binary_bitflip_bottom, alpha=0.2, facecolor='C1')
plt.ylabel('percentage off classical value (%)')
plt.xlabel('single-qubit gate error (%)')
ax.legend()
fig.tight_layout()
fig.savefig('results/{}-{}_qubits_K:{}_S0:{}_sig:{}_r:{}_T:{}_bitflip_percentage_off.png'.format(unary_qubits, binary_qubits, K, S0, sig, r, T))
#------------------------------------------------------
fig, ax = plt.subplots()
ax.scatter(phaseflip_error, unary_phaseflip_mean, s=10, color='C0', label='unary', marker='x')
ax.scatter(phaseflip_error, binary_phaseflip_mean, s=10, color='C1', label='binary', marker='+')
ax.fill_between(phaseflip_error, unary_phaseflip_top, unary_phaseflip_bottom, alpha=0.2, facecolor='C0')
ax.fill_between(phaseflip_error, binary_phaseflip_top, binary_phaseflip_bottom, alpha=0.2, facecolor='C1')
plt.ylabel('percentage off classical value (%)')
plt.xlabel('single-qubit gate error (%)')
ax.legend()
fig.tight_layout()
fig.savefig('results/{}-{}_qubits_K:{}_S0:{}_sig:{}_r:{}_T:{}_phaseflip_percentage_off.png'.format(unary_qubits, binary_qubits, K, S0, sig, r, T))
#------------------------------------------------------    
fig, ax = plt.subplots()
ax.scatter(measurement_error, unary_measurement_mean, s=10, color='C0', label='unary', marker='x')
ax.scatter(measurement_error, binary_measurement_mean, s=10, color='C1', label='binary', marker='+')
ax.fill_between(measurement_error, unary_measurement_top, unary_measurement_bottom, alpha=0.2, facecolor='C0')
ax.fill_between(measurement_error, binary_measurement_top, binary_measurement_bottom, alpha=0.2, facecolor='C1')
plt.ylabel('percentage off classical value (%)')
plt.xlabel('single-qubit gate error (%)')
ax.legend()
fig.tight_layout()
fig.savefig('results/{}-{}_qubits_K:{}_S0:{}_sig:{}_r:{}_T:{}_measurement_percentage_off.png'.format(unary_qubits, binary_qubits, K, S0, sig, r, T))
#------------------------------------------------------     
fig, ax = plt.subplots()
ax.scatter(noise_error, unary_noise_mean, s=10, color='C0', label='unary', marker='x')
ax.scatter(noise_error, binary_noise_mean, s=10, color='C1', label='binary', marker='+')
ax.fill_between(noise_error, unary_noise_top, unary_noise_bottom, alpha=0.2, facecolor='C0')
ax.fill_between(noise_error, binary_noise_top, binary_noise_bottom, alpha=0.2, facecolor='C1')
plt.ylabel('percentage off classical value (%)')
plt.xlabel('single-qubit gate error (%)')
ax.legend()
fig.tight_layout()
fig.savefig('results/{}-{}_qubits_K:{}_S0:{}_sig:{}_r:{}_T:{}_noise_percentage_off.png'.format(unary_qubits, binary_qubits, K, S0, sig, r, T))
