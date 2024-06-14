from Phase1 import *
import numpy as np
import matplotlib.pyplot as plt

# returnVal = RunSimulationThread(0,np.random.rand(11))
returnVal = read_file("D:\\1 - Study\\6 - DTW_project\\Container\\Log_Run_Burning_1.txt")
# print(returnVal)
param = returnVal[0][0]
strain = returnVal[1][0]
stress = returnVal[2][0]
bodyOpen = returnVal[3][0]
MSE, interpolate = findF(stress, bodyOpen, strain)

print(MSE)

plt.plot(np.concatenate((np.flip(strain),bodyOpen)) , np.concatenate((np.flip(stress),stress)), label = "Simulation")
plt.plot(np.concatenate((np.flip(strain_exp),bodyOpen_exp)) , interpolate, marker = "x", label = "Interpolate")
plt.plot(strain_exp , stress_exp, label = "Experiment",color="green")
plt.plot(bodyOpen_exp , stress_exp,color="green")

# Add titles and labels
plt.title('Example')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()