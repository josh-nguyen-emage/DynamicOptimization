import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))

from matplotlib import pyplot as plt
import numpy as np

from real_simulation.GlobalLib import findF
from real_simulation.RunSequent.function import ReadLabFile, read_file

# ------------------------------------------

filename = 'stdFile\\G7-Uni-AxialTest.dat'  # Replace 'data.txt' with your file path
list_a, list_b, list_c = ReadLabFile(filename)
list_c = np.array(list_c)*(-1000)
list_a = np.array(list_a)*1

Y_exp = list_c
Z_exp = list_a

X, Y, Z = read_file("RunLog\\Log_Run_E_Phase1.txt")
Y = Y[:,1:]
Z = Z[:,1:]
Y *= -1000
Z *= -1

for idx in range(len(X)):
    idx = 10
    plt.plot(Y_exp,Z_exp, label='Simulation Line')
    plt.plot(Y[idx][:50],Z[idx][:50], label='Predict Line')
    plt.xlabel('Strain')
    plt.ylabel('Stress')
    mean, predic = findF(Y[idx],Z[idx])
    print(np.mean((predic-Z_exp)**2,1))
    plt.title(str(idx) + " : " + str(mean))
    plt.legend()
    plt.show()

    break
