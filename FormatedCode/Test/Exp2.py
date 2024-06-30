import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))

from Phase1 import read_file
import numpy as np
import matplotlib.pyplot as plt


returnVal = read_file("D:\\1 - Study\\6 - DTW_project\\Container\\Log_Run_Burning_A_1.txt")

param = returnVal[0]
strain = np.array(returnVal[1])/(-4)
stress = np.array(returnVal[2])/(300)
bodyOpen = np.array(returnVal[3])/(20)

TrainVal = [[ai, bi, ci] for ai, bi, ci in zip(strain, stress, bodyOpen)]

TrainVal = np.array(TrainVal)

for i in range(10):
    index = 10+i
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
    
    ax1.plot(TrainVal[index][0]*(-4),TrainVal[index][1]*(300), label='Simulation Line', color="blue")
    ax1.plot(TrainVal[index][2]*(20),TrainVal[index][1]*(300), color="blue")

    ax2.plot(TrainVal[index][0]*(-4),(((TrainVal[index][2]*(20))/(TrainVal[index][0]*(-4)))), label='Predict Line', color="red")

    ax1.set_xlabel('Strain')
    ax1.set_ylabel('Stress')
    ax1.set_title("Simulation")

    ax2.set_xlabel('Strain')
    ax2.set_ylabel('Strain + /Strain -')
    ax1.set_title("V plot")
    print(TrainVal[index][0][:10]*(-4))
    print(TrainVal[index][2][:10]*(20))
    # plt.title('Predict - Simulation compare')
    plt.legend()
    plt.show()