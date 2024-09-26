import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))
from Phase1 import *
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":
    # returnVal = RunSimulationThread(16,np.random.rand(11))
    # print(returnVal)
    # strain = returnVal[0]
    # stress = returnVal[1]
    # bodyOpen = returnVal[2]
    # MSE, interpolate = findF(stress, bodyOpen, strain)

    # returnVal = RunSimulationThread(0,np.random.rand(11))
    returnVal = read_file("D:\\1 - Study\\6 - DTW_project\\Container\\Log_Run_Bayes_4-7.txt")
    # print(returnVal)
    minIdx = 0
    minMSE = 1000000
    container = []
    for idx in range(len(returnVal[0])):
    # for idx in range(100):
        print(idx)
        param = returnVal[0][idx]
        strain = returnVal[1][idx]
        stress = returnVal[2][idx]
        bodyOpen = returnVal[3][idx]
        MSE, interpolate = findF(stress, bodyOpen, strain)
        if MSE < minMSE:
            minMSE = MSE
            minIdx = idx
        container.append(MSE)

        # plt.plot(np.concatenate((np.flip(strain),bodyOpen)) , np.concatenate((np.flip(stress),stress)), label = "Simulation")
        # plt.plot(np.concatenate((np.flip(strain_exp),bodyOpen_exp)) , interpolate, marker = "x", label = "Interpolate")
        # plt.plot(strain_exp , stress_exp, label = "Experiment",color="green")
        # plt.plot(bodyOpen_exp , stress_exp,color="green")
        # plt.show()

    param = returnVal[0][minIdx]
    strain = returnVal[1][minIdx]
    stress = returnVal[2][minIdx]
    bodyOpen = returnVal[3][minIdx]
    MSE, interpolate = findF(stress, bodyOpen, strain)
    print("idx:",minIdx)
    print("MSE:",MSE)
    print("param:",param)

    plt.plot(np.concatenate((np.flip(strain),bodyOpen)) , np.concatenate((np.flip(stress),stress)), label = "Simulation")
    # plt.plot(np.concatenate((np.flip(strain_exp),bodyOpen_exp)) , interpolate, marker = "x", label = "Interpolate")
    plt.plot(strain_exp , stress_exp, label = "Experiment",color="green")
    plt.plot(bodyOpen_exp , stress_exp,color="green")
    plt.xlabel('Strain (‰)')
    plt.ylabel('Stress (MPa)')
    plt.legend()
    plt.show()

    # Add titles and labels
    plt.plot(np.array(range(len(container)))/16 , container, label = "MSE", marker = "o", linestyle = '')
    min_values = [min(container[:i+1]) for i in range(len(container))]
    plt.plot(np.array(range(len(min_values)))/16 , min_values)

    plt.title('ID:'+str(minIdx) + " MSE:"+str(minMSE))
    plt.xlabel('Run time')
    plt.ylabel('MSE')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()