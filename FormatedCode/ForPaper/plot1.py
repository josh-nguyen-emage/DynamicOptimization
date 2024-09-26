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
    returnVal = read_file("D:\\1 - Study\\6 - DTW_project\\Container\\Log_Run_Bayes_1-7.txt")
    # print(returnVal)
    minIdx = 0
    minMSE = 1000000
    container = []
    eachPoint = [[],[],[]]
    fig = plt.figure()
    ax = fig.add_subplot(111)

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
        eachPoint[0].append(param[0])
        eachPoint[1].append(param[9])
        eachPoint[2].append(MSE)
        if (idx % 16 == 15):
            ax.scatter([(idx+1)/16]*len(eachPoint[0]),eachPoint[0],label=str(idx+1))
            eachPoint = [[],[],[]]

        if idx == 640:
            break

    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.set_title("Change log of E parameter")
    plt.show()