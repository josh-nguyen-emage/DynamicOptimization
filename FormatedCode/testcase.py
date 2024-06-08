from Phase1 import *
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":
    returnVal = RunSimulationThread(16,np.random.rand(11))
    # print(returnVal)
    strain = returnVal[0]
    stress = returnVal[1]
    bodyOpen = returnVal[2]
    MSE, interpolate = findF(stress, bodyOpen, strain)

    print(MSE)

    plt.plot(np.concatenate((np.flip(strain),bodyOpen)) , np.concatenate((np.flip(stress),stress)), label = "Simulation")
    # plt.plot(np.concatenate((np.flip(strain_exp),bodyOpen_exp)) , interpolate, marker = "x", label = "Interpolate")
    plt.plot(strain_exp , stress_exp, label = "Experiment",color="green")
    plt.plot(bodyOpen_exp , stress_exp,color="green")

    # Add titles and labels
    plt.title('Example')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()