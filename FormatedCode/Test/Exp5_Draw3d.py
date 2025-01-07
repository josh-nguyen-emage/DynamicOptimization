import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))

from Libary.function import findF
from Phase1 import read_file
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def draw3dPlt(x,y,val):
    grid_x, grid_y = np.mgrid[0:1:0.01, 0:1:0.01]

    # Interpolate the values over the grid
    grid_z = griddata((x, y), val, (grid_x, grid_y), method='cubic')

    # Plotting
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')

    # Add the original data points
    # ax.scatter(x, y, allMSE, c='red', label='Data Points')

    # Labels and title
    ax.set_title('3D Interpolated Surface')
    ax.set_xlabel('K1')
    ax.set_ylabel('E')
    ax.set_zlabel('Interpolated Value')

    # Color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Interpolated Value')

    plt.legend()
    plt.show()

# [0.84469084, 1, 0.30963334, 0.55527287, 0.43204053, 1, 1, 0.13892581, 0, 0.9941866,  1]
def changedValue(param):
    orgValue = np.array([0.84469084, 1, 0.30963334, 0.55527287, 0.43204053, 1, 1, 0.13892581, 0, 0.9941866, 1])
    offset = abs(np.array(param) - orgValue)
    sorted_indices = np.argsort(offset)[-2:]
    sorted_indices = np.sort(sorted_indices)
    return sorted_indices[0], sorted_indices[1]

returnVal = read_file("D:\\1 - Study\\6 - DTW_project\\Container\\BurningTest_3d_13_9.txt")

param = np.array(returnVal[0])
strain = np.array(returnVal[1])/(-4)
stress = np.array(returnVal[2])/(300)
bodyOpen = np.array(returnVal[3])/(20)

param = param[:3000]

allMSE = []
listValue = [[],[]]

# 0 1 4 5 7 10

firstIdx = 5
secondIdx = 7

print(allMSE)

for firstIdx in [0,1,4,5,7,10]:
    for secondIdx in [0,1,4,5,7,10]:
        if (firstIdx >= secondIdx):
            continue
        allMSE = []
        listValue = [[],[]] 

        for idx in range(len(param)):
            MSE, interpolate = findF(stress[idx], bodyOpen[idx], strain[idx])

            changedPosition = changedValue(param[idx])
            if (changedPosition[0] == firstIdx and changedPosition[1] == secondIdx):
                allMSE.append(MSE)
                listValue[0].append(param[idx][changedPosition[0]])
                listValue[1].append(param[idx][changedPosition[1]])

            # if (MSE == 0):
            #     print(idx,param[idx][changedPosition[0]],param[idx][changedPosition[1]])
            # if (idx%1000 == 0):
            #     print(idx)

        paramName = ["K1","C1","C3","C4","C5","C7","C8","C10","C11","C12","E"]
        grid_x, grid_y = np.mgrid[0:1:0.01, 0:1:0.01]

        # Interpolate the values over the grid
        grid_z = griddata((listValue[0], listValue[1]), allMSE, (grid_x, grid_y), method='cubic')

        # Plotting
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')

        # Labels and title
        ax.set_title('MSE Surface')
        ax.set_xlabel(paramName[firstIdx])
        ax.set_ylabel(paramName[secondIdx])
        ax.set_zlabel('MSE Value')

        # Color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Interpolated Value')

        plt.legend()
        # plt.savefig(str(firstIdx)+"-"+str(secondIdx)+".png")
        plt.show()


