import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))

from Libary.function import WriteParameter, findF, getExpData, getExpectChart
from Phase1 import read_file
import numpy as np
import matplotlib.pyplot as plt


returnVal = read_file("D:\\1 - Study\\6 - DTW_project\\Container\\FDSE2_1212.txt")
# returnVal = read_file("D:\\1 - Study\\6 - DTW_project\\Container\\BurningTest_3dDraw_10-8.txt")

param = np.array(returnVal[0])
# strain = np.array(returnVal[1])/(-4)
# stress = np.array(returnVal[2])/(300)
# bodyOpen = np.array(returnVal[3])/(20)
strain = np.array(returnVal[1])
stress = np.array(returnVal[2])
bodyOpen = np.array(returnVal[3])

param = param[:16*32]

allMSE = []
stress_exp,strain_exp,bodyOpen_exp = getExpData()

for idx in range(len(param)):
    MSE, interpolate = findF(stress[idx], bodyOpen[idx], strain[idx])
    allMSE.append(MSE)

    if (MSE < 50):
        print(idx)
        # plt.scatter(bodyOpen[idx],stress[idx],label='Simulate Line')
        plt.scatter(strain[idx],stress[idx],label='Simulate Line')
        plt.scatter(strain_exp,getExpectChart(),label='Experiment line')
        plt.title('Run ' + str(idx))
        # Show plot
        # plt.legend()
        plt.show()
    

print(min(allMSE),stress[idx], bodyOpen[idx], strain[idx])
    
plt.scatter(range(len(allMSE)),allMSE)

plt.xlabel('Run times')
plt.ylabel('MSE')
plt.title('Bayes - AI model')

# Show plot
plt.show()

np.savetxt("tmpResult.txt", [stress[idx], bodyOpen[idx], strain[idx]], fmt="%f")


