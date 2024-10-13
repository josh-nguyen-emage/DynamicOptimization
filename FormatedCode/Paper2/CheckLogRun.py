import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))

from Libary.function import findF, getExpData, getExpectChart
from Phase1 import read_file
import numpy as np
import matplotlib.pyplot as plt


returnVal = read_file("D:\\1 - Study\\6 - DTW_project\\Container\\Log_Run_Bayes_AI_2-7.txt")
# returnVal = read_file("D:\\1 - Study\\6 - DTW_project\\Container\\BurningTest_3dDraw_10-8.txt")

param = np.array(returnVal[0])
# strain = np.array(returnVal[1])/(-4)
# stress = np.array(returnVal[2])/(300)
# bodyOpen = np.array(returnVal[3])/(20)
strain = np.array(returnVal[1])
stress = np.array(returnVal[2])
bodyOpen = np.array(returnVal[3])

param = param[:3000]

allMSE = []
stress_exp,strain_exp,bodyOpen_exp = getExpData()

for idx in range(len(param)):
    MSE, interpolate = findF(stress[idx], bodyOpen[idx], strain[idx])
    allMSE.append(MSE)

    if (MSE == 0):
        print(idx)
        # plt.scatter(bodyOpen[idx],stress[idx],label='real Line')
        # plt.scatter(strain[idx],stress[idx],label='real Line')
        # plt.scatter(np.concatenate((np.flip(strain_exp),bodyOpen_exp)),getExpectChart(),label='interpolate Line')

        # # Show plot
        # plt.legend()
        # plt.show()
    

print(min(allMSE),stress[idx], bodyOpen[idx], strain[idx])
    
plt.scatter(range(len(allMSE)),allMSE)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot Example')

# Show plot
plt.show()


