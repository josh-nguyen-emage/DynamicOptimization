import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))

from Libary.function import *
from Phase1 import read_file
import numpy as np
import matplotlib.pyplot as plt

def write_arrays_to_file(arr1, arr2, txt_file):
    with open(txt_file, 'w') as f:
        for a, b in zip(arr1, arr2):
            f.write(f"{a} {b}\n")

returnVal = read_file("D:\\1 - Study\\6 - DTW_project\\Container\\FDSE2_1223.txt")
# returnVal = read_file("D:\\1 - Study\\6 - DTW_project\\Container\\BurningTest_3dDraw_10-8.txt")

param = np.array(returnVal[0])
# strain = np.array(returnVal[1])/(-4)
# stress = np.array(returnVal[2])/(300)
# bodyOpen = np.array(returnVal[3])/(20)
strain = np.array(returnVal[1])
stress = np.array(returnVal[2])
bodyOpen = np.array(returnVal[3])

# param = param[:16*32]

allMSE = []
stress_exp,strain_exp,bodyOpen_exp = getExpData()


for idx in range(len(param)):
    MSE, interpolate = findF(stress[idx], bodyOpen[idx], strain[idx])
    allMSE.append(MSE)

    # if (MSE < 50):
    #     print(idx)
    #     plt.scatter(bodyOpen[idx],stress[idx],label='Simulate Line')
    #     plt.scatter(strain[idx],stress[idx],label='Simulate Line')
    #     plt.scatter(np.concatenate((np.flip(strain_exp),bodyOpen_exp)),getExpectChart(),label='Experiment line')
    #     plt.title('Run ' + str(idx))
    #     WriteParameter(param[idx],0)
    #     # Show plot
    #     # plt.legend()
    #     plt.show()

minIdx = allMSE.index(min(allMSE))
print("min idx",minIdx)
a = param[minIdx]
WriteParameter(a,0)
MSE, interpolate = findF(stress[minIdx], bodyOpen[minIdx], strain[minIdx])
# plt.scatter(bodyOpen[minIdx],stress[minIdx],label='Simulate Line')
plt.plot(-1*strain[minIdx],stress[minIdx],'o-',label='Simulate Line')
plt.plot(-1*strain_exp,getExpectChart(),'o-',label='Real Experiment')
# plt.plot(-1*strain_exp,interpolate,'o-',label='Atena Simulation')

plt.title('Run ' + str(minIdx))
plt.xlabel("ε (‰)")
plt.ylabel("σ MPa")
# Show plot
plt.legend()
plt.show()

# write_arrays_to_file(-1*strain_exp,interpolate,"Atena Sim FDSE2 P2.txt")
# write_arrays_to_file(-1*strain_exp,getExpectChart(),"Real Experiment FDSE2 P2.txt")

print(min(allMSE),calculate_correlation(interpolate,getExpectChart()))
    
# plt.scatter(range(len(allMSE)),allMSE)
# plt.xlabel('Run times')
# plt.ylabel('MSE')
# plt.title('Bayes - AI model')
# # Show plot
# plt.show()

write_arrays_to_txt("FDSEV2 P3.txt",[-1*strain_exp,getExpectChart()],[-1*strain[minIdx],stress[minIdx]])
# np.savetxt("FDSEV2 P2.txt", [-1*strain_exp,getExpectChart(),-1*strain_exp,interpolate], fmt="%f")


