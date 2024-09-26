import matplotlib.pyplot as plt
import numpy as np

import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))

from Libary.RunSequent import read_integers_from_file

def extractFile(nodeList,fileName, columnIdx, value = "avg"):
    # Open the file in read mode
    dataExtracted = []

    with open(fileName, 'r') as file:
        # Iterate through each line in the file
        currentSum = 0
        for line in file:
            currentLine = line.strip().split(" ")
            filtered_list = [string for string in currentLine if len(string) != 0]
            try:
                if (int(filtered_list[0]) in nodeList):
                    currentSum += float(filtered_list[columnIdx])
                    
            except:
                pass    
            if "Step" in line:
                if (value == "avg"):
                    dataExtracted.append(currentSum/len(nodeList))
                else:
                    dataExtracted.append(currentSum)
                currentSum = 0
    return np.array(dataExtracted)

def ReadLabFile(filename):
    list_a = []
    list_b = []

    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split()
            if len(values) != 2:
                print(f"Ignoring line: {line.strip()}. Expected 3 values per line.")
                continue
            try:
                a, b= map(float, values)
                list_a.append(a)
                list_b.append(b)
            except ValueError:
                print(f"Ignoring line: {line.strip()}. Could not convert values to floats.")

    return list_a, list_b

sim_x = [0]
sim_y = [0]

node_suport = read_integers_from_file("D:\\1 - Study\\6 - DTW_project\\Container\\stdFile\\NodeList-At-RightSupport.txt")
node_top = read_integers_from_file("D:\\1 - Study\\6 - DTW_project\\Container\\stdFile\\NodeList-DISP-At-Top.txt")
sim_x = extractFile(node_top,"D:\\1 - Study\\6 - DTW_project\\Container\\UHPC-RILEM-2024-V5LZ-M7_NODES_DISPLACEMENTS.atf" ,2)*-1000
sim_y = extractFile(node_suport,"D:\\1 - Study\\6 - DTW_project\\Container\\UHPC-RILEM-2024-V5LZ-M7_NODES_REACTIONS.atf",2,"sum")*352

realX, realY = ReadLabFile("D:\\1 - Study\\6 - DTW_project\\Container\\stdFile\\G7-RILEM-BeamTest.dat")

realY = np.array(realY)

# realY = realY /1000*352

print(sim_x[:10])
    
plt.plot(sim_x,sim_y,color="blue",label="Simulation")
plt.scatter(sim_x,sim_y,color="blue")

plt.plot(realX, realY,color="green",label="Experiment")
plt.scatter(realX, realY,color="green")
# plt.ylim(0, 20)
# plt.xlim(0, 10)
plt.ylabel("Stress (MPa)")
plt.xlabel("Displacement (mm)")
plt.title("Rilem M7 std param")
plt.legend()
plt.grid(True)
plt.show()

