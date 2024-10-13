import matplotlib.pyplot as plt
import numpy as np

import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))

# Function to convert string to NumPy array
def string_to_numpy_array(string):
    return np.fromstring(string.replace('[', '').replace(']', ''), sep=',')

# Function to read the file and convert the content to a NumPy array
def read_txt_and_convert_to_numpy(file_path):
    # Read the file content as a single string
    with open(file_path, 'r') as file:
        file_content = file.read()
    
    # Convert the string to a NumPy array
    allSim = file_content.split("\n\n")
    finalData = []
    for eachSim in allSim:
        if len(eachSim) == 0:
            continue
        param = eachSim.split(":")[0].split(" ")
        value = string_to_numpy_array(eachSim.split(":")[1]).reshape(2,202)
        finalData.append([param,value])

    return finalData

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

# Example usage
file_path = "D:/1 - Study/6 - DTW_project/Container/Rilem_26-9.txt"  # Replace with the path to your file
numpy_array = read_txt_and_convert_to_numpy(file_path)

print("Converted NumPy array:", numpy_array[1][1])

realX, realY = ReadLabFile("D:\\1 - Study\\6 - DTW_project\\Container\\stdFile\\G7-RILEM-BeamTest.dat")

realY = np.array(realY)

for i in range(len(numpy_array)):

    plt.plot(numpy_array[i][1][0],numpy_array[i][1][1],color="blue",label="Simulation")
    plt.scatter(numpy_array[i][1][0],numpy_array[i][1][1],color="blue")

    plt.plot(realX, realY,color="green",label="Experiment")
    plt.scatter(realX, realY,color="green")
    # plt.ylim(0, 20)
    # plt.xlim(0, 10)
    plt.ylabel("Stress (MPa)")
    plt.xlabel("Displacement (mm)")
    plt.title("Rilem M7 Random param "+str(i))
    plt.legend()
    plt.grid(True)
    plt.show()