import sys
import numpy as np

from real_simulation.GlobalLib import pathIdx, pathName

def read_integers_from_file(txt_path):
    # Initialize an empty list to store integers
    integers = []
    
    # Open the file in read mode
    with open(txt_path, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Strip newline characters and convert the line to an integer
            integer = int(line.strip())
            # Append the integer to the list
            integers.append(integer)
    
    # Convert the list of integers to a numpy array
    integers_array = np.array(integers) 
    
    return integers_array

def extractFile(nodeList,fileName,idx):
    # Open the file in read mode
    dataExtracted = []

    with open(pathIdx(idx)+fileName, 'r') as file:
        # Iterate through each line in the file
        currentSum = 0
        for line in file:
            currentLine = line.strip().split(" ")
            filtered_list = [string for string in currentLine if len(string) != 0]
            try:
                if (int(filtered_list[0]) in nodeList):
                    currentSum += float(filtered_list[2])
                    
            except:
                pass
            if "-----" in line:
                dataExtracted.append(currentSum/len(nodeList))
                currentSum = 0
    return dataExtracted


def ExtractResult(idx):

    nodeList = read_integers_from_file(pathName+'NodeList_Mid.txt')

    strainVal = extractFile(nodeList,"G7-Cyl-Trial-1_NODES_STRAIN.atf",idx)
    stressVal = extractFile(nodeList,"G7-Cyl-Trial-1_NODES_STRESS.atf",idx)

    

    # for num in dataExtracted:
    #     print(num)

    return [strainVal, stressVal]

# ExtractResult()

