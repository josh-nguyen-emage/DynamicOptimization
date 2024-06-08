import os
import time
import subprocess

import numpy as np

from Libary.function import *

def RunSimulation(idx):
    cwd = "C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\AtenaPool\\"+str(idx)
    command = ["C:\\Program Files (x86)\\CervenkaConsulting\\AtenaV5\\AtenaConsole.exe", cwd + "\\G7-Cyl-Trial-1.inp", "a.out", "a.msg", "a.err"]
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
    process.wait()

def RunTool4Atena(idx):
    if os.path.exists(pathIdx(idx)+'G7-Cyl-Trial-1_NODES_REACTIONS.atf'):
        os.remove(pathIdx(idx)+'G7-Cyl-Trial-1_NODES_REACTIONS.atf')
        os.remove(pathIdx(idx)+'G7-Cyl-Trial-1_NODES_STRESS.atf')
        os.remove(pathIdx(idx)+'G7-Cyl-Trial-1_NODES_DISPLACEMENTS.atf')
    cwd = "C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\AtenaPool\\"+str(idx)
    command = ["C:\\Program Files (x86)\\CervenkaConsulting\\AtenaV5\\AtenaConsole.exe", "C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\DynamicOptimization-ST\\Container\\stdFile\\Post_Exp1.atn"]
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
    process.wait()

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

def extractFile(nodeList,fileName,idx, columnIdx):
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
                    currentSum += float(filtered_list[columnIdx])
                    
            except:
                pass    
            if "-----" in line:
                dataExtracted.append(currentSum/len(nodeList))
                currentSum = 0
    return dataExtracted


def ExtractResult(idx):
    nodeList = read_integers_from_file("C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\DynamicOptimization-ST\\Container\\stdFile\\NodeList_Mid.txt")
    nodeCenter = read_integers_from_file("C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\DynamicOptimization-ST\\Container\\stdFile\\NodeList_bodyOpen.txt")
    strainVal = extractFile(nodeList,"G7-Cyl-Trial-1_NODES_DISPLACEMENTS.atf",idx,2)
    stressVal = extractFile(nodeList,"G7-Cyl-Trial-1_NODES_STRESS.atf",idx,2)
    midStrainVal = extractFile(nodeCenter,"G7-Cyl-Trial-1_NODES_DISPLACEMENTS.atf",idx,2)
    return [1000*1000*np.array(strainVal)[0:51]/150, -1*np.array(stressVal)[0:51], -1000*np.array(midStrainVal)[0:51]/0.3]

def RunSimulationThread(idx, inputData):
    # WriteParameter(inputData,idx)
    # RunSimulation(idx)
    # RunTool4Atena(idx)
    outputData = ExtractResult(idx)
    save_to_file(inputData,outputData,"C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\DynamicOptimization-ST\\Container\\Log_Run_B_A_Phase1.txt")
    return outputData