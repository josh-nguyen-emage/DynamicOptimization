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
    if os.path.exists(pathIdx(idx)+'G7-Cyl-Trial-1_NODES_STRAIN.atf'):
        os.remove(pathIdx(idx)+'G7-Cyl-Trial-1_NODES_STRAIN.atf')
        os.remove(pathIdx(idx)+'G7-Cyl-Trial-1_NODES_STRESS.atf')
    cwd = "C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\AtenaPool\\"+str(idx)
    command = ["C:\\Program Files (x86)\\CervenkaConsulting\\AtenaV5\\AtenaConsole.exe", "H:\\02.Working-Thinh\\ATENA-WORKING\\Post.atn"]
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
    return [np.array(strainVal), -1*np.array(stressVal)]

def RunSimulationThread(idx, inputData):
    WriteParameter(inputData,idx)
    RunSimulation(idx)
    RunTool4Atena(idx)
    outputData = ExtractResult(idx)
    # save_to_file(inputData,outputData,"RunLog\\Log_Run_G_Phase1.txt")
    return outputData