import os, sys
import subprocess
sys.path.append(os.path.abspath(os.path.join('.')))
import numpy as np

from Libary.function import *

def RunSimulation(idx):
    cwd = "G:\\2.Working-Thinh\\AtenaPool\\"+str(idx)
    command = ["C:\\Program Files (x86)\\CervenkaConsulting\\AtenaV5\\AtenaConsole.exe", cwd + "\\G7-Cyl-Trial-1.inp", "a.out", "a.msg", "a.err"]
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
    process.wait()

def RunSimulationRilem(idx):
    cwd = "G:\\2.Working-Thinh\\AtenaPool\\"+str(idx)
    command = ["C:\\Program Files (x86)\\CervenkaConsulting\\AtenaV5\\AtenaConsole.exe", cwd + "\\UHPC-RILEM-2024-V5LZ-M7.inp", "a.out", "a.msg", "a.err"]
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
    process.wait()

def RunTool4Atena(idx):
    if os.path.exists(pathIdx(idx)+'G7-Cyl-Trial-1_NODES_REACTIONS.atf'):
        os.remove(pathIdx(idx)+'G7-Cyl-Trial-1_NODES_REACTIONS.atf')
        os.remove(pathIdx(idx)+'G7-Cyl-Trial-1_NODES_STRESS.atf')
        os.remove(pathIdx(idx)+'G7-Cyl-Trial-1_NODES_DISPLACEMENTS.atf')
        os.remove(pathIdx(idx)+'a.err')
        os.remove(pathIdx(idx)+'a.msg')
        os.remove(pathIdx(idx)+'a.out')
    cwd = "G:\\2.Working-Thinh\\AtenaPool\\"+str(idx)
    command = ["C:\\Program Files (x86)\\CervenkaConsulting\\AtenaV5\\AtenaConsole.exe", "G:\\2.Working-Thinh\\DynamicOptimization-ST\\Container\\stdFile\\Post_Exp1.atn"]
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
    process.wait()

def RunTool4AtenaRilem(idx):
    if os.path.exists(pathIdx(idx)+'UHPC-RILEM-2024-V5LZ-M7_NODES_REACTIONS.atf'):
        os.remove(pathIdx(idx)+'UHPC-RILEM-2024-V5LZ-M7_NODES_REACTIONS.atf')
        os.remove(pathIdx(idx)+'UHPC-RILEM-2024-V5LZ-M7_NODES_DISPLACEMENTS.atf')
        os.remove(pathIdx(idx)+'a.err')
        os.remove(pathIdx(idx)+'a.msg')
        os.remove(pathIdx(idx)+'a.out')
    cwd = "G:\\2.Working-Thinh\\AtenaPool\\"+str(idx)
    command = ["C:\\Program Files (x86)\\CervenkaConsulting\\AtenaV5\\AtenaConsole.exe", "G:\\2.Working-Thinh\\DynamicOptimization-ST\\Container\\stdFile\\Post_Rilem.atn"]
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

def extractFile(nodeList,fileName,idx, columnIdx, value = "avg"):
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
            if "Step" in line:
                if (value == "avg"):
                    dataExtracted.append(currentSum/len(nodeList))
                else:
                    dataExtracted.append(currentSum)
                currentSum = 0
    return np.array(dataExtracted)


def ExtractResult(idx):
    nodeList = read_integers_from_file("G:\\2.Working-Thinh\\DynamicOptimization-ST\\Container\\stdFile\\NodeList_Mid.txt")
    nodeCenter = read_integers_from_file("G:\\2.Working-Thinh\\DynamicOptimization-ST\\Container\\stdFile\\NodeList_bodyOpen.txt")
    strainVal = extractFile(nodeList,"G7-Cyl-Trial-1_NODES_DISPLACEMENTS.atf",idx,2)
    stressVal = extractFile(nodeList,"G7-Cyl-Trial-1_NODES_STRESS.atf",idx,2)
    midStrainVal = extractFile(nodeCenter,"G7-Cyl-Trial-1_NODES_DISPLACEMENTS.atf",idx,1)
    return [1000*1000*np.array(strainVal)[0:51]/150, -1*np.array(stressVal)[0:51], 1000*np.array(midStrainVal)[0:51]/0.075]

def ExtractResultRilem(idx):
    node_suport = read_integers_from_file("G:\\2.Working-Thinh\\DynamicOptimization-ST\\Container\\stdFile\\NodeList-At-RightSupport.txt")
    node_top = read_integers_from_file("G:\\2.Working-Thinh\\DynamicOptimization-ST\\Container\\stdFile\\NodeList-DISP-At-Top.txt")
    sim_x = extractFile(node_top,"UHPC-RILEM-2024-V5LZ-M7_NODES_DISPLACEMENTS.atf" , idx,2)
    sim_y = extractFile(node_suport,"UHPC-RILEM-2024-V5LZ-M7_NODES_REACTIONS.atf", idx,2,"sum")
    return [sim_x*-1000,sim_y*352]


def RunSimulationThread(idx, inputData):
    WriteParameter(inputData,idx)
    RunSimulation(idx)
    RunTool4Atena(idx)
    outputData = ExtractResult(idx)
    save_to_file(inputData,outputData,"G:\\2.Working-Thinh\\DynamicOptimization-ST\\Container\\BurningTest_3dDraw_10-8.txt")
    return outputData

def RunSimulationRilemThread(idx, inputData):
    WriteParameterRilem(inputData,idx)
    RunSimulationRilem(idx)
    RunTool4AtenaRilem(idx)
    outputData = ExtractResultRilem(idx)
    save_to_file_rilem(inputData,outputData,"G:\\2.Working-Thinh\\DynamicOptimization-ST\\Container\\BurningTest_3dDraw_10-8.txt")
    return outputData