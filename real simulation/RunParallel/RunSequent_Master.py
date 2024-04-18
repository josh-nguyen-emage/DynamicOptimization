# save_to_file(input_list, output_list, 'output.txt')

import random

from function import *
from RunSequent1 import *
from RunSequent2 import *
from RunSequent3 import *

def RunSimulationThread(idx):
    inputData = [random.random() for _ in range(10)]
    WriteParameter_K1_Only(inputData,idx)
    printWithTime("Start Simulation")
    RunSimulation(idx)
    printWithTime("Start Extract Result")
    RunTool4Atena(idx)
    outputData = ExtractResult(idx)
    save_to_file(inputData,outputData,"Log_Run_B_"+str(idx)+"_1_0604.txt")

