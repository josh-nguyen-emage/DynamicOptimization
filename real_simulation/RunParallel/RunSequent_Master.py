# save_to_file(input_list, output_list, 'output.txt')

import random

from function import *
from RunSequent1 import *
from RunSequent2 import *
from RunSequent3 import *

def RunSimulationThread(idx, inputData):
    WriteParameter(inputData,idx)
    RunSimulation(idx)
    RunTool4Atena(idx)
    outputData = ExtractResult(idx)
    save_to_file(inputData,outputData,"RunLog\\Log_Run_E_Phase1.txt")
    return outputData

