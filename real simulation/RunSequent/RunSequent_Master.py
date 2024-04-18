# save_to_file(input_list, output_list, 'output.txt')

import random

from function import *
from RunSequent1 import *
from RunSequent2 import *
from RunSequent3 import *

from scipy.optimize import minimize


# -------------------------- 
def randomRun():
    counter = 1
    while True:
        print("---",counter,"---")
        inputData = [random.random() for _ in range(11)]
        WriteParameter(inputData)
        printWithTime("Start Simulation")
        RunSimulation()
        printWithTime("Start Extract Result")
        RunTool4Atena()
        outputData = ExtractResult()
        save_to_file(inputData,outputData,"Log_Run_A_1_0304.txt")
        printWithTime("Run Completed")

        counter += 1
        current_time = time.localtime()
        if current_time.tm_hour == 6:
            print("Run Completed",counter,"times")
        else:
            print("Run in hours ",current_time.tm_hour)


# --------------------------
def RunAlgo(initParam):
    counter = 1
    while True:
        print("---",counter,"---")
        inputData = [random.random() for _ in range(11)]
        WriteParameter(inputData)
        printWithTime("Start Simulation")
        RunSimulation()
        printWithTime("Start Extract Result")
        RunTool4Atena()
        outputData = ExtractResult()
        save_to_file(inputData,outputData,"Log_Run_A_1_0304.txt")
        printWithTime("Run Completed")

        counter += 1
        current_time = time.localtime()
        if current_time.tm_hour == 6:
            print("Run Completed",counter,"times")
        else:
            print("Run in hours ",current_time.tm_hour)


