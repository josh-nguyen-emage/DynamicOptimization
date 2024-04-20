# save_to_file(input_list, output_list, 'output.txt')

import random

from GenerateInpFile import changeInpFile
from function import *
from RunSequent1 import *

def RunSimulationThread(idx):
    while True:
        time.sleep(np.random.rand()*5)
        E = np.random.randint(35,50)
        S = np.random.randint(50,200)
        changeInpFile(E,S,idx)
        printWithTime("Start Simulation")
        runTime = RunSimulation_timeCheck(idx)
        save_to_file([idx,E,S],runTime,"Log_Time_0.txt")

# RunSimulationThread(0)

