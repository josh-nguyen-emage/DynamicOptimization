# save_to_file(input_list, output_list, 'output.txt')

import random

from GenerateInpFile import changeInpFile
from function import *
from RunSequent1 import *

def RunSimulationThread(idx):
    E = np.random.randint(35,50)
    S = np.random.randint(50,200)
    changeInpFile(E,S,idx)
    printWithTime("Start Simulation")
    time = RunSimulation_timeCheck(idx)
    save_to_file([E,S],time,"Log_Time_0.txt")

