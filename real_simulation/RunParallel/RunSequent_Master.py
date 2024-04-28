

from real_simulation.RunParallel.RunSequent1 import RunSimulation
from real_simulation.RunParallel.RunSequent2 import RunTool4Atena
from real_simulation.RunParallel.RunSequent3 import ExtractResult
from real_simulation.RunParallel.function import WriteParameter, save_to_file


def RunSimulationThread(idx, inputData):
    WriteParameter(inputData,idx)
    RunSimulation(idx)
    RunTool4Atena(idx)
    outputData = ExtractResult(idx)
    save_to_file(inputData,outputData,"RunLog\\Log_Run_E_Phase1.txt")
    return outputData

