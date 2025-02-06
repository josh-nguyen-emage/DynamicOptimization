import threading
from scipy.interpolate import interp1d
from scipy.optimize import minimize

import sys, os

sys.path.append(os.path.abspath(os.path.join('.')))

from Libary.RunSequent import *
from Libary.function import *

class DTW:
    def __init__(self, method, index):
        self.method = method
        self.index = index

    def objective_function(self,params):
        params = np.clip(params,0,1)
        simulation_result = RunSimulationThread(self.index, params)
        strain = simulation_result[0]
        stress = simulation_result[1]
        bodyOpen = simulation_result[2]
        feature, interpolate = findF(stress, bodyOpen, strain)
        save_to_file(params,simulation_result,"Run_A_"+self.method+".txt")
        print("curent MSE of",self.method,":",feature)
        return feature

def RunAlgo(index):
    initParam = [0.35309832, 0.42692865, 0.38362842, 0.35467606, 0.72721566,
       0.08852128, 0.63023561, 0.35069087, 0.06886009, 0.67790651,
       0.72799781]
    methodList = ['Nelder-Mead']
    dtw = DTW(methodList[index],index)
    bounds = [(0, 1)] * 11
    result = minimize(dtw.objective_function, initParam, method=methodList[index], bounds=bounds)
    print(f"{methodList[index]} Run Completed")

if __name__ == "__main__":

    timeColector = []
    numThread = 1


    # Create a list to hold the thread objects
    threads = []
    for idx in range(numThread):
        # Create a thread for each index and pass the index as an argument to the function
        thread = threading.Thread(target=RunAlgo, args=(idx,))
        # Start the thread
        thread.start()
        # Add the thread object to the list
        threads.append(thread)

    # Main thread waits for all threads to complete
    for thread in threads:
        thread.join()

