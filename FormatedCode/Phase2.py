import threading
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from Libary.RunSequent import *
from Libary.function import *

class DTW:
    def __init__(self, method, index):
        self.method = method
        self.index = index

    def objective_function(self,params):
        params = np.clip(params,0,1)
        WriteParameter(params, self.index)
        RunSimulation(self.index)
        RunTool4Atena(self.index)
        outputData = ExtractResult(self.index)
        # save_to_file(params,outputData,"Log_Run_F_"+self.method+".txt")
        strain = outputData[0]
        stress = outputData[1]
        MSE = findF(strain,stress)
        print("curent MSE of",self.method,":",MSE)
        return MSE

def RunAlgo(index):
    initParam = [0.5]*11
    methodList = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr']
    dtw = DTW(methodList[index],index)
    bounds = [(0, 1)] * 11
    result = minimize(dtw.objective_function, initParam, method=methodList[index], bounds=bounds)
    print(f"{methodList[index]} Run Completed")

if __name__ == "__main__":

    timeColector = []
    numThread = 9


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

