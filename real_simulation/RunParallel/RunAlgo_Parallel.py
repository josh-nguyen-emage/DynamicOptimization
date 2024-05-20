
from function import *

from scipy.interpolate import interp1d
from scipy.optimize import minimize

from real_simulation.RunParallel.RunSequent1 import RunSimulation
from real_simulation.RunParallel.RunSequent2 import RunTool4Atena
from real_simulation.RunParallel.RunSequent3 import ExtractResult
from real_simulation.try_trainModel import ReadLabFile

def interpolate_line(x_values, y_values, X_interpolate):
    """
    Interpolates y values for given x coordinates using linear interpolation.

    Parameters:
        x_values (list): List of x coordinates of the line.
        y_values (list): List of y coordinates of the line.
        X_interpolate (list): List of x coordinates where y values are interpolated.

    Returns:
        list: List of interpolated y values corresponding to X_interpolate.
    """
    # Perform linear interpolation
    interp_func = interp1d(x_values[:50], y_values[:50], kind='cubic', bounds_error=False)

    # Find y values for X_interpolate using interpolation function
    Y_interpolate = interp_func(X_interpolate)

    return Y_interpolate

def findF(predictY, predictZ):
    global Y_exp
    global Z_exp
    Z_perdict_expBase = interpolate_line(predictY, predictZ,Y_exp)
    sumSquare = (Z_perdict_expBase-Z_exp)**2
    return np.nanmean(sumSquare)


filename = 'stdFile\G7-Uni-AxialTest.dat'  # Replace 'data.txt' with your file path
list_a, list_b, list_c = ReadLabFile(filename)
list_c = np.array(list_c)*(-1000)
list_a = np.array(list_a)*1

Y_exp = list_c
Z_exp = list_a


class DTW:
    # Constructor method (initializer)
    def __init__(self, method, index):
        # Instance attributes
        self.method = method
        self.index = index

    # Instance method
    def objective_function(self,params):
        params = np.clip(params,0,1)
        WriteParameter(params, self.index)
        RunSimulation(self.index)
        RunTool4Atena(self.index)
        outputData = ExtractResult(self.index)
        save_to_file(params,outputData,"Log_Run_F_"+self.method+".txt")
        # save_to_file(params,outputData,"Log_Run_debug.txt")
        strain = np.array(outputData[0])
        stress = -1*np.array(outputData[1])
        MSE = findF(strain,stress[1:52])
        print("curent MSE of",self.method,":",MSE)
        return MSE

    # Instance method
    def speak(self, sound):
        return f"{self.name} says {sound}"



# --------------------------
def RunAlgo(index):
    initParam = [0.5]*11
    initParam = [0.20398719096599705,0.9553720675791246,0.7055419216098967,0.9941247570413594,0.8911491304079948,0.8929546254803749,0.9816650429194598,0.5469467002382876,0.8414270051383659,0.705789405032828,0.3891777396691328]
    methodList = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr']
    dtw = DTW(methodList[index],index)
    bounds = [(0, 1)] * 11
    result = minimize(dtw.objective_function, initParam, method=methodList[index], bounds=bounds)
    # print(result.x)
    # with open("result_"+methodList[index]+".txt", 'w') as file:
    #     for number in result.x:
    #         file.write(f"{number}\n")
    print(f"{methodList[index]} Run Completed")

if __name__ == "__main__":
    RunAlgo(0)

