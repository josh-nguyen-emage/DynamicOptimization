# save_to_file(input_list, output_list, 'output.txt')

import random

from function import *
from RunSequent1 import *
from RunSequent2 import *
from RunSequent3 import *

from scipy.interpolate import interp1d
from scipy.optimize import minimize

def calculate_covariance(list1, list2):
    covariance_matrix = np.cov(list1, list2)
    covariance = covariance_matrix[0, 1]  # Covariance is the off-diagonal element of the covariance matrix
    return covariance

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
    interp_func = interp1d(x_values, y_values, kind='cubic', bounds_error=False)

    # Find y values for X_interpolate using interpolation function
    Y_interpolate = interp_func(X_interpolate)

    return Y_interpolate

def findF(predictY, predictZ):
    global Y_exp
    global Z_exp

    Z_perdict_expBase = interpolate_line(predictY, predictZ,Y_exp)

    sumSquare = (Z_perdict_expBase-Z_exp)**2

    return np.nanmean(sumSquare), Z_perdict_expBase


filename = 'G7-Uni-AxialTest.dat'  # Replace 'data.txt' with your file path
list_a, list_b, list_c = ReadLabFile(filename)
list_c = np.array(list_c)*(-1000)
list_a = np.array(list_a)*1

Y_exp = list_c
Z_exp = list_a

def objective_function(params):
    params = np.clip(params,0,1)
    WriteParameter(params)
    RunSimulation()
    RunTool4Atena()
    outputData = ExtractResult()
    save_to_file(params,outputData,"Log_Run_B.txt")
    MSE = findF(-1000*outputData[0],-1*outputData[1])
    return MSE

# --------------------------
def RunAlgo(initParam):
    result = minimize(objective_function, initParam, method='SLSQP')
    print(result.x)

RunAlgo([0.5]*11)


