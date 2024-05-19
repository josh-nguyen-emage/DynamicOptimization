import numpy as np
from scipy.interpolate import interp1d

from real_simulation.try_trainModel import ReadLabFile

pathName = "H:\\02.Working-Thinh\\ATENA-WORKING\\"

def pathIdx(idx):
    return "C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\AtenaPool\\"+str(idx)+"\\"

filename = 'stdFile\G7-Uni-AxialTest.dat'  # Replace 'data.txt' with your file path
list_a, list_b, list_c = ReadLabFile(filename)
list_c = np.array(list_c)*(-1000)
list_a = np.array(list_a)

Y_exp = list_c
Z_exp = list_a

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
    Z_perdict_expBase[0] = Z_exp[0]
    sumSquare = (Z_perdict_expBase-Z_exp)**2

    return np.nanmean(sumSquare), Z_perdict_expBase

def getExpectChart():
    global Z_exp
    return Z_exp



# ------------------------------------------




