from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from try_trainModel import ReadLabFile, read_file

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

    Z_perdict_expBase = interpolate_line(predictY[:50], predictZ[:50],Y_exp)
    sumSquare = (Z_perdict_expBase-Z_exp)**2
    Z_perdict_expBase[0] = 0

    return np.nanmean(sumSquare), Z_perdict_expBase



# ------------------------------------------



filename = 'stdFile\G7-Uni-AxialTest.dat'  # Replace 'data.txt' with your file path
list_a, list_b, list_c = ReadLabFile(filename)
list_c = np.array(list_c)*(-1000)
list_a = np.array(list_a)*1

Y_exp = list_c
Z_exp = list_a

# X, Y, Z = read_file("RunSequent\Log_Run_D_SLSQP.txt")
# Y = Y[:,1:]
# Z = Z[:,1:]
# Y *= -1000
# Z *= -1

def draw_dot_plot(x_list, y_list, name_list):
    if len(x_list) != len(y_list) or len(x_list) != len(name_list):
        raise ValueError("Lists must be of equal length")
    
    plt.figure(figsize=(8, 6))
    
    # Plotting dots and names
    for i in range(len(x_list)):
        plt.scatter(x_list[i], y_list[i], color='blue', s=100)
        plt.text(x_list[i], y_list[i], name_list[i], fontsize=12, ha='right', va='bottom')
    
    # Set labels and title
    plt.xlabel('Simulation Time')
    plt.ylabel('MSE')
    plt.title('Dot Plot')
    
    # Show plot
    plt.grid(True)
    plt.show()


X, Y, Z = read_file("RunLog\Log_Run_B_0_1_0604.txt")
Y = Y[:,1:]
Z = Z[:,1:]
Y *= -1000
Z *= -1
Z *= 0.8

index = 20
F_value, interpolateValue = findF(Y[index],Z[index][:50])

print(interpolateValue)

