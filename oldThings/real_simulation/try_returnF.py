import time
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
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

def drawInterpolateStep():

    X, Y, Z = read_file("Log_Run_F_Nelder-Mead.txt")
    Z = Z[:,1:52]
    Z *= -1

    Z = Z*0.8

    index = len(Z) - 1

    scaleDownY = Y[index][::4]
    scaleDownZ = Z[index][::4]

    global list_c
    global list_a

    list_c = list_c[::4]
    list_a = list_a[::4]

    F_value, interpolateValue = findF(scaleDownY,scaleDownZ)

    plt.figure(figsize=(6, 4))
    # plt.text(3,0, TextVal, fontsize=14)
    plt.xlabel('ε (‰)', fontsize=20)
    plt.ylabel('σ (MPa)', fontsize=20)
    plt.title("D",fontsize=20)

    ax = plt.gca()

    # plt.plot(scaleDownY,scaleDownZ, label="Simulation Value", linestyle='-')
    plt.plot(list_c,list_a, marker="o", label="Experiment Value", linestyle=' ')
    plt.plot(list_c,interpolateValue[::4], marker="o", label="Simulation Interpolation", linestyle=' ')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(40))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

    ax.tick_params(which='major', length=5, width=2, colors='black', labelsize=14, direction='in')
    ax.tick_params(which='minor', length=2, width=1, colors='black', direction='in')

    plt.tight_layout()

    ax.grid(which='major', linestyle='--')
    plt.legend()
    plt.show()

def draw1():

    X, Y, Z = read_file("Log_Run_F_Nelder-Mead.txt")
    Z = Z[:,1:52]
    Z *= -1

    index = len(Z) - 1
    F_value, interpolateValue = findF(Y[index],Z[index])

    K1 = X[index][0]*0.00034+0.000114
    C1 = X[index][1]*0.8+0.1
    C3 = X[index][2]*70+10
    C4 = X[index][3]*220+30
    C5 = X[index][4]*3+1
    C7 = X[index][5]*180+20
    C8 = X[index][6]*16+4
    C10= X[index][7]*1.2+0.2
    C11= X[index][8]*0.6+0.1
    C12= X[index][9]*6000+5000
    E   = X[index][10]*13000 + 57000

    TextVal = ""
    TextVal += "E : "+    "{:.6f}".format(E)  + '\n\n'
    TextVal += "K1 : "+      "{:.6f}".format(K1)  + '\n\n'
    TextVal += "C1 : "+      "{:.6f}".format(C1)  + '\n\n'
    TextVal += "C3 : "+      "{:.6f}".format(C3)  + '\n\n'
    TextVal += "C4 : "+      "{:.6f}".format(C4)  + '\n\n'
    TextVal += "C5 : "+      "{:.6f}".format(C5)  + '\n\n'
    TextVal += "C7 : "+      "{:.6f}".format(C7)  + '\n\n'
    TextVal += "C8 : "+      "{:.6f}".format(C8)  + '\n\n'
    TextVal += "C10 : "+    "{:.6f}".format(C10) + '\n\n'
    TextVal += "C11 : "+    "{:.6f}".format(C11) + '\n\n'
    TextVal += "C12 : "+    "{:.6f}".format(C12)
    # ------------------------------------------------------------

    plt.figure(figsize=(6, 4))
    plt.plot(Y[index],Z[index], label="Simulation Value")
    # plt.text(3,0, TextVal, fontsize=14)
    plt.xlabel('ε (‰)', fontsize=20)
    plt.ylabel('σ (MPa)', fontsize=20)

    ax = plt.gca()

    plt.plot(list_c,list_a, marker="o", label="Experiment Value", markersize=5)
    plt.plot(list_c,interpolateValue, marker="o", label="Simulation Interpolation", markersize=5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(40))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

    # plt.plot(list_c,list_a, marker="o", label="Experiment Value", markersize=7)
    # plt.plot(list_c,interpolateValue, marker="o", label="Simulation Interpolation", markersize=7)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(2))

    ax.tick_params(which='major', length=5, width=2, colors='black', labelsize=14, direction='in')
    ax.tick_params(which='minor', length=2, width=1, colors='black', direction='in')

    plt.tight_layout()

    ax.grid(which='major', linestyle='--')
    # plt.legend()
    plt.show()

def drawP1():
    plt.figure(figsize=(6, 4))

    X, Y, Z = read_file("RunLog\Log_Run_G_Phase1.txt")
    Z = Z[:,1:52]
    Z *= -1

    fList = []

    for index in range(len(Y)):
        F_value, interpolateValue = findF(Y[index],Z[index])
        fList.append(F_value)

    fList = np.clip(fList,0,1500)

    # fList = fList[:300]
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    min_values = [min(fList[:i+1]) for i in range(len(fList))]

    
    plt.plot(range(len(fList)),fList,marker = 'x', linestyle = '', color=color, markersize=8)
    plt.plot(range(len(min_values)), min_values, linestyle='-', color=color)

    plt.xlabel('Run times', fontsize=14)
    plt.ylabel('MSE', fontsize=14)

    ax = plt.gca()
    ax.tick_params(which='major', length=5, width=2, colors='black', labelsize=14, direction='in')
    ax.tick_params(which='minor', length=2, width=1, colors='black', direction='in')

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(40))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(4))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(4))

    plt.tight_layout()

    ax.grid(which='major', linestyle='--')
    plt.legend()

    plt.title("")
    plt.legend()
    plt.show()

def drawNelder():
    plt.figure(figsize=(6, 4))
    methodList = ['Nelder-Mead']
    for eachMethod in methodList:

        X, Y, Z = read_file("Log_Run_F_"+eachMethod+".txt")
        # X, Y, Z = read_file("RunLog\Log_Run_G_Phase1.txt")
        Z = Z[:,1:52]
        Z *= -1

        fList = []

        for index in range(len(Y)):
            F_value, interpolateValue = findF(Y[index],Z[index])
            fList.append(F_value)

        # fList = np.clip(fList,0,50)

        fList = fList[:250]
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        min_values = [min(fList[:i+1]) for i in range(len(fList))]

        
        plt.plot(range(len(fList)),fList,marker = 'x', linestyle = '', color=color, markersize=8)
        plt.plot(range(len(min_values)), min_values, linestyle='-', color=color)

        break

    plt.xlabel('Run times', fontsize=14)
    plt.ylabel('MSE', fontsize=14)

    ax = plt.gca()
    ax.tick_params(which='major', length=5, width=2, colors='black', labelsize=14, direction='in')
    ax.tick_params(which='minor', length=2, width=1, colors='black', direction='in')

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(2))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    plt.tight_layout()

    ax.grid(which='major', linestyle='--')
    plt.legend()

    plt.title("")
    plt.legend()
    plt.show()

def drawAll():
    plt.figure(figsize=(6, 4))
    methodList = ['Nelder-Mead', 'Powell', 'CG', 'TNC', 'SLSQP', 'trust-constr']
    for eachMethod in methodList:

        X, Y, Z = read_file("Log_Run_F_"+eachMethod+".txt")
        # X, Y, Z = read_file("RunLog\Log_Run_G_Phase1.txt")
        Z = Z[:,1:52]
        Z *= -1

        fList = []

        for index in range(len(Y)):
            F_value, interpolateValue = findF(Y[index],Z[index])
            fList.append(F_value)

        fList = np.clip(fList,0,50)

        fList = fList[:250]
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        min_values = [min(fList[:i+1]) for i in range(len(fList))]

        
        plt.plot(range(len(fList)),fList,marker = 'x', linestyle = '', color=color, markersize=6)
        plt.plot(range(len(min_values)), min_values, linestyle='-', color=color, label=eachMethod)

    plt.xlabel('Run times', fontsize=14)
    plt.ylabel('MSE', fontsize=14)

    ax = plt.gca()
    ax.tick_params(which='major', length=5, width=2, colors='black', labelsize=14, direction='in')
    ax.tick_params(which='minor', length=2, width=1, colors='black', direction='in')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(2))

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    plt.tight_layout()

    ax.grid(which='major', linestyle='--')
    plt.legend()

    plt.title("")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # drawAll()
    # drawP1()
    # drawInterpolateStep()
    draw1()