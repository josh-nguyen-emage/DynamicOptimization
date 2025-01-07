
import datetime
import sys, os

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
sys.path.append(os.path.abspath(os.path.join('.')))
import re
import subprocess

import numpy as np
from scipy.interpolate import interp1d

def pathIdx(idx):
    return "C:\\BuiDucVinh\\01.Duy Thinh\\AtenaPool\\"+str(idx)+"\\"

def ReadLabFile(filename):
    list_a = []
    list_b = []
    list_c = []

    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split()
            if len(values) != 3:
                print(f"Ignoring line: {line.strip()}. Expected 3 values per line.")
                continue
            try:
                a, b, c = map(float, values)
                list_a.append(a)
                list_b.append(b)
                list_c.append(c)
            except ValueError:
                print(f"Ignoring line: {line.strip()}. Could not convert values to floats.")

    return list_a, list_b, list_c

if os.path.exists("C:\\BuiDucVinh\\01.Duy Thinh\\SourceCode\\Container\\stdFile"):
    filename = 'C:\\BuiDucVinh\\01.Duy Thinh\\SourceCode\\Container\\stdFile\\ACI_239C_processed.dat'  # Replace 'data.txt' with your file path
else:
    filename = "D:\\1 - Study\\6 - DTW_project\\Container\\stdFile\\ACI_239C_processed.dat"

list_a, list_b, list_c = ReadLabFile(filename)
list_c = np.array(list_c)*(-1000)
list_b = np.array(list_b)*(1000)
list_a = np.array(list_a)

stress_exp = list_a
bodyOpen_exp = list_b
strain_exp = list_c

def getExpData():
    return stress_exp,strain_exp,bodyOpen_exp


def read_file(filename):
    with open(filename, 'r') as file:
        first_values = []
        secondValue = []
        last_values = []
        last_values2 = []
        counter = 0
        for line in file:
            counter += 1
            # if counter > 5000:
            #     break
            # Split the line by colon ":"
            parts = line.strip().split(':')
            if len(parts) == 4:
                first_values.append(list(map(float, parts[0].split())))
                secondValue.append(list(map(float, parts[1].split())))
                last_values.append(list(map(float, parts[2].split())))
                last_values2.append(list(map(float, parts[3].split())))
            else:
                print(len(last_values))
                print("Invalid line:")
            # break
    return np.array(first_values), np.array(secondValue),np.array(last_values),np.array(last_values2)



def save_to_file(inputs, outputs, filename):
    strainVal = outputs[0]
    stressVal = outputs[1]
    bodyOpen = outputs[2]
    inStr = [str(num) for num in inputs]
    outStrain = [str(num) for num in strainVal]
    outStress = [str(num) for num in stressVal]
    outbodyOpen = [str(num) for num in bodyOpen]
    with open(filename, 'a') as file:
        file.write(' '.join(inStr)+" : "+' '.join(outStrain)+" : "+' '.join(outStress)+" : "+' '.join(outbodyOpen)+"\n")
        
# Function to save input and output to a text file
def save_to_file_rilem(inputs, outputs, filename):
    inStr = [str(num) for num in inputs]
    array_string = np.array2string(outputs, separator=',')  # Convert array to string
    with open(filename, 'a') as file:
        file.write(' '.join(inStr)+" : "+array_string+"\n\n")


def WriteParameter(data,idx):
    data = np.clip(data,0,1)
    K1 = data[0]*0.00034+0.000114
    C1 = data[1]*0.8+0.1
    C3 = data[2]*70+10
    C4 = data[3]*220+30
    C5 = data[4]*3+1
    C7 = data[5]*250+20
    C8 = data[6]*16+4
    C10= data[7]*1.2+0.2
    C11= data[8]*0.6+0.1
    C12= data[9]*6000+5000
    E  = data[10]*20000 + 57000
    writeInpFile(K1,C1,C3,C4,C5,C7,C8,C10,C11,C12, E,idx)

def WriteParameterRilem(data,idx):
    data = np.clip(data,0,1)
    K1 = data[0]*0.00034+0.000114
    C1 = data[1]*0.8+0.1
    C3 = data[2]*70+10
    C4 = data[3]*220+30
    C5 = data[4]*3+1
    C7 = data[5]*250+20
    C8 = data[6]*16+4
    C10= data[7]*1.2+0.2
    C11= data[8]*0.6+0.1
    C12= data[9]*6000+5000
    E  = data[10]*20000 + 57000
    writeInpFileRilem(K1,C1,C3,C4,C5,C7,C8,C10,C11,C12, E,idx)

def writeInpFile(
        K1=0.00027, 
        C1=0.62, 
        C3=4, 
        C4=70, 
        C5=2.5, 
        C7=50, 
        C8=8, 
        C10=0.73, 
        C11=0.2, 
        C12=7000,
        E  =66131,
        idx=0):
    # Read the content of the file
    file_path = pathIdx(idx) + 'G7-Cyl-Trial-2.inp'
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Modify the line with the new text
    lines[46 - 1] = "        E	"+      "{:.6f}".format(E)  + '\n'
    lines[49 - 1] = "        K1	"+      "{:.6f}".format(K1)  + '\n'
    lines[53 - 1] = "        C1	"+      "{:.6f}".format(C1)  + '\n'
    lines[56 - 1] = "        C3	"+      "{:.6f}".format(C3)  + '\n'
    lines[57 - 1] = "        C4	"+      "{:.6f}".format(C4)  + '\n'
    lines[58 - 1] = "        C5	"+      "{:.6f}".format(C5)  + '\n'
    lines[60 - 1] = "        C7	"+      "{:.6f}".format(C7)  + '\n'
    lines[61 - 1] = "        C8	"+      "{:.6f}".format(C8)  + '\n'
    lines[63 - 1] = "        C10	"+  "{:.6f}".format(C10) + '\n'
    lines[64 - 1] = "        C11	"+  "{:.6f}".format(C11) + '\n'
    lines[65 - 1] = "        C12	"+  "{:.6f}".format(C12) + '\n'

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

def writeInpFileRilem(
        K1=0.00027, 
        C1=0.62, 
        C3=4, 
        C4=70, 
        C5=2.5, 
        C7=50, 
        C8=8, 
        C10=0.73, 
        C11=0.2, 
        C12=7000,
        E  =66131,
        idx=0):
    # Read the content of the file
    file_path = pathIdx(idx) + 'UHPC-RILEM-2024-V5LZ-M7.inp'
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Modify the line with the new text
    lines[64 - 1] = "        E	"+      "{:.6f}".format(E)  + '\n'
    lines[70 - 1] = "        K1	"+      "{:.6f}".format(K1)  + '\n'
    lines[76 - 1] = "        C1	"+      "{:.6f}".format(C1)  + '\n'
    lines[79 - 1] = "        C3	"+      "{:.6f}".format(C3)  + '\n'
    lines[81 - 1] = "        C4	"+      "{:.6f}".format(C4)  + '\n'
    lines[82 - 1] = "        C5	"+      "{:.6f}".format(C5)  + '\n'
    lines[84 - 1] = "        C7	"+      "{:.6f}".format(C7)  + '\n'
    lines[85 - 1] = "        C8	"+      "{:.6f}".format(C8)  + '\n'
    lines[87 - 1] = "        C10	"+  "{:.6f}".format(C10) + '\n'
    lines[88 - 1] = "        C11	"+  "{:.6f}".format(C11) + '\n'
    lines[89 - 1] = "        C12	"+  "{:.6f}".format(C12) + '\n'

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

def printWithTime(outString):
    current_time = datetime.datetime.now()

    # Extract hours, minutes, and seconds
    hours = current_time.hour
    minutes = current_time.minute
    seconds = current_time.second

    print(f"{hours:02d}-{minutes:02d}-{seconds:02d} : "+outString)

def calculate_covariance(list1, list2):
    covariance_matrix = np.cov(list1, list2)
    covariance = covariance_matrix[0, 1]  # Covariance is the off-diagonal element of the covariance matrix
    return covariance

def remove_duplicates(x, y):
    # Ensure both lists have the same length
    if len(x) != len(y):
        raise ValueError("Both lists must have the same length.")
    
    # Create sets to identify duplicates
    x_set = set()
    y_set = set()
    
    # Find indices of duplicates
    x_duplicates = {i for i, val in enumerate(x) if val in x_set or x_set.add(val)}
    y_duplicates = {i for i, val in enumerate(y) if val in y_set or y_set.add(val)}
    
    # Combine indices of all duplicates
    all_duplicates = x_duplicates.union(y_duplicates)
    
    # Remove elements at duplicate indices from both lists
    new_x = [val for i, val in enumerate(x) if i not in all_duplicates]
    new_y = [val for i, val in enumerate(y) if i not in all_duplicates]
    
    return new_x, new_y

def interpolate_line(x_values, y_values, X_interpolate):
    x_values, y_values = remove_duplicates(x_values, y_values)
    interp_func = interp1d(x_values, y_values, kind='cubic', bounds_error=False)

    Y_interpolate = interp_func(X_interpolate)

    return Y_interpolate

def calculate_mse(array1, array2):
    """
    Calculate the Mean Squared Error (MSE) between two numpy arrays.
    
    For 1D arrays, it returns a single MSE value.
    For 2D arrays, it calculates the MSE for each pair of corresponding rows 
    and returns a list of MSE values.

    Parameters:
        array1 (np.ndarray): First input array.
        array2 (np.ndarray): Second input array.

    Returns:
        float or list: MSE value(s) between the arrays.
    """
    if array1.ndim == 1:
        # Calculate MSE for 1D arrays
        mse = np.mean((array1 - array2) ** 2)
        return mse
    elif array1.ndim == 2:
        # Calculate MSE for each row in 2D arrays
        mses = []
        for idx in range(array1.shape[0]):
            row1 = array1[idx]
            row2 = array2[idx]
            mse = np.mean((row1 - row2) ** 2)
            mses.append(mse)
        return mses
    else:
        raise ValueError("Arrays must be 1D or 2D.")

def calculate_correlation(array1, array2):
    """
    Calculate the correlation coefficient between two numpy arrays.
    
    For 1D arrays, it returns a single correlation value.
    For 2D arrays, it calculates the correlation for each pair of corresponding rows 
    and returns a list of correlation values.

    Parameters:
        array1 (np.ndarray): First input array.
        array2 (np.ndarray): Second input array.

    Returns:
        float or list: Correlation coefficient(s) between the arrays.
    """
    if array1.ndim == 1:
        # Calculate correlation for 1D arrays
        mean1 = np.mean(array1)
        mean2 = np.mean(array2)
        std1 = np.std(array1)
        std2 = np.std(array2)
        covariance = np.mean((array1 - mean1) * (array2 - mean2))
        correlation = covariance / (std1 * std2)
        return 1 - abs(correlation)
    elif array1.ndim == 2:
        # Calculate correlation for each row in 2D arrays
        correlations = []
        for idx in range(array1.shape[0]):
            row1 = array1[idx]
            row2 = array2
            mean1 = np.mean(row1)
            mean2 = np.mean(row2)
            std1 = np.std(row1)
            std2 = np.std(row2)
            covariance = np.mean((row1 - mean1) * (row2 - mean2))
            correlation = covariance / (std1 * std2)
            correlations.append(correlation)
        result = [1 - abs(num) for num in correlations]
        return result
    else:
        raise ValueError("Arrays must be 1D or 2D.")

def findF(stress_run ,bodyOpen_run, strain_run):
    global strain_exp
    global stress_exp
    global bodyOpen_exp

    stress_perdict_exp_strain = interpolate_line(strain_run, stress_run,strain_exp)
    stress_perdict_exp_strain[0] = stress_exp[0]
    stress_perdict_exp_strain[-1] = stress_perdict_exp_strain[-2]
    sumSquare1 = calculate_correlation(stress_perdict_exp_strain,stress_exp)
    mse1 = calculate_mse(stress_perdict_exp_strain,stress_exp)

    stress_perdict_exp_bodyOpen = interpolate_line(bodyOpen_run, stress_run,bodyOpen_exp)
    stress_perdict_exp_bodyOpen[0] = stress_exp[0]
    sumSquare2 = calculate_correlation(stress_perdict_exp_bodyOpen,stress_exp)

    # interpolateArray = np.concatenate((np.flip(stress_perdict_exp_strain), stress_perdict_exp_bodyOpen))
    # interpolateArray[np.isnan(interpolateArray)] = 300

    # if len(interpolateArray) != 150:
    #     raise ValueError("interpolateArray len is not correct")

    # return (np.nanmean(sumSquare1)*0.5+np.nanmean(sumSquare2)*0.5), interpolateArray
    return sumSquare1*mse1, stress_perdict_exp_strain

def find_first_point_exceeding_threshold(x, y, idx, draw):
    # Take the first 40% of points for approximation
    num_points = len(x)
    approx_points = int(0.3 * num_points)

    threshold_point = None

    for eachPoint in range(approx_points,num_points):
        coeffs = np.polyfit(x[:eachPoint], y[:eachPoint], 1)
        approx_line = np.poly1d(coeffs)

        if abs(y[eachPoint] - approx_line(x[eachPoint])) > 5:
            threshold_point = (x[eachPoint-1], y[eachPoint-1])
            break

    if threshold_point is None:
        threshold_point = (x[-1], y[-1])

    # Plotting
    if draw:
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, label="Input Line", color='blue')
        plt.plot(x, approx_line(x), label="Approximate Line for Alpha", color='green', linestyle='--')
        if threshold_point:
            plt.scatter(*threshold_point, color='red', label="Threshold point")

    top_index = np.argmax(y)
    polygonX = x[:top_index+1]
    polygonX = np.append(polygonX,[polygonX[-1],0])
    polygonY = y[:top_index+1]
    polygonY = np.append(polygonY,[0,0])
    polygonCollection = list(zip(polygonX,polygonY))
    polygon = Polygon(polygonCollection, closed=True, edgecolor='black', facecolor='lightblue')

    if draw:
        plt.gca().add_patch(polygon)
        plt.scatter(x[top_index],y[top_index], color='orange', label="Max Value")
        # Labels and legend
        plt.xlabel("ε (‰)")
        plt.ylabel("σ (MPa)")
        plt.ylim(0, 300)
        plt.legend()
        plt.title("Input Line with Approximate Line and Threshold")
        plt.show()
        # plt.savefig("Log/"+str(idx)+".png")
        plt.close()

    slope = coeffs[0]
    angle_with_x = np.degrees(np.arctan(slope))

    vertices = polygon.get_xy()

    # Calculate the area using the Shoelace formula
    x = vertices[:, 0]
    y = vertices[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    return [angle_with_x, threshold_point[0], threshold_point[1], x[top_index],y[top_index], area]

def findFeatureVal(stress_run ,bodyOpen_run, strain_run):
    global strain_exp
    global stress_exp
    global bodyOpen_exp

    stress_perdict_exp_strain = interpolate_line(strain_run, stress_run,strain_exp)
    stress_perdict_exp_strain[0] = stress_exp[0]
    feature = find_first_point_exceeding_threshold(-1*strain_exp,stress_perdict_exp_strain,0,0)
    return feature

def getExpectChart():
    global stress_exp
    # return np.concatenate((np.flip(stress_exp),stress_exp))
    return stress_exp

def getExpectFeature():
    global stress_exp
    global strain_exp
    feature = find_first_point_exceeding_threshold(-1*strain_exp,stress_exp,0,0)
    return feature

def get_arp_table():
    try:
        # Run the arp command to get the ARP table
        output = subprocess.check_output("getmac", shell=True, text=True)
        return output
    except subprocess.CalledProcessError as e:
        print(f"Error executing arp command: {e}")
        return None

def check_mac_address(target_mac):
    arp_table = get_arp_table()
    if arp_table is None:
        return False

    # Use regular expression to find MAC addresses in the ARP table
    mac_regex = re.compile(r"(([0-9A-Fa-f]{1,2}[:-]){5}([0-9A-Fa-f]{1,2}))")
    matches = mac_regex.findall(arp_table)

    for match in matches:
        mac = match[0]
        if mac.lower() == target_mac.lower():
            return True
    return False

