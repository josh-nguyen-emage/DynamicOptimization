
import datetime
import sys, os

sys.path.append(os.path.abspath(os.path.join('.')))
import numpy as np
from scipy.interpolate import interp1d

pathName = "H:\\02.Working-Thinh\\ATENA-WORKING\\"

def pathIdx(idx):
    return "C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\AtenaPool\\"+str(idx)+"\\"

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
if os.path.exists("C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\DynamicOptimization-ST\\Container\\stdFile"):
    filename = 'C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\DynamicOptimization-ST\\Container\\stdFile\\G7-Uni-AxialTest.dat'  # Replace 'data.txt' with your file path
else:
    filename = "D:\\1 - Study\\6 - DTW_project\\Container\\stdFile\\G7-Uni-AxialTest.dat"

list_a, list_b, list_c = ReadLabFile(filename)
list_c = np.array(list_c)*(1000)
list_b = np.array(list_b)*(1000)
list_a = np.array(list_a)

stress_exp = list_a
bodyOpen_exp = list_b
strain_exp = list_c


def read_file(filename):
    with open(filename, 'r') as file:
        first_values = []
        secondValue = []
        last_values = []
        last_values2 = []
        for line in file:
            # Split the line by colon ":"
            parts = line.strip().split(':')
            if len(parts) == 4:
                first_values.append(list(map(float, parts[0].split())))
                secondValue.append(list(map(float, parts[1].split())))
                last_values.append(list(map(float, parts[2].split())))
                last_values2.append(list(map(float, parts[3].split())))
            else:
                print("Invalid line:")
            # break
    return np.array(first_values), np.array(secondValue),np.array(last_values),np.array(last_values2)



# Function to save input and output to a text file
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
        

def WriteParameter(data,idx):
    data = np.clip(data,0,1)
    K1 = data[0]*0.00034+0.000114
    C1 = data[1]*0.8+0.1
    C3 = data[2]*70+10
    C4 = data[3]*220+30
    C5 = data[4]*3+1
    C7 = data[5]*180+20
    C8 = data[6]*16+4
    C10= data[7]*1.2+0.2
    C11= data[8]*0.6+0.1
    C12= data[9]*6000+5000
    E  = data[10]*13000 + 57000
    writeInpFile(K1,C1,C3,C4,C5,C7,C8,C10,C11,C12, E,idx)

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
    file_path = pathIdx(idx) + 'G7-Cyl-Trial-1.inp'
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

def findF(stress_run ,bodyOpen_run, strain_run):
    global strain_exp
    global stress_exp

    stress_perdict_exp_strain = interpolate_line(strain_run, stress_run,strain_exp)
    stress_perdict_exp_strain[0] = stress_exp[0]
    sumSquare1 = (stress_perdict_exp_strain-stress_exp)**2

    stress_perdict_exp_bodyOpen = interpolate_line(bodyOpen_run, stress_run,bodyOpen_exp)
    stress_perdict_exp_bodyOpen[0] = stress_exp[0]
    sumSquare2 = (stress_perdict_exp_bodyOpen-stress_exp)**2
    interpolateArray = np.concatenate((np.flip(stress_perdict_exp_strain), stress_perdict_exp_bodyOpen))
    interpolateArray[np.isnan(interpolateArray)] = 250

    if len(interpolateArray) != 150:
        raise ValueError("interpolateArray len is not correct")

    return (np.nanmean(sumSquare1)+np.nanmean(sumSquare2))/2, interpolateArray

def getExpectChart():
    global stress_exp
    return np.concatenate((np.flip(stress_exp),stress_exp))