
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

filename = 'stdFile\G7-Uni-AxialTest.dat'  # Replace 'data.txt' with your file path
list_a, list_b, list_c = ReadLabFile(filename)
list_c = np.array(list_c)*(-1000)
list_a = np.array(list_a)

Y_exp = list_c
Z_exp = list_a

def read_file(filename):
    with open(filename, 'r') as file:
        first_values = []
        secondValue = []
        last_values = []
        for line in file:
            # Split the line by colon ":"
            parts = line.strip().split(':')
            if len(parts) == 3:
                first_values.append(list(map(float, parts[0].split())))
                secondValue.append(list(map(float, parts[1].split())))
                last_values.append(list(map(float, parts[2].split())))
            else:
                print("Invalid line:")
    return np.array(first_values), np.array(secondValue),np.array(last_values)



# Function to save input and output to a text file
def save_to_file(inputs, outputs, filename):
    strainVal = outputs[0]
    stressVal = outputs[1]
    with open(filename, 'a') as file:
        inStr = [str(num) for num in inputs]
        outStrain = [str(num) for num in strainVal]
        outStress = [str(num) for num in stressVal]
        file.write(' '.join(inStr)+" : "+' '.join(outStrain)+" : "+' '.join(outStress)+"\n")
        

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

def interpolate_line(x_values, y_values, X_interpolate):
    interp_func = interp1d(x_values, y_values, kind='cubic', bounds_error=False)

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