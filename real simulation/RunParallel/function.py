
import datetime
import time
import numpy as np
import cv2
import pyautogui

from define import *

def find_and_click_template(template_path):
    print("Press template ",template_path)
    while True:
        # Load the template image
        
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print("Error: Template image not found.")
            return

        # Capture the screen
        screenshot = pyautogui.screenshot()
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # Convert screenshot to grayscale
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # print(template_path," - template thres: ",max_val)

        # If similarity is higher than 80%, click at the center of the template
        if max_val >= 0.8:
            template_width, template_height = template.shape[::-1]
            center_x = max_loc[0] + template_width // 2
            center_y = max_loc[1] + template_height // 2
            time.sleep(0.5)
            pyautogui.click(x=center_x, y=center_y)
            break
        else:
            time.sleep(1)

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

def WriteParameter_K1_Only(data,idx):
    data = np.clip(data,0,1)
    K1 = data[0]*0.00034+0.000114
    writeInpFile(K1=K1,idx=idx)

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