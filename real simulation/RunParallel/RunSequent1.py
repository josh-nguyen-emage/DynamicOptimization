import os
import time
import cv2
import numpy as np
import pyautogui
import subprocess

from function import find_and_click_template
from define import *

# def RunSimulation():
#     subprocess.Popen("H:\\02.Working-Thinh\\ATENA-WORKING\\G7-Cyl-Trial-1.inp", shell=True)
#     time.sleep(2)
#     find_and_click_template("snap\\Step1.png")
#     time.sleep(1)
#     find_and_click_template("snap\\Step2.png")
#     # Press Enter key
#     time.sleep(0.5)
#     pyautogui.press('enter')

#     # Press Ctrl+F7
#     time.sleep(0.5)
#     pyautogui.hotkey('ctrl', 'f7')

#     time.sleep(0.5)
#     find_and_click_template("snap\\SimuCompleted.png")

#     time.sleep(0.5)
#     pyautogui.hotkey('alt', 'f4')

#     time.sleep(0.5)
#     pyautogui.press('right')

#     time.sleep(0.5)
#     pyautogui.press('enter')

def RunSimulation(idx):
    cwd = "C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\AtenaPool\\"+str(idx)
    subprocess.run("start /wait cmd /c \"C:\\Program Files (x86)\\CervenkaConsulting\\AtenaV5\\AtenaConsole.exe\" "+cwd+"\\G7-Cyl-Trial-1.inp a.out a.msg a.err",
                cwd="H:\\02.Working-Thinh\\ATENA-WORKING",
                stdout=subprocess.DEVNULL,
                shell=True,
                check=True)
    


