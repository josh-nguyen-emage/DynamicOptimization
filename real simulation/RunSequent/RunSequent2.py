import os
import subprocess
import time
import cv2
import numpy as np
import pygetwindow as gw
import pyautogui

from function import *
from define import *

def maximize_window_with_name_starting_with(name_prefix):
    # Get all windows with titles starting with the specified prefix
    windows = gw.getWindowsWithTitle(name_prefix)
    
    # Check if any such window exists
    if windows:
        # Maximize each window
        for window in windows:
            window.maximize()
        print("Window maximized successfully.")
    else:
        print("No window found with the specified name prefix.")

def minimize_window_with_name_starting_with(name_prefix):
    # Get all windows with titles starting with the specified prefix
    windows = gw.getWindowsWithTitle(name_prefix)
    
    # Check if any such window exists
    if windows:
        # Minimize each window
        for window in windows:
            window.minimize()
        print("Window maximized successfully.")
    else:
        print("No window found with the specified name prefix.")

# def RunTool4Atena():
#     time.sleep(3)
#     window_name_prefix = "Tool4Atena - Delphi"
#     maximize_window_with_name_starting_with(window_name_prefix)
#     time.sleep(1)
#     pyautogui.click(x=500,y=500)
#     time.sleep(1)
#     pyautogui.hotkey('ctrl', 'shift', 'f9')

#     time.sleep(0.5)
#     find_and_click_template("snap\\tool1.png")

#     time.sleep(0.5)
#     find_and_click_template("snap\\tool2.png")
#     pyautogui.press('enter')

#     time.sleep(0.5)
#     find_and_click_template("snap\\tool3.png")

#     time.sleep(0.5)
#     find_and_click_template("snap\\tool4.png")

#     time.sleep(0.5)
#     pyautogui.hotkey('alt', 'f4')

#     time.sleep(0.5)
#     find_and_click_template("snap\\tool5.png")

#     minimize_window_with_name_starting_with(window_name_prefix)
        
def RunTool4Atena():
    os.remove(pathName+'G7-Cyl-Trial-1_NODES_STRAIN.atf')
    os.remove(pathName+'G7-Cyl-Trial-1_NODES_STRESS.atf')
    subprocess.run("\"C:\\Program Files (x86)\\CervenkaConsulting\\AtenaV5\\AtenaConsole.exe\" \"H:\\02.Working-Thinh\\ATENA-WORKING\\Post.atn\"",
                cwd="H:\\02.Working-Thinh\\ATENA-WORKING",
                stdout=subprocess.DEVNULL,
                shell=True,
                check=True)

# RunTool4Atena()


