import os
import time
import cv2
import numpy as np
import pyautogui
import subprocess

from function import find_and_click_template
from define import *


def RunSimulation_timeCheck(idx):
    cwd = "C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\AtenaPool\\" + str(idx)
    command = ["C:\\Program Files (x86)\\CervenkaConsulting\\AtenaV5\\AtenaConsole.exe", cwd + "\\G7-Cyl-Trial-1.inp", "a.out", "a.msg", "a.err"]

    try:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        # process = subprocess.Popen(command, cwd=cwd, shell=True)

        # Start time
        start_time = time.time()

        while True:
            if process.poll() is not None:
                # Process has terminated, calculate elapsed time
                elapsed_time = time.time() - start_time
                return elapsed_time
            
            # Check if timeout reached
            if time.time() - start_time > 1800:  # 900 seconds = 15 minutes
                # Timeout reached, terminate the process
                elapsed_time = time.time() - start_time
                process.terminate()
                return elapsed_time
            
            # Sleep for a short while to avoid busy waiting
            time.sleep(1)

    except subprocess.CalledProcessError:
        # Handle any errors raised by subprocess.run
        return None

    


