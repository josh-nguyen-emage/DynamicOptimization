import os
import subprocess
import time
import cv2
import numpy as np
import pygetwindow as gw
import pyautogui

from function import *
from define import *

        
def RunTool4Atena(idx):
    if os.path.exists(pathIdx(idx)+'G7-Cyl-Trial-1_NODES_STRAIN.atf'):
        os.remove(pathIdx(idx)+'G7-Cyl-Trial-1_NODES_STRAIN.atf')
        os.remove(pathIdx(idx)+'G7-Cyl-Trial-1_NODES_STRESS.atf')
    cwd = "C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\AtenaPool\\"+str(idx)

    # subprocess.run("start /wait cmd /c \"C:\\Program Files (x86)\\CervenkaConsulting\\AtenaV5\\AtenaConsole.exe\" H:\\02.Working-Thinh\\ATENA-WORKING\\Post.atn",
    #             cwd=cwd,
    #             stdout=subprocess.DEVNULL,
    #             shell=True,
    #             check=True)

    subprocess.run("\"C:\\Program Files (x86)\\CervenkaConsulting\\AtenaV5\\AtenaConsole.exe\" H:\\02.Working-Thinh\\ATENA-WORKING\\Post.atn",
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            shell=True,
            check=True)
    


