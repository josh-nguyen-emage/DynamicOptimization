import os
import subprocess

from real_simulation.GlobalLib import pathIdx
        
def RunTool4Atena(idx):
    if os.path.exists(pathIdx(idx)+'G7-Cyl-Trial-1_NODES_STRAIN.atf'):
        os.remove(pathIdx(idx)+'G7-Cyl-Trial-1_NODES_STRAIN.atf')
        os.remove(pathIdx(idx)+'G7-Cyl-Trial-1_NODES_STRESS.atf')
    # cwd = "C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\AtenaPool\\"+str(idx)
    # subprocess.run("start /wait cmd /c \"C:\\Program Files (x86)\\CervenkaConsulting\\AtenaV5\\AtenaConsole.exe\" H:\\02.Working-Thinh\\ATENA-WORKING\\Post.atn",
    #             cwd=cwd,
    #             stdout=subprocess.DEVNULL,
    #             shell=True,
    #             check=True)
    
    cwd = "C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\AtenaPool\\"+str(idx)
    command = ["C:\\Program Files (x86)\\CervenkaConsulting\\AtenaV5\\AtenaConsole.exe", "H:\\02.Working-Thinh\\ATENA-WORKING\\Post.atn"]
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
    process.wait()

    # subprocess.run("\"C:\\Program Files (x86)\\CervenkaConsulting\\AtenaV5\\AtenaConsole.exe\" H:\\02.Working-Thinh\\ATENA-WORKING\\Post.atn",
    #         cwd=cwd,
    #         stdout=subprocess.DEVNULL,
    #         shell=True,
    #         check=True)
    


