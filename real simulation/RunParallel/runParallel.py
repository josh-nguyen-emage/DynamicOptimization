import subprocess
from matplotlib import pyplot as plt
import time
import threading

from RunSequent_Master import RunSimulationThread

def run_command_in_terminal(command, currentCwd):
    start_time = time.time()
    completed_process = subprocess.run(command, shell=True, check=True,
                                       cwd=currentCwd)
    return start_time


# Function to be executed by each thread
# def Process(idx):
#     cwd = "C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\AtenaPool\\"+str(idx)
#     cmd_string = "\"C:\\Program Files (x86)\\CervenkaConsulting\\AtenaV5\\AtenaConsole.exe\" "+cwd+"\\G7-Cyl-Trial-1.inp a.out a.msg a.err"
#     run_command_in_terminal("start /wait cmd /c " + cmd_string, cwd)

def Process(idx):
    RunSimulationThread(idx)

timeColector = []
numThread = 8
counter = 0

while True:
    # Create a list to hold the thread objects
    threads = []
    for idx in range(numThread):
        # Create a thread for each index and pass the index as an argument to the function
        thread = threading.Thread(target=Process, args=(idx,))
        # Start the thread
        thread.start()
        # Add the thread object to the list
        threads.append(thread)

    # Main thread waits for all threads to complete
    for thread in threads:
        thread.join()
    counter += 1
    print("---",counter,"---")

