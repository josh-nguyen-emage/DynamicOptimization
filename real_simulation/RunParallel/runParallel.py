import subprocess
from matplotlib import pyplot as plt
import time
import threading

from RunAlgo_Parallel import RunAlgo


def Process(idx):
    RunAlgo(idx)

timeColector = []
numThread = 9


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

