from Phase1 import *
import numpy as np
import matplotlib.pyplot as plt

returnVal = RunSimulationThread(0,np.random.rand(11))
print(returnVal)
strain = returnVal[0]
stress = returnVal[1]
bodyOpen = returnVal[2]


plt.plot(strain , stress)
plt.plot(bodyOpen , stress)

# Add titles and labels
plt.title('Example')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()