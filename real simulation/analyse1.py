from matplotlib import pyplot as plt
import numpy as np


from try_trainModel import ReadLabFile, read_file

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------


# X, Y, Z = read_file("RunParallel\Log_Run_B_0_1_0604.txt")
X, Y, Z = read_file("Log_Run_A_1_0304.txt")
Y = Y[:,1:51]
Z = Z[:,1:51]
Y *= -100
Z *= -0.01



# Create a new figure
fig = plt.figure()

# Add 3D subplot
ax = fig.add_subplot(111, projection='3d')

# Plot each dataset
for d in range(100):
    ax.scatter([X[d][0]]*50, Y[d], Z[d])

# Set labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show plot
plt.show()