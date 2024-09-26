import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))

from Libary.function import findF
from Phase1 import read_file
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


returnVal = read_file("D:\\1 - Study\\6 - DTW_project\\Container\\BurningTest_3dDraw_10-8.txt")

param = returnVal[0]
strain = np.array(returnVal[1])/(-4)
stress = np.array(returnVal[2])/(300)
bodyOpen = np.array(returnVal[3])/(20)

allMSE = []
x = []
y = []

for idx in range(len(returnVal[0])):

    MSE, interpolate = findF(stress[idx], bodyOpen[idx], strain[idx])
    allMSE.append(MSE)
    print(idx,MSE)
    x.append(param[idx][0])
    y.append(param[idx][10])

print(param[0])

print("min",min(allMSE))

grid_x, grid_y = np.mgrid[0:1:0.01, 0:1:0.01]

# Interpolate the values over the grid
grid_z = griddata((x, y), allMSE, (grid_x, grid_y), method='cubic')

# Plotting
fig = plt.figure(figsize=(13, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')

# Add the original data points
# ax.scatter(x, y, allMSE, c='red', label='Data Points')

# Labels and title
ax.set_title('3D Interpolated Surface')
ax.set_xlabel('K1')
ax.set_ylabel('E')
ax.set_zlabel('Interpolated Value')

# Color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Interpolated Value')

plt.legend()
plt.show()
