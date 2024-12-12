import sys, os

import pandas as pd
sys.path.append(os.path.abspath(os.path.join('.')))

from Libary.function import findF, getExpData, getExpectChart
from Phase1 import read_file
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def analyze_impact(A, B, output_file="impact_analysis.xlsx"):
    n_cols_A = A.shape[1]
    n_cols_B = B.shape[1]
    
    # Create a DataFrame to store the impact results
    impact_df = pd.DataFrame(index=[f'B_col_{i}' for i in range(n_cols_B)],
                             columns=[f'A_col_{j}' for j in range(n_cols_A)])
    
    # Calculate the correlation between each column of B and each column of A
    for i in range(n_cols_B):
        for j in range(n_cols_A):
            impact_df.iloc[i, j] = np.corrcoef(B[:, i], A[:, j])[0, 1]
    
    # Save the impact results to an Excel file
    impact_df.to_excel(output_file)

def find_first_point_exceeding_threshold(x, y, idx, draw):
    # Take the first 40% of points for approximation
    num_points = len(x)
    approx_points = int(0.3 * num_points)

    threshold_point = None

    for eachPoint in range(approx_points,num_points):
        coeffs = np.polyfit(x[:eachPoint], y[:eachPoint], 1)
        approx_line = np.poly1d(coeffs)

        if abs(y[eachPoint] - approx_line(x[eachPoint])) > 5:
            threshold_point = (x[eachPoint-1], y[eachPoint-1])
            break

    if threshold_point is None:
        threshold_point = (x[-1], y[-1])

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label="Input Line", color='blue')
    
    # Plot the approximate line over the entire x range
    plt.plot(x, approx_line(x), label="Approximate Line for Alpha", color='green', linestyle='--')
    
    # Plot the threshold point if found
    if threshold_point:
        plt.scatter(*threshold_point, color='red', label="Threshold point")

    top_index = np.argmax(y)
    polygonX = x[:top_index+1]
    polygonX = np.append(polygonX,[polygonX[-1],0])
    polygonY = y[:top_index+1]
    polygonY = np.append(polygonY,[0,0])
    polygonCollection = list(zip(polygonX,polygonY))
    polygon = Polygon(polygonCollection, closed=True, edgecolor='black', facecolor='lightblue')

    plt.gca().add_patch(polygon)
    plt.scatter(x[top_index],y[top_index], color='orange', label="Max Value")
    # Labels and legend
    plt.xlabel("ε (‰)")
    plt.ylabel("σ (MPa)")
    plt.ylim(0, 300)
    plt.legend()
    plt.title("Input Line with Approximate Line and Threshold")
    # plt.show()
    plt.savefig("Log/"+str(idx)+".png")
    plt.close()

    slope = coeffs[0]
    angle_with_x = np.degrees(np.arctan(slope))

    return threshold_point

returnVal = read_file("D:\\1 - Study\\6 - DTW_project\\Container\\LogRun_weightMSE.txt")
# returnVal = read_file("D:\\1 - Study\\6 - DTW_project\\Container\\BurningTest_3dDraw_10-8.txt")

param = np.array(returnVal[0])
strain = np.array(returnVal[1])
stress = np.array(returnVal[2])
bodyOpen = np.array(returnVal[3])

param = param[:3000]

allMSE = []
stress_exp,strain_exp,bodyOpen_exp = getExpData()

for idx in range(len(param)):
    MSE, interpolate = findF(stress[idx], bodyOpen[idx], strain[idx])
    allMSE.append(MSE)

    print(idx)
    # plt.scatter(bodyOpen[idx],stress[idx],label='Simulate Line')
    # plt.scatter(strain[idx],stress[idx],label='Simulate Line')

    find_first_point_exceeding_threshold(strain[idx]*-1,stress[idx],idx,0)

    if (idx > 50):
        sys.exit()
    

print(min(allMSE),stress[idx], bodyOpen[idx], strain[idx])
    
plt.scatter(range(len(allMSE)),allMSE)

plt.xlabel('Run times')
plt.ylabel('MSE')
plt.title('Bayes - AI model')

# Show plot
plt.show()


