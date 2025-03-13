import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def read_file(txt_file):
    data = np.loadtxt(txt_file)
    x = data[:, 0]
    y = data[:, 1]
    return x, y

def extrapolate(x, y, degree, target=4.6):
    if x[-1] >= target:
        return x, y
    poly_coeffs = np.polyfit(x, y, degree)
    dx = x[-1] - x[-2]
    new_x = np.arange(x[-1] + dx, target + dx/2, dx)
    if new_x[-1] != target:
        new_x[-1] = target
    new_y = np.polyval(poly_coeffs, new_x)
    x_extended = np.concatenate([x, new_x[10:]-0.6])
    y_extended = np.concatenate([y, new_y[10:]-6])
    return x_extended, y_extended

def plot_data(x, y, i):
    idx1 = x <= 3.5
    idx2 = x >= 3.5

    plt.plot(x[idx1], y[idx1], color='blue', marker='o', label='Original')
    plt.plot(x[idx2], y[idx2], color='red', marker='o', label='Extrapolate')
    plt.xlabel("ε (‰)")
    plt.ylabel("σ (MPa)")
    plt.title('Extrapolated Data degree '+str(i))
    plt.legend()
    plt.show()

x,y = read_file("D:\\1 - Study\\6 - DTW_project\\Container\\stdFile\\G7-Uni-AxialTest-preprocess.dat")
i = 4
newX, newY = extrapolate(x,y,i)
plot_data(newX, newY,i)

output_file = "D:\\1 - Study\\6 - DTW_project\\Container\\stdFile\\G7-Uni-AxialTest-extrapolate.dat"
with open(output_file, 'w') as outfile:
    for x, y in zip(newX, newY):
        outfile.write(f"{y:.5f} 0 {x:.7f}\n")
