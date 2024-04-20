import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def read_txt_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                print(line)
                parts = line.split(' ')
                a, b, c, d = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[4])
                data.append((a, b, c, d))
    return data

def plot_3d(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    data = np.array(data)
    
    # for point in data[:200]:
    ax.scatter(data[:500,1], data[:500,2], data[:500,3])
    
    ax.set_xlabel('B')
    ax.set_ylabel('C')
    ax.set_zlabel('D')
    
    plt.show()

if __name__ == "__main__":
    file_path = "Log_Time_1.txt"  # Replace with your file path
    data = read_txt_file(file_path)
    plot_3d(data)
