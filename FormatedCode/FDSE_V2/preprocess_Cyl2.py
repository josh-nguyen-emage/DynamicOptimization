import numpy as np
from scipy.signal import savgol_filter

def interpolate_lines(input_file, output_file, num_points=10):
    """
    Reads a text file with two floats on each line, interpolates the values, and writes the results to a new file.

    :param input_file: Path to the input text file.
    :param output_file: Path to the output text file.
    :param num_points: Number of interpolated points per line (default is 10).
    """
import numpy as np
import matplotlib.pyplot as plt

def interpolate_lines(input_file, output_file, num_points=10):
    """
    Reads a text file with two floats on each line, interpolates the values, and writes the results to a new file.

    :param input_file: Path to the input text file.
    :param output_file: Path to the output text file.
    :param num_points: Number of interpolated points per line (default is 10).
    """
    x_vals = []
    y_vals = []
    # Read the input file and parse the data
    with open(input_file, 'r') as infile:
        for line in infile:
            try:
                y, _, x = map(float, line.strip().split())
                x_vals.append(-1*x)
                y_vals.append(y)
            except ValueError:
                print(f"Skipping invalid line: {line.strip()}")

    # Perform interpolation
    interpolated_x = np.linspace(x_vals[0], x_vals[-1], num_points)
    interpolated_y = np.interp(interpolated_x, x_vals, y_vals)

    y_smoothed = savgol_filter(interpolated_y, window_length=15, polyorder=2)

    interpolated_y[30:] = y_smoothed[30:]

    # Plot the original and interpolated points
    plt.figure()
    plt.plot(x_vals, y_vals, 'ro-', label='Original Points')
    plt.plot(interpolated_x, interpolated_y, 'bo-', label='Interpolated Points')
    plt.legend()
    plt.title("Interpolation Across Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    with open(output_file, 'w') as outfile:
        for x, y in zip(interpolated_x, interpolated_y):
            outfile.write(f"{y:.5f} 0 {x:.7f}\n")

# Example usage
input_file = "FDSE_V2\Cyl-2.txt"  # Replace with the path to your input file
output_file = "FDSE_V2\Cyl-2_processed.txt"  # Replace with the path to your output file
interpolate_lines(input_file, output_file, num_points=50)
