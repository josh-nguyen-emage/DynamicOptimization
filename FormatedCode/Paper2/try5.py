import numpy as np

# Function to calculate the volume of a simplex in 11D
def simplex_volume(points):
    """
    Calculate the volume of a simplex defined by points in 11D space.
    """
    matrix = np.array(points[:-1]) - points[-1]  # Construct a matrix from the points
    det = np.linalg.det(matrix.T @ matrix)  # Determinant of the Gram matrix
    return np.sqrt(det) / np.math.factorial(len(points) - 1)  # Volume formula

# Generate 512 random points in 11D space
dimension = 11
num_points = 20
points = np.random.rand(num_points, dimension)

# Monte Carlo simulation to calculate the average volume of valid simplices
num_samples = 10000  # Number of random samples
volumes = []

for _ in range(num_samples):
    # Randomly select 12 points to form a simplex in 11D
    selected_indices = np.random.choice(num_points, dimension + 1, replace=False)
    simplex_points = points[selected_indices]

    # Check if the simplex is valid (non-degenerate)
    try:
        volume = simplex_volume(simplex_points)
        if volume > 0:  # Only consider non-degenerate simplices
            volumes.append(volume)
    except np.linalg.LinAlgError:
        continue  # Skip degenerate simplices

# Calculate the average volume of the simplices
average_volume = np.mean(volumes) if volumes else 0

print(average_volume)
