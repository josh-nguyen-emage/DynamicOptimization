import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def heatmap_formula(x, y):
    # Compute the original formula
    value = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    # Apply logarithm to the result, adding a small constant to avoid log(0)
    return np.log(value + 1e-6)

def plot_points_on_heatmap(file_path, output_folder):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse lines into points
    points = [tuple(map(float, line.strip().split(':'))) for line in lines]

    # Generate heatmap data
    x_vals = np.linspace(-6, 6, 500)
    y_vals = np.linspace(-6, 6, 500)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    z_grid = heatmap_formula(x_grid, y_grid)

    # Process points in chunks of 16
    chunk_size = 16
    for i in range(0, len(points), chunk_size):
        chunk = points[i:i + chunk_size]
        if len(chunk) < chunk_size:
            break  # Stop if fewer than 16 points remain

        # Separate x and y values
        x_points, y_points = zip(*chunk)

        # Plot the heatmap
        plt.figure(figsize=(8, 8))
        plt.contourf(x_grid, y_grid, z_grid, levels=50, cmap='viridis')
        plt.colorbar(label='Heatmap Value')

        # Overlay the points
        plt.scatter(x_points, y_points, color='red', label='Points', edgecolor='black')
        plt.title(f'Points {i + 1} to {i + chunk_size}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.grid(True)
        plt.legend()

        # Save the plot as an image
        output_file = f"{output_folder}/plot_{i // chunk_size + 1}.png"
        plt.savefig(output_file)
        plt.close()

def images_to_video(image_folder, output_video, fps=10):
    # Get the list of image files in the folder, sorted by name
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    if not images:
        print("No images found in the folder!")
        return

    # Read the first image to get the size
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Add images to the video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the video writer
    video.release()
    print(f"Video saved to {output_video}")

# Example usage:
plot_points_on_heatmap('Log/Himmeblau.txt', 'Log/draw image')
# images_to_video('Log/draw image', 'Log/output_video.mp4', fps=10)
