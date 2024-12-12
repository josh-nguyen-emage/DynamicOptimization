import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# Define the folder containing the models and output directory
model_folder = "D:\\1 - Study\\6 - DTW_project\\FormatedCode\\Log\\model"
output_folder = "D:\\1 - Study\\6 - DTW_project\\FormatedCode\\Log\\outputModel"
os.makedirs(output_folder, exist_ok=True)

# Define the input space
y = np.linspace(-6, 6, 100)
x = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x, y)
input_data = np.stack([X.ravel(), Y.ravel()], axis=-1)

# Process each model
for model_file in os.listdir(model_folder):
    if model_file.endswith(".h5"):
        model_path = os.path.join(model_folder, model_file)
        model = load_model(model_path)

        # Predict using the model
        predictions = model.predict(input_data)
        Z = predictions.reshape(X.shape)

        # Plot the results
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, levels=50, cmap="viridis")
        plt.colorbar(label="Prediction")
        plt.title(f"Predictions from {model_file}")
        plt.xlabel("X")
        plt.ylabel("Y")

        # Save the plot
        output_path = os.path.join(output_folder, f"{model_file}.png")
        plt.savefig(output_path)
        plt.close()

print("Plots saved to:", output_folder)
