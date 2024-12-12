import numpy as np
from keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def train_himmelblau_model(file_path, epochs=10, batch_size=1):
    # Load data from file
    data = np.loadtxt(file_path, delimiter=':')
    X = data[:, :2]  # First two columns are x and y
    y = (X[:, 0]**2 + X[:, 1] - 11)**2 + (X[:, 0] + X[:, 1]**2 - 7)**2  # Himmelblau function
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the model
    model = models.Sequential([
        layers.Dense(8, activation='relu', input_shape=(2,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='relu')
    ])
    
    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Save the model
    # model.save('himmelblau_model.h5')
    # print("Model saved as himmelblau_model.h5")
    
    # Draw prediction map
    plot_prediction_map(model)
    
    # Draw training points
    plot_training_points(X_train)
    
    return model

def plot_prediction_map(model):
    # Create a grid in the range [-6, 6]
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    input_data = np.stack([X.ravel(), Y.ravel()], axis=-1)
    
    # Predict using the model
    Z = model.predict(input_data).reshape(X.shape)
    
    # Plot the prediction map
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(label="Prediction")
    plt.title("Prediction Map")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.show()

def plot_training_points(X_train):
    # Scatter plot of training points
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], color='red', s=10, label='Training Points')
    plt.title("Training Data Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.legend()
    plt.show()

# Example usage:
train_himmelblau_model("D:\\1 - Study\\6 - DTW_project\\FormatedCode\\Log\\Himmeblau.txt", epochs=10, batch_size=1)
