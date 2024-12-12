import numpy as np
import tensorflow as tf
from keras import models, layers
import matplotlib.pyplot as plt

# Define the Himmelblau function
def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Generate input data (x, y) and output data (Himmelblau value)
def generate_data(samples=10000):
    x = np.random.uniform(-6, 6, samples)
    y = np.random.uniform(-6, 6, samples)
    z = himmelblau(x, y)
    return np.stack((x, y), axis=1), z

# Generate training and testing data
X = np.loadtxt("Log/x.txt", delimiter=',')
y = np.loadtxt("Log/y.txt", delimiter=',')

for i in range(len(X)-10,len(X)):
    x_val = X[i][0]*12-6
    y_val = X[i][1]*12-6
    expectResult = (x_val**2+y_val-11)**2 +(x_val+y_val**2-7)**2
    print(x_val,y_val,y[i],expectResult)

X_train, X_test = X[:170], X[170:]
y_train, y_test = y[:170], y[170:]

# Build the model
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
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=1, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Test and visualize the model's predictions
def test_model():
    x_test = np.linspace(-6, 6, 100)
    y_test = np.linspace(-6, 6, 100)
    x_mesh, y_mesh = np.meshgrid(x_test, y_test)
    z_true = himmelblau(x_mesh, y_mesh)

    # Prepare test input for the model
    test_points = np.c_[x_mesh.ravel(), y_mesh.ravel()]
    z_pred = model.predict(test_points).reshape(x_mesh.shape)

    # Plot true values vs predicted values
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.contourf(x_mesh, y_mesh, z_true, levels=50, cmap='viridis')
    plt.title('True Himmelblau Function')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.contourf(x_mesh, y_mesh, z_pred, levels=50, cmap='viridis')
    plt.title('Predicted Himmelblau Function')
    plt.colorbar()

    plt.show()

# Test the model
test_model()
