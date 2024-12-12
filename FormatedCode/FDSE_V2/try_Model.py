import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# Read data from files
def read_shape_data(param_file, value_file):
    shape_param = np.loadtxt(param_file, dtype=float)
    shape_value = np.loadtxt(value_file, dtype=float)
    return shape_param, shape_value

# File paths (replace with your actual file paths)
param_file = 'FDSE_V2/shapeParam.txt'
value_file = 'FDSE_V2/shapeValue.txt'

# Load data
shapeParam, shapeValue = read_shape_data(param_file, value_file)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(shapeParam, shapeValue, test_size=0.2, random_state=42)

# Build a simple DNN model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='linear')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# Evaluate the model
eval_results = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {eval_results[0]:.4f}, Test MAE: {eval_results[1]:.4f}")

# Save the trained model
model.save('dnn_model.h5')

# Example prediction
sample_input = X_test[:5]  # Take first 5 samples from test set
predictions = model.predict(sample_input)
print("Predictions:", predictions)
print("Actual Values:", y_test[:5])
