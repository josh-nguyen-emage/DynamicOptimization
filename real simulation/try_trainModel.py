import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import keras.backend as K

def read_file(filename):
    with open(filename, 'r') as file:
        first_values = []
        secondValue = []
        last_values = []
        for line in file:
            # Split the line by colon ":"
            parts = line.strip().split(':')
            if len(parts) == 3:
                first_values.append(list(map(float, parts[0].split())))
                secondValue.append(list(map(float, parts[1].split())))
                last_values.append(list(map(float, parts[2].split())))
            else:
                print("Invalid line:")
    return np.array(first_values), np.array(secondValue),np.array(last_values)

def accuracy(y_true, y_pred):
    threshold = 0.1  # Define a threshold for accuracy calculation
    diff = K.abs(y_true - y_pred)
    return K.mean(K.cast(K.less_equal(diff, threshold), dtype='float32'))

def ReadLabFile(filename):
    list_a = []
    list_b = []
    list_c = []

    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split()
            if len(values) != 3:
                print(f"Ignoring line: {line.strip()}. Expected 3 values per line.")
                continue
            try:
                a, b, c = map(float, values)
                list_a.append(a)
                list_b.append(b)
                list_c.append(c)
            except ValueError:
                print(f"Ignoring line: {line.strip()}. Could not convert values to floats.")

    return list_a, list_b, list_c


# ------------------------------------------

if __name__ == "__main__":

    filename = 'G7-Uni-AxialTest.dat'  # Replace 'data.txt' with your file path
    list_a, list_b, list_c = ReadLabFile(filename)
    list_c = np.array(list_c)*(-100)
    list_a = np.array(list_a)*0.01

    X, Y, Z = read_file("Log_Run_A_1_0304.txt")
    Y = Y[:,1:]
    Z = Z[:,1:]
    Y *= -100
    Z *= -0.01


    TrainVal = np.concatenate((X, Y), axis=1)

    split_index = 32

    X_train, X_test = TrainVal[:split_index], TrainVal[split_index:]
    y_train, y_test = Z[:split_index], Z[split_index:]

    model = Sequential([
        Dense(256, input_shape=(60,), activation='relu'),  # First fully connected layer
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(52, activation='linear')                    # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=[accuracy])  # Using Mean Squared Error loss for regression

    # Train the model
    history = model.fit(X_train, y_train, epochs=128, batch_size=32, validation_data=(X_test, y_test))  # Adjust epochs and batch size as needed

    model.save("last.h5")

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

