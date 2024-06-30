import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))
from Phase1 import *
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras import models, layers, metrics

def accuracy(y_true, y_pred):
    threshold = 0.01  # Define a threshold for accuracy calculation
    diff = K.abs(y_true - y_pred)
    return K.mean(K.cast(K.less_equal(diff, threshold), dtype='float32'))

returnVal = read_file("D:\\1 - Study\\6 - DTW_project\\Container\\Log_Run_Burning_2.txt")

param = returnVal[0]
strain = np.array(returnVal[1])/(-4)
stress = np.array(returnVal[2])/(300)
bodyOpen = np.array(returnVal[3])/(20)

TrainVal = [[ai, bi, ci] for ai, bi, ci in zip(strain, stress, bodyOpen)]

TrainVal = np.array(TrainVal)

split_index = 17920

X_train, X_test = param[:split_index], param[split_index:]
y_train, y_test = TrainVal[:split_index], TrainVal[split_index:]


model = models.Sequential([
    # Convolutional layers
    layers.Conv1D(64, 3, activation='relu', padding='same', input_shape=(11, 1)),
    layers.Conv1D(128, 3, activation='relu', padding='same'),
    layers.MaxPooling1D(2),
    
    layers.Conv1D(256, 3, activation='relu', padding='same'),
    layers.Conv1D(256, 3, activation='relu', padding='same'),
    layers.MaxPooling1D(2),
    
    layers.Conv1D(512, kernel_size=5, strides=1, activation='relu', padding='same'),
    layers.Conv1D(512, kernel_size=5, strides=1, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    
    # Fully connected layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(153, activation='sigmoid'),
    
    # Reshape the output to the desired shape (3, 51)
    layers.Reshape((3, 51))
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=[metrics.MeanSquaredError(),accuracy])
# Train the model
history = model.fit(X_train, y_train, epochs=64, batch_size=256, validation_data=(X_test, y_test))  # Adjust epochs and batch size as needed

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
model.save("Model\\last.h5")
# save_lists_to_file(history.history['accuracy'],history.history['val_accuracy'],"pltDraw.txt")

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()