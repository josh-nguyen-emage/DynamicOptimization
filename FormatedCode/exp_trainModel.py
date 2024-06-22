from Phase1 import *
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K

def accuracy(y_true, y_pred):
    threshold = 0.01  # Define a threshold for accuracy calculation
    diff = K.abs(y_true - y_pred)
    return K.mean(K.cast(K.less_equal(diff, threshold), dtype='float32'))

def save_lists_to_file(list1, list2, filename):
    with open(filename, 'w') as file:
        for item in list1:
            file.write(f"{item}\n")
        file.write("===\n")  # Separator between lists
        for item in list2:
            file.write(f"{item}\n")


returnVal = read_file("D:\\1 - Study\\6 - DTW_project\\Container\\Log_Run_Burning_1.txt")

param = returnVal[0]
strain = np.array(returnVal[1])/(-4)
stress = np.array(returnVal[2])/(300)
bodyOpen = np.array(returnVal[3])/(20)

TrainVal = np.concatenate((strain, stress, bodyOpen),axis=1)

split_index = 8096

X_train, X_test = param[:split_index], param[split_index:]
y_train, y_test = TrainVal[:split_index], TrainVal[split_index:]

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(11,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(153, activation='relu'),
    layers.Dense(153, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=[accuracy])  # Using Mean Squared Error loss for regression
# Train the model
history = model.fit(X_train, y_train, epochs=64, batch_size=256, validation_data=(X_test, y_test))  # Adjust epochs and batch size as needed

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
model.save("last.h5")
save_lists_to_file(history.history['accuracy'],history.history['val_accuracy'],"pltDraw.txt")

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()