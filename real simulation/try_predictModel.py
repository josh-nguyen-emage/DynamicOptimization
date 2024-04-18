from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
import keras.backend as K

from try_trainModel import ReadLabFile, read_file

# ------------------------------------------

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

# Load the model
model = load_model('last.h5')

# Make predictions
predictions = model.predict(np.array(TrainVal,dtype="float32"))

for i in range(10):
    index = -1*i
    plt.plot(TrainVal[index][-50:],Z[index][:50], label='Simulation Line')
    plt.plot(TrainVal[index][-50:],predictions[index][:50], label='Predict Line')
    plt.xlabel('Strain')
    plt.ylabel('Stress')
    plt.title('Predict - Simulation compare')
    plt.legend()
    plt.show()
