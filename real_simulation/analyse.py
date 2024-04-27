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

Y_exp = list_c
Z_exp = list_a

X, Y, Z = read_file("D:/1 - Study/6 - DTW_project/RunLog/Log_Run_A_1_0304.txt")
Y = Y[:,1:]
Z = Z[:,1:]
Y *= -100
Z *= -0.01

index = 126

# Load the model
model = load_model('last.h5')

testcase = np.concatenate((X[index],Y_exp[:50]))

# Make predictions
predictions = model.predict(np.array([testcase],dtype="float32"))

index = 126
plt.plot(Y_exp[:50],Z_exp[:50], label='Simulation Line')
plt.plot(Y_exp[:50],predictions[0][:50], label='Predict Line')
plt.xlabel('Strain')
plt.ylabel('Stress')
plt.title('Predict - Simulation compare')
plt.legend()
plt.show()
