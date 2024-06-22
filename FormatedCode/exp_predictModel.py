from Phase1 import *
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import load_model

def accuracy(y_true, y_pred):
    threshold = 0.02  # Define a threshold for accuracy calculation
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

model = load_model('last.h5')

predictions = model.predict(np.array(param,dtype="float32"))

for i in range(10):
    index = -1*i
    plt.plot(predictions[index][:51]*(-4),predictions[index][51:102]*(300), label='Predict Line', color="red")
    plt.plot(predictions[index][102:]*(20),predictions[index][51:102]*(300), color="red")
    
    plt.plot(TrainVal[index][:51]*(-4),TrainVal[index][51:102]*(300), label='Simulation Line', color="blue")
    plt.plot(TrainVal[index][102:]*(20),TrainVal[index][51:102]*(300), color="blue")

    plt.xlabel('Strain')
    plt.ylabel('Stress')
    plt.title('Predict - Simulation compare')
    plt.legend()
    plt.show()