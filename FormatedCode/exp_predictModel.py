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


returnVal = read_file("D:\\1 - Study\\6 - DTW_project\\Container\\Log_Run_Burning_A_1.txt")

param = returnVal[0]
strain = np.array(returnVal[1])/(-4)
stress = np.array(returnVal[2])/(300)
bodyOpen = np.array(returnVal[3])/(20)

TrainVal = [[ai, bi, ci] for ai, bi, ci in zip(strain, stress, bodyOpen)]

TrainVal = np.array(TrainVal)

model = load_model('Model/last.h5')

predictions = model.predict(np.array(param,dtype="float32"))

for i in range(10):
    index = 5+i
    plt.plot(predictions[index][0]*(-4),predictions[index][1]*(300), label='Predict Line', color="red")
    plt.plot(predictions[index][2]*(20),predictions[index][1]*(300), color="red")
    
    plt.plot(TrainVal[index][0]*(-4),TrainVal[index][1]*(300), label='Simulation Line', color="blue")
    plt.plot(TrainVal[index][2]*(20),TrainVal[index][1]*(300), color="blue")

    # plt.title('Predict - Simulation compare')
    plt.legend()
    plt.show()

# for i in range(10):
#     index = 8096+i
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
    
#     ax1.plot(TrainVal[index][:51]*(-4),TrainVal[index][51:102]*(300), label='Simulation Line', color="blue")
#     ax1.plot(TrainVal[index][102:]*(20),TrainVal[index][51:102]*(300), color="blue")

#     ax2.plot((TrainVal[index][:51]*(-4)),((TrainVal[index][102:]*(20)/TrainVal[index][:51]*(-4))), label='Predict Line', color="red")

#     ax1.set_xlabel('Strain')
#     ax1.set_ylabel('Stress')
#     ax1.set_title("Simulation")

#     ax2.set_xlabel('Strain')
#     ax2.set_ylabel('Strain + /Strain -')
#     ax1.set_title("V plot")
#     # plt.title('Predict - Simulation compare')
#     plt.legend()
#     plt.show()