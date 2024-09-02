import sys, os
import threading
from bayes_opt import BayesianOptimization

sys.path.append(os.path.abspath(os.path.join('.')))
import numpy as np
from keras import layers, models, metrics
from Libary.function import *
from keras.models import load_model
from Libary.RunSequent import RunSimulationThread

def create_model():
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
    layers.Dense(150, activation='sigmoid'),
    
    # Reshape the output to the desired shape (3, 51)
    layers.Reshape((2, 75))
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=[metrics.MeanSquaredError()])
    return model

def train_model(model, X_train, y_train, epochs=32, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

def predict(model, X_test):
    return model.predict(X_test, verbose=0)

globalModel = None

if check_mac_address("48-5F-99-56-F4-36"):
    globalModel = load_model("Model/std_1.h5")

def run_simulation_thread(paramIdx, param, resultInterpolate):
    simulation_result = RunSimulationThread(paramIdx, param)
    strain = simulation_result[0]
    stress = simulation_result[1]
    bodyOpen = simulation_result[2]
    MSE, interpolate = findF(stress, bodyOpen, strain)
    if np.isnan(MSE):
        raise ValueError("MSE is nan")
    resultInterpolate[paramIdx] = interpolate

def run_simulation(params):
    params = np.clip(params, 0, 1)
    resultInterpolate = [None] * len(params)  # Initialize resultInterpolate list with None

    if check_mac_address("48-5F-99-56-F4-36"):
        resultPredict = globalModel.predict(np.array(params,dtype="float32"))
        resultPredict[:, 0, :] *= -1
        resultPredict[:, 1, :] *= 300
        resultPredict[:, 2, :] *= 20

        for paramIdx in range(len(resultPredict)):
            simulation_result = resultPredict[paramIdx]
            strain = simulation_result[0]
            stress = simulation_result[1]
            bodyOpen = simulation_result[2]
            save_to_file(params[paramIdx],simulation_result,"D:\\1 - Study\\6 - DTW_project\\Container\\Log_Run_Bayes_AI_2-7.txt")

            MSE, interpolate = findF(stress, bodyOpen, strain)
            if np.isnan(MSE):
                raise ValueError("MSE is nan")
            resultInterpolate[paramIdx] = interpolate
    else:
        threads = []
        for paramIdx in range(len(params)):
            thread = threading.Thread(target=run_simulation_thread, args=(paramIdx, params[paramIdx], resultInterpolate))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to finish
        for thread in threads:
            thread.join()
    
    return np.array(resultInterpolate)


class DTW:
    def __init__(self):
        self.model = create_model()
        self.container = []

    def objective_function(self,x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
        params = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]
        params = np.clip(params,0,1)
        params = np.array(params).reshape(1, -1)  # Reshape to (1, 11)
        predictVal = self.model.predict(params, verbose=0)

        strain = np.array(predictVal[0][0])*(300)
        stress = np.array(predictVal[0][1])*(300)

        predictStrainVal = np.concatenate((strain,stress))
        expectChart = getExpectChart()
        MSE = np.nanmean((predictStrainVal-expectChart)**2)

        self.container.append([MSE,[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]])
        return -1*MSE
    
    def getBestValue(self,numBest):
        sorted_lst = sorted(self.container, key=lambda x: x[0])
        value_of_smallest = []
        seen_values = set()
        
        for value in sorted_lst:
            if value[0] not in seen_values:
                value_of_smallest.append(value[1])
                seen_values.add(value[0])
            if len(value_of_smallest) == numBest:
                break
        return value_of_smallest
    
def preprocessTrainVal(simulationVal):
    strain = np.array(simulationVal[:,:75])/(300)
    bodyOpen = np.array(simulationVal[:,75:])/(300)

    TrainVal = [[ai, ci] for ai, ci in zip(strain, bodyOpen)]

    TrainVal = np.array(TrainVal)
    return TrainVal

if __name__ == "__main__":
    

    X_train = []
    y_train = []

    expectChart = getExpectChart()
    dtwObj = DTW()

    addX = np.random.rand(16,11)
    for eachX in addX:
        X_train.append(eachX)
    simulationResult = run_simulation(addX)
    y_train = preprocessTrainVal(simulationResult)

    print("Gen data completed")
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)

    train_model(dtwObj.model, X_train_np, y_train_np)
    counter = 0

    pbounds = {'x1': (0, 1), 'x2': (0, 1), 'x3': (0, 1), 'x4': (0, 1), 'x5': (0, 1),
    'x6': (0, 1), 'x7': (0, 1), 'x8': (0, 1), 'x9': (0, 1), 'x10': (0, 1), 'x11': (0, 1)
    }

    

    while 1:
        counter += 1

        optimizer = BayesianOptimization(
                f=dtwObj.objective_function,
                pbounds=pbounds,
                random_state=1,
                allow_duplicate_points=True
            )
        
        dtwObj.container = []
        # Thực hiện quá trình tối ưu hóa
        optimizer.maximize(
            init_points=16,  # Số lượng điểm khởi tạo ngẫu nhiên
            n_iter=100      # Số lần lặp tối ưu hóa
        )

        nexVal = dtwObj.getBestValue(16)
        for eachVal in nexVal:
            X_train.append(eachVal)
        simulationResult = run_simulation(nexVal)
        y_train = np.concatenate((y_train,preprocessTrainVal(simulationResult)))

        print(counter," time: Predic MSE:",optimizer.max["params"])
        expectChart = getExpectChart()
        MSEcollector = np.nanmean((simulationResult-expectChart)**2,1)
        print("Best MSE: ",np.min(MSEcollector))

        X_train_np = np.array(X_train)
        y_train_np = np.array(y_train)
        dtwObj.model = create_model()
        train_model(dtwObj.model, X_train_np, y_train_np)