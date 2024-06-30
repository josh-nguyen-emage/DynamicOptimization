import sys, os
import threading
from bayes_opt import BayesianOptimization
from Libary.RunSequent import RunSimulationThread
sys.path.append(os.path.abspath(os.path.join('.')))
import numpy as np
from keras import layers, models
from Libary.function import *

normalizeRatio = 250
def create_model():
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(11,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(75, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

def predict(model, X_test):
    return model.predict(X_test, verbose=0)

def runSingleSimulation(params):
    params = np.clip(params, 0, 1)
    simulation_result = RunSimulationThread(0, params)
    strain = simulation_result[0]
    stress = simulation_result[1]
    bodyOpen = simulation_result[2]
    MSE, interpolate = findF(stress, bodyOpen, strain)
    return np.array(interpolate)

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
    
    # Create and start threads
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
        predictVal = np.array(predictVal[0])*normalizeRatio
        expectChart = getExpectChart()
        MSE = np.nanmean((predictVal-expectChart)**2)
        self.container.append([MSE,[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]])
        return -1*MSE
    
    def getBestValue(self,numBest):
        sorted_lst = sorted(self.container, key=lambda x: x[0])
        value_of_smallest = [value[1] for value in sorted_lst[:numBest]]
        return value_of_smallest

if __name__ == "__main__":
    

    X_train = []
    y_train = []

    expectChart = getExpectChart()
    dtwObj = DTW()

    addX = np.random.rand(16,11)
    for eachX in addX:
        X_train.append(eachX)
    simulationResult = run_simulation(addX)
    for eachY in simulationResult:
        y_train.append(eachY)

    print("Gen data completed")
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)

    train_model(dtwObj.model, X_train_np, y_train_np/normalizeRatio)
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
            init_points=10,  # Số lượng điểm khởi tạo ngẫu nhiên
            n_iter=300      # Số lần lặp tối ưu hóa
        )

        # In kết quả tối ưu
        # print("optimizer.max : ",optimizer.max)

        nexVal = dtwObj.getBestValue(16)
        for eachVal in nexVal:
            X_train.append(eachVal)
        simulationResult = run_simulation(nexVal)
        for idx, eachResult in enumerate(simulationResult):
            y_train.append(eachResult)

        print(counter," time: Predic MSE:",optimizer.max["params"])
        expectChart = getExpectChart()
        MSEcollector = np.nanmean((simulationResult-expectChart)**2,1)
        print("Best MSE: ",np.min(MSEcollector))

        f = open("bayesLog.txt", "a")
        for eachMSE in MSEcollector:
            f.write(str(eachMSE) + " ")
        f.write(str(-1*optimizer.max["target"]) + "\n")
        f.close()

        X_train_np = np.array(X_train)
        y_train_np = np.array(y_train)
        dtwObj.model = create_model()
        train_model(dtwObj.model, X_train_np, y_train_np/normalizeRatio)
