import sys, os
import threading

sys.path.append(os.path.abspath(os.path.join('.')))
import numpy as np
from keras import layers, models


from Libary.RunSequent import RunSimulationThread
from Libary.function import *

def create_model():
    model = models.Sequential([
        layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(y_train.shape[1], activation='linear')  # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

def predict(model, X_test):
    return model.predict(X_test, verbose=0)

def run_simulation_thread(paramIdx, param, resultInterpolate):
    simulation_result = RunSimulationThread(paramIdx, param)
    strain = simulation_result[0]
    stress = simulation_result[1]
    bodyOpen = simulation_result[2]
    feature = findFeatureVal(stress, bodyOpen, strain)
    resultInterpolate[paramIdx] = feature

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

def generateSeed(bestSeed):
    seedCollector = [bestSeed]
    return np.concatenate((np.array(seedCollector),np.random.rand(4096,11)),axis=0)

def getBestValue(lst,randomSeed,numValue):
    expectChart = getExpectFeature()
    offset = np.mean((lst-expectChart)**2,1)
    enumerated_lst = list(enumerate(offset))
    sorted_lst = sorted(enumerated_lst, key=lambda x: x[1])
    value_of_smallest = [randomSeed[index] for index, _ in sorted_lst[:numValue]]
    
    return value_of_smallest


if __name__ == "__main__":

    X_train = []
    y_train = []

    expectChart = getExpectFeature()

    addX = np.random.rand(32,11)
    for eachX in addX:
        X_train.append(eachX)
    simulationResult = run_simulation(addX)
    for eachY in simulationResult:
        y_train.append(eachY)

    print("Gen data completed")
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)

    model = create_model()
    train_model(model, X_train_np, y_train_np)

    counter = 0
    bestSeedIdx = np.argmin(np.mean((y_train_np-expectChart)**2,1))
    bestSeed = X_train[bestSeedIdx]
    oldMSE = np.mean((y_train_np[bestSeedIdx]-expectChart)**2)

    while 1:
        counter += 1
        randomSeed = generateSeed(bestSeed)
        predictions = predict(model, randomSeed)
        predictions = np.array(predictions)

        nexVal = getBestValue(predictions,randomSeed,16)
        for eachVal in nexVal:
            X_train.append(eachVal)
        simulationResult = run_simulation(nexVal)
        for idx, eachResult in enumerate(simulationResult):
            y_train.append(eachResult)

            currentMSE = np.mean((eachResult-expectChart)**2)
            if currentMSE < oldMSE:
                oldMSE = currentMSE
                bestSeed = nexVal[idx]

        print(counter,"time: real offset is",currentMSE," : Min offset",oldMSE)

        X_train_np = np.array(X_train)
        y_train_np = np.array(y_train)
        train_model(model, X_train_np, y_train_np)

        # if counter > 512:
        #     break