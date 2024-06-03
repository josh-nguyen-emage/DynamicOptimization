import sys, os
import threading

from Libary.RunSequent import RunSimulationThread
sys.path.append(os.path.abspath(os.path.join('.')))
import numpy as np
from keras import layers, models

from Libary.function import *

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
    simulationResult = RunSimulationThread(0, params)
    strain = simulationResult[0]
    stress = simulationResult[1]
    MSE, interpolate = findF(strain,stress)
    return np.array(interpolate)

def run_simulation_thread(paramIdx, param, resultInterpolate):
    simulation_result = RunSimulationThread(paramIdx, param)
    strain = simulation_result[0]
    stress = simulation_result[1]
    MSE, interpolate = findF(strain, stress)
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

def generateSeed(bestSeed):
    seedCollector = [bestSeed]
    for _ in range(2047):
        changeTime = np.random.randint(1,11)
        newSeed = np.copy(bestSeed)
        for _ in range(changeTime):
            newSeed[np.random.randint(11)] = np.random.rand()
        seedCollector.append(np.clip(newSeed,0,1))
    return np.concatenate((np.array(seedCollector),np.random.rand(2048,11)),axis=0)

def getBestValue(lst,randomSeed,numValue):
    expectChart = getExpectChart()
    offset = np.mean((lst-expectChart)**2,1)
    
    # Enumerate the list to keep track of indices
    enumerated_lst = list(enumerate(offset))
    
    # Sort the list of tuples by the second element (the values) in ascending order
    sorted_lst = sorted(enumerated_lst, key=lambda x: x[1])
    
    # Get the indices of the 8 smallest values
    value_of_smallest = [randomSeed[index] for index, _ in sorted_lst[:numValue]]
    
    return value_of_smallest

if __name__ == "__main__":
    normalizeRatio = 250

    X_train = []
    y_train = []

    expectChart = getExpectChart()

    addX = np.random.rand(16,11)
    for eachX in addX:
        X_train.append(eachX)
    simulationResult = run_simulation(addX)
    for eachY in simulationResult:
        y_train.append(eachY)

    print("Gen data completed")
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)

    model = create_model()
    train_model(model, X_train_np, y_train_np/normalizeRatio)

    counter = 0
    bestSeedIdx = np.argmin(np.mean((y_train_np-expectChart)**2,1))
    bestSeed = X_train[bestSeedIdx]
    oldMSE = np.mean((y_train_np[bestSeedIdx]-expectChart)**2)

    while 1:
        counter += 1
        randomSeed = generateSeed(bestSeed)
        predictions = predict(model, randomSeed)
        predictions = np.array(predictions)*normalizeRatio

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
        model = create_model()
        train_model(model, X_train_np, y_train_np/normalizeRatio)

        # if counter > 512:
        #     break