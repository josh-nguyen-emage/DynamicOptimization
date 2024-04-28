import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))
import numpy as np
from keras import layers, models
import matplotlib.pyplot as plt

from real_simulation.RunParallel.RunSequent_Master import RunSimulationThread
from real_simulation.GlobalLib import findF, getExpectChart

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

def runSimulation(params):
    params = np.clip(params, 0, 1)
    simulationResult = RunSimulationThread(0, params)
    strain = simulationResult[0]
    stress = simulationResult[1]
    strain = -1000*np.array(strain)
    stress = -1*np.array(stress)
    MSE, interpolate = findF(strain,stress)
    return np.array(interpolate)

def generateSeed(bestSeed):
    seedCollector = [bestSeed]
    for _ in range(511):
        changeTime = np.random.randint(1,11)
        newSeed = np.copy(bestSeed)
        for _ in range(changeTime):
            newSeed[np.random.randint(11)] += np.random.rand()
        seedCollector.append(np.clip(newSeed,0,1))
    return np.concatenate((np.array(seedCollector),np.random.rand(512,11)),axis=0)

if __name__ == "__main__":
    X_train = []
    y_train = []

    expectChart = getExpectChart()

    addX = np.random.rand(11)
    X_train.append(addX)
    y_train.append(runSimulation(addX))

    print("Gen data completed")
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)

    model = create_model()
    train_model(model, X_train_np, y_train_np)

    counter = 0
    bestSeedIdx = np.argmin(np.sum((y_train_np-expectChart)**2,1))
    bestSeed = X_train[bestSeedIdx]
    oldOffset = min(np.sum((y_train_np-expectChart)**2,1))

    while 1:
        counter += 1
        randomSeed = generateSeed(bestSeed)
        predictions = predict(model, randomSeed)

        offset = np.sum((predictions-expectChart)**2,1)
        nexVal = randomSeed[np.argmin(offset)]

        X_train.append(nexVal)
        simulationResult = runSimulation(nexVal)
        y_train.append(simulationResult)

        currentOffset = np.sum((simulationResult-expectChart)**2)
        if currentOffset < oldOffset:
            oldOffset = currentOffset
            bestSeed = nexVal

        print(counter,"time: expect offset ",np.min(offset),"but real offset is",currentOffset," : Min offset",oldOffset)

        X_train_np = np.array(X_train)
        y_train_np = np.array(y_train)
        model = create_model()
        train_model(model, X_train_np, y_train_np)

        if counter > 128:
            break