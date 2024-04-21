import sys, os
import threading
sys.path.append(os.path.abspath(os.path.join('.')))
import numpy as np
from keras import layers, models
import matplotlib.pyplot as plt


from RunParallel.RunSequent_Master import RunSimulationThread_WithInputVal

def create_model():
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(11,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(50, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

def predict(model, X_test):
    return model.predict(X_test, verbose=0)

def runSimulation(params):
    params = np.clip(params, 0, 1)
    threads = []
    numThread = len(params)
    y_values = [0]*numThread
    
    # Define a function to run in each thread
    def run_thread(idx, param):
        result = RunSimulationThread_WithInputVal(idx, param)
        y_values[idx] = result

    for idx in range(numThread):
        thread = threading.Thread(target=run_thread, args=(idx, params[idx]))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
        
    return np.array(y_values)

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

    expectChart = runSimulation(params=expectParams)

    addX = np.random.rand(11)
    X_train.append(addX)
    y_train.append(runSimulation(addX))

    print("Gen data completed")
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)

    model = create_model()
    train_model(model, X_train_np, y_train_np)

    counter = 0
    bestSeedIdx = np.argmin(np.sum(abs(y_train_np-expectChart),1))
    bestSeed = X_train[bestSeedIdx]
    oldOffset = min(np.sum(abs(y_train_np-expectChart),1))

    while 0:
        counter += 1
        randomSeed = generateSeed(bestSeed)
        predictions = predict(model, randomSeed)

        offset = np.sum(abs(predictions-expectChart),1)
        nexVal = randomSeed[np.argmin(offset)]

        X_train.append(nexVal)
        y_train.append(runSimulation(nexVal))

        currentOffset = np.sum(abs(runSimulation(nexVal)-expectChart))
        if currentOffset < oldOffset:
            oldOffset = currentOffset
            bestSeed = nexVal

        print(counter,"time: expect offset ",np.min(offset),"but real offset is",currentOffset," : Min offset",oldOffset)

        X_train_np = np.array(X_train)
        y_train_np = np.array(y_train)
        model = create_model()
        train_model(model, X_train_np, y_train_np)

        if counter > 16:
            break
        if abs(currentOffset) < 1:
            print("FUCK yeah")
            break