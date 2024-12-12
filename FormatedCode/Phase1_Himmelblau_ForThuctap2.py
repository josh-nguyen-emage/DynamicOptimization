import sys, os
import threading

sys.path.append(os.path.abspath(os.path.join('.')))
import numpy as np
from keras import layers, models, optimizers
import matplotlib.pyplot as plt
from Libary.function import *

def create_model():
    model = models.Sequential([
        layers.Dense(8, activation='relu', input_shape=(2,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='relu')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=2):
    # np.savetxt("Log/x.txt", X_train, delimiter=',', fmt='%f')
    # np.savetxt("Log/y.txt", y_train, delimiter=',', fmt='%f')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    # model.save("Log\\last.h5")
    return model

def predict(model, X_test):
    return model.predict(X_test, verbose=0)


def run_simulation_thread(paramIdx, param, resultInterpolate):
    x = param[0]*12-6
    y = param[1]*12-6
    resultInterpolate[paramIdx] = (x**2+y-11)**2 +(x+y**2-7)**2

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
    return np.concatenate((np.array(seedCollector),np.random.rand(4095,2)),axis=0)

def getBestValue(lst, randomSeed, numValue):
    def is_valid_point(candidate, selected_points, min_distance=0.04):
        return all(np.linalg.norm(candidate - point) >= min_distance for point in selected_points)
    
    enumerated_lst = list(enumerate(lst))
    sorted_lst = sorted(enumerated_lst, key=lambda x: x[1], reverse=False)  # Sort by label (lst) in descending order

    selected_points = []
    for index, _ in sorted_lst:
        if len(selected_points) >= numValue:
            break
        candidate = randomSeed[index]
        if is_valid_point(candidate, selected_points):
            selected_points.append(candidate)

    return np.array(selected_points)

def writeLog(array, filename):
    with open(filename, 'a') as file:
        for row in array:
            file.write(str(row[0]*12-6) + ":" + str(row[1]*12-6) + '\n')

def heatmap_formula(x, y):
    # Compute the original formula
    value = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    # Apply logarithm to the result, adding a small constant to avoid log(0)
    return np.log(value + 1e-6)

def plot_points_on_heatmap(chunk, counter, model):

    # Generate heatmap data
    x_vals = np.linspace(-6, 6, 100)
    y_vals = np.linspace(-6, 6, 100)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    z_grid = heatmap_formula(x_grid, y_grid)

    x_points, y_points = zip(*chunk)

    # Plot the heatmap
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.contourf(x_grid, y_grid, z_grid, levels=50, cmap='viridis')
    plt.colorbar(label='Heatmap Value')

    # Overlay the points
    plt.scatter(np.array(x_points)*12-6, np.array(y_points)*12-6, color='red', label='Points', edgecolor='black')
    plt.title(f'{counter}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.grid(True)
    plt.legend()

    x_vals = np.linspace(0, 1, 100)
    y_vals = np.linspace(0, 1, 100)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    test_points = np.c_[x_grid.ravel(), y_grid.ravel()]
    z_pred = model.predict(test_points).reshape(x_grid.shape)

    plt.subplot(1, 2, 2)
    plt.contourf(x_grid, y_grid, z_pred, levels=50, cmap='viridis')
    plt.title('Predicted Himmelblau Function')
    plt.colorbar()

    # Save the plot as an image
    output_file = f"Log/draw image/plot_{counter}.png"
    plt.savefig(output_file)
    plt.close()
        


if __name__ == "__main__":
    X_train = []
    y_train = []

    addX = np.random.rand(32,2)
    for eachX in addX:
        X_train.append(eachX)
    simulationResult = run_simulation(addX)
    for eachY in simulationResult:
        y_train.append(eachY)

    print("Gen data completed")
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)

    model = create_model()
    model = train_model(model, X_train_np, y_train_np)

    plot_points_on_heatmap(addX,0,model)

    counter = 0
    bestSeedIdx = np.argmin(y_train_np)
    bestSeed = X_train[bestSeedIdx]
    oldMSE = y_train_np[bestSeedIdx]

    while 1:
        counter += 1
        randomSeed = generateSeed(bestSeed)
        predictions = predict(model, randomSeed)
        predictions = np.array(predictions)

        nexVal = getBestValue(predictions,randomSeed,16)
        writeLog(nexVal,"Log\Himmeblau.txt")
        plot_points_on_heatmap(nexVal,counter,model)
        for eachVal in nexVal:
            X_train.append(eachVal)
        simulationResult = run_simulation(nexVal)
        for idx, eachResult in enumerate(simulationResult):
            y_train.append(eachResult)
            # print("think ",nexVal[idx],"is",predictions[idx],"but it was",eachResult)

            currentMSE = eachResult
            if currentMSE < oldMSE:
                oldMSE = currentMSE
                bestSeed = nexVal[idx]

        print("y_train:", y_train[-16:])

        print(counter,"time: real offset is",currentMSE," : Min offset",oldMSE)

        X_train_np = np.array(X_train)
        y_train_np = np.array(y_train)
        # model = create_model()
        model = train_model(model, X_train_np, y_train_np)
        # model.save("Log\\model\\"+str(counter)+".h5")
            

        # if counter > 5:
        #     break