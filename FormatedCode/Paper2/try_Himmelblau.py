import math
import random
import sys, os
from bayes_opt import BayesianOptimization

sys.path.append(os.path.abspath(os.path.join('.')))
import numpy as np
from keras import layers, models, metrics

def create_model():
    model = models.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(2, activation='relu'),
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=[metrics.MeanSquaredError()])
    return model

def train_model(model, X_train, y_train, epochs=32, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

def predict(model, X_test):
    return model.predict(X_test, verbose=0)

def generate_random_point_within_manhattan_distance(point):
    x, y = point

    # Generate random offsets within the Manhattan distance constraint
    dx = random.uniform(-0.2, 0.2)
    dy = random.uniform(-0.2, 0.2)
    
    # Ensure the Manhattan distance is within 0.2
    while abs(dx) + abs(dy) > 0.2:
        dx = random.uniform(-0.2, 0.2)
        dy = random.uniform(-0.2, 0.2)

    # Calculate the new point
    new_x = max(0, min(1, x + dx))  # Crop x to the range [-1, 1]
    new_y = max(0, min(1, y + dy))  # Crop y to the range [-1, 1]

    return new_x, new_y

globalModel = None

def writeLog(array, filename):
    with open(filename, 'a') as file:
        for row in array:
            file.write(str(row[0]*12-6) + ":" + str(row[1]*12-6) + '\n')

def run_simulation(params):
    params = np.clip(params, 0, 1)
    resultInterpolate = [None] * len(params)  # Initialize resultInterpolate list with None
    for paramIdx in range(len(params)):
        x = params[paramIdx][0]*12-6
        y = params[paramIdx][1]*12-6
        resultInterpolate[paramIdx] = [(x**2+y-11)**2,(x+y**2-7)**2]
        # print("call himbelau",x,y,((x**2+y-11)**2+(x+y**2-7)**2)/2)
    return np.array(resultInterpolate)

class DTW:
    def __init__(self):
        self.model = create_model()
        self.container = []

    def objective_function(self,x1, x2):
        params = [x1, x2]
        params = np.clip(params,0,1)
        params = np.array(params).reshape(1, -1)
        predictVal = self.model.predict(params, verbose=0)

        MSE = (predictVal[0][0] + predictVal[0][1])/2
        self.container.append([MSE,[x1, x2]])
        return -1*MSE
    
    def getBestValue(self,numBest):
        sorted_lst = sorted(self.container, key=lambda x: x[0])
        print(sorted_lst[-10:])
        value_of_smallest = []
        seen_values = set()
        
        for value in sorted_lst:
            if value[0] not in seen_values:
                value_of_smallest.append(value[1])
                seen_values.add(value[0])
            if len(value_of_smallest) == numBest:
                break
        return value_of_smallest
    
    def checkSomeRandomArea(self):
        # currentBestValue = self.getBestValue(16)
        # for i in range(512):
        #     currentVal = currentBestValue[random.randint(0,len(currentBestValue)-1)]
        #     newVal = generate_random_point_within_manhattan_distance(currentVal)
        #     self.objective_function(newVal[0],newVal[1])
        
        for i in range(32):
            self.objective_function(random.random(),random.random())

if __name__ == "__main__":
    
    X_train = []
    y_train = []

    dtwObj = DTW()

    addX = np.random.rand(16,2)
    for eachX in addX:
        X_train.append(eachX)
    y_train = run_simulation(addX)

    print("Gen data completed")
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)

    train_model(dtwObj.model, X_train_np, y_train_np)
    counter = 0

    pbounds = {'x1': (0, 1), 'x2': (0, 1)}

    

    for _ in range(200):
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
            n_iter=30      # Số lần lặp tối ưu hóa
        )

        dtwObj.checkSomeRandomArea()
        nexVal = dtwObj.getBestValue(16)
        writeLog(nexVal,"Log\Himmeblau.txt")
        for eachVal in nexVal:
            X_train.append(eachVal)
        simulationResult = run_simulation(nexVal)
        y_train = np.concatenate((y_train,simulationResult))

        print(counter," time: Predic MSE:",optimizer.max["params"])
        MSEcollector = np.nanmean(simulationResult,1)
        print("Best MSE: ",np.min(MSEcollector))

        X_train_np = np.array(X_train)
        y_train_np = np.array(y_train)
        dtwObj.model = create_model()
        train_model(dtwObj.model, X_train_np, y_train_np)
        # dtwObj.model.save("Log\\"+str(counter)+".h5")
        # break