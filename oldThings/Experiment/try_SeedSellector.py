import sys
import numpy as np
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from FlyObjectPlot import simulateFunction

def create_model():
    model = models.Sequential([
        layers.Dense(10, activation='relu', input_shape=(10,)),
        layers.Dense(10, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(10, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

def predict(model, X_test):
    return model.predict(X_test, verbose=0)

# def f(x,params):
#     params = np.array(params) - 0.5
#     return ((params[0]*x+params[1])*(params[2]*x+params[3]))/((params[4]*x+params[5])*(params[6]*x+params[7])) + params[8]*x+params[9]

# def f(x,params):
#     params = np.array(params) - 0.5
#     return ((params[0]*x+params[1])*(params[2]*x+params[3]))/((params[4]*x+params[5])*(params[6]*x+params[7])) + params[8]*x**2+params[9]*x


# def runSimulation(params):
#     x_values = np.linspace(-10, 10, 10)
#     y_values = f(x_values,params)
#     return y_values

def runSimulation(params):
    y_values = simulateFunction(np.clip(params,0,1))
    return np.array(y_values)

def generateSeed(bestSeed):
    seedCollector = [bestSeed]
    for _ in range(511):
        changeTime = np.random.randint(1,10)
        newSeed = np.copy(bestSeed)
        for _ in range(changeTime):
            newSeed[np.random.randint(10)] += np.random.rand()
        seedCollector.append(np.clip(newSeed,0,1))
    return np.concatenate((np.array(seedCollector),np.random.rand(512,10)),axis=0)

algoResultCollector = []
def objective_function(params):
    global expectChart
    generated_plot = runSimulation(params)
    # Calculate difference between generated plot and expected plot
    difference = np.sum((generated_plot - expectChart) ** 2)
    algoResultCollector.append(difference)
    return difference


X_train = []
y_train = []

expectParams = [0.4091833262103374, 0.5130220281545994, 0.8026017156789826, 0.7977263023546852, 0.1264259278175166, 0.3861428795291987, 0.8570859068716669, 0.061997980332113456, 0.8970020561039936, 0.2274497039245046]
# expectParams = [0.15273518928143348, 0.11941661473886944, 0.6032405353477348, 0.9793515631403089, 0.10801887334150606, 0.8131765148972823, 0.46985111712825256, 0.07402215620542729, 0.6544830022065109, 0.48228580118399533]
# expectParams = [0.3574996668831032, 0.20047580770744522, 0.779075702437577, 0.5256825876454524, 0.2965101758921993, 0.04949707456837571, 0.45227470024293703, 0.13945057332811794, 0.7943646984018623, 0.9323721294838536]

expectChart = runSimulation(params=expectParams)

#Init
for _ in range(16):
    addX = np.random.rand(10)
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

result = minimize(objective_function, bestSeed, method='SLSQP')

print(bestSeed)
algoResultCollector = np.clip(algoResultCollector,0,10)
plt.plot(range(len(algoResultCollector)), algoResultCollector, label='Algo output', color='green', linestyle='-')
plt.legend()

# Displaying the plot
plt.grid(True)  # Add grid
plt.show()

plt.clf()

# Plotting the first line
plt.plot(range(10), expectChart, label='Real Data', color='green', linestyle='-')

# Plotting the second line
plt.plot(range(10), runSimulation(bestSeed), label='AI step', color='red', linestyle='--')
plt.plot(range(10), runSimulation(result.x), label='Algo step', color='blue', linestyle=':')

# Adding labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plot with Two Lines')

# Adding legend
plt.legend()

# Displaying the plot
plt.grid(True)  # Add grid
plt.show()
