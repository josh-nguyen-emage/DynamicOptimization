import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from FlyObjectPlot import simulateFunction

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
counter = 0

def runSimulation(params):
    y_values = simulateFunction(np.clip(params,0,1))
    return np.array(y_values)

def objective_function(params):
    global expectChart
    global counter
    counter += 1
    if counter > 100:
        counter = 0
        print("---")
    generated_plot = runSimulation(params)
    # Calculate difference between generated plot and expected plot
    difference = np.sum((generated_plot - expectChart) ** 2)
    if difference < 2:
        difference = 0
    return difference


numberTestCase = 0

expectParams = np.random.rand(numberTestCase,10)
initialParams = np.random.rand(numberTestCase,10)

methodList = ['Nelder-Mead','SLSQP','Powell','CG','BFGS','L-BFGS-B','TNC','COBYLA','trust-constr']
# methodList = ['Nelder-Mead','BFGS','L-BFGS-B','TNC','COBYLA','SLSQP']

runtimeCollector = {}
for methodName in methodList:
    runtimeCollector[methodName] = 0

for methodCounter, methodName in enumerate(methodList):
    for testCaseCounter in range(numberTestCase):
        expectChart = runSimulation(expectParams[testCaseCounter])
        result = minimize(objective_function, initialParams[testCaseCounter], method=methodName,options={"maxiter":1000})
        runtimeCollector[methodName] += result.nfev
        print(methodName," : ",result.nfev)
    print(methodCounter,": Test method",methodName,"completed!")

names = list(runtimeCollector.keys())

runtimeCollector = {'Nelder-Mead': 3504, 'SLSQP': 1533, 'Powell': 5757, 'CG': 4931, 'BFGS': 5669, 'L-BFGS-B': 3036, 'TNC': 4422, 'COBYLA': 1212, 'trust-constr': 1815}
numberTestCase = 16
values = np.array(list(runtimeCollector.values()))/numberTestCase

print("Run total",sum(values)*numberTestCase,"times")
print(runtimeCollector)

# Vẽ đồ thị cột
methodName = ['Nelder-Mead 1965','SLSQP 1984','Powell 1964','CG 1957','BFGS 1970','L-BFGS-B 1997','TNC 1990','COBYLA 1994','trust-constr 2001']
plt.bar(methodName, values)

# Đặt tiêu đề cho đồ thị và các trục
plt.title('Biểu đồ So Sánh')
plt.xlabel('Phương Pháp')
plt.ylabel('Số lần lặp')

# Hiển thị đồ thị
plt.show()