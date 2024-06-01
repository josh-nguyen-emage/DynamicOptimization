import sys
import numpy as np
import matplotlib.pyplot as plt

value = np.random.rand(10)
print(list(value))
value = value-0.5

# Define the function
def f(x):
    return ((value[0]*x+value[1])*(value[2]*x+value[3]))/((value[4]*x+value[5])*(value[6]*x+value[7])) + value[8]*x*x+value[9]*x

# Generate x values
x_values = np.linspace(-10, 10, 10)


# Calculate y values
y_values = f(x_values)

y_values = np.clip(y_values,-10,10)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label=r'$y$')
plt.title('Plot$')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
