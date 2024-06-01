import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory(angle_deg, power, air_resistance):
    angle_deg *= 50 + 30
    power *= 10 + 10
    air_resistance *= 0.1
    air_resistance = np.clip(air_resistance,0,0.9)
    # Constants
    g = 9.81  # Acceleration due to gravity (m/s^2)
    dt = 0.01  # Time step (s)
    
    # Convert angle from degrees to radians
    angle_rad = np.deg2rad(angle_deg)
    
    # Initial velocity components
    vx = power * np.cos(angle_rad)  # Horizontal velocity component (m/s)
    vy = power * np.sin(angle_rad)  # Vertical velocity component (m/s)
    
    # Initialize lists to store trajectory coordinates
    x_traj = [0]  # x-coordinate of trajectory
    y_traj = [0]  # y-coordinate of trajectory

    returnVal = []
    
    # Simulation loop
    while True:
        # Calculate air resistance
        air_resistance_x = air_resistance * vx**2
        air_resistance_y = air_resistance * vy**2
        air_resistance_x = np.clip(air_resistance_x,-10,10)
        air_resistance_y = np.clip(air_resistance_y,-10,10)
        
        # Update position using projectile motion equations with air resistance
        x_next = x_traj[-1] + vx * dt
        y_next = y_traj[-1] + vy * dt - 0.5 * g * dt**2
        
        # Update velocities with air resistance
        vx = vx - air_resistance_x * dt  # Horizontal velocity decreases due to air resistance
        vy = vy - g * dt - air_resistance_y * dt  # Vertical velocity decreases due to gravity and air resistance
        
        # Check if the object hits the ground (y-coordinate becomes negative)
        if y_next < 0:
            break
        
        # Append new coordinates to trajectory lists
        x_traj.append(x_next)
        y_traj.append(y_next)

        if len(returnVal) < x_next:
            returnVal.append(y_next)

    if len(returnVal) > 10:
        return returnVal[:10]
    else:
        while len(returnVal) < 10:
            returnVal.append(0)
        return returnVal

def simulateFunction(x):
    air_resistance = np.sum(x[2:])/8
    return plot_trajectory(angle_deg=x[0], power=x[1], air_resistance=air_resistance)
    

# for _ in range(10):
#     outputPlot = plot_trajectory(angle_deg=np.random.rand(), power=np.random.rand(), air_resistance=np.random.rand())
#     plt.plot(range(10), outputPlot)
#     outputPlot = plot_trajectory(angle_deg=np.random.rand(), power=np.random.rand(), air_resistance=np.random.rand())
#     plt.plot(range(10), outputPlot)
#     outputPlot = plot_trajectory(angle_deg=np.random.rand(), power=np.random.rand(), air_resistance=np.random.rand())
#     plt.plot(range(10), outputPlot)
#     outputPlot = plot_trajectory(angle_deg=np.random.rand(), power=np.random.rand(), air_resistance=np.random.rand())
#     plt.plot(range(10), outputPlot)

#     plt.xlabel('Horizontal Distance (m)')
#     plt.ylabel('Vertical Distance (m)')
#     plt.title('Trajectory of Object with Air Resistance')
#     plt.grid(True)
    
#     # Set x and y limits to fix the width and height
#     plt.xlim(0, 10)
#     plt.ylim(0, 10)
    
#     plt.show()
