import numpy as np
import pyautogui
import time
import cv2

def move_mouse_and_input_floats(float1, float2, float3):
    # Define the hardcoded positions to move the mouse to
    positions = [(1188,346), (1182, 442), (1258,503)]
    value = [float1, float2, float3]

    # Move mouse to each position and click left mouse button
    for i in range(3):
        pyautogui.moveTo(positions[i][0], positions[i][1], duration=0.5)
        pyautogui.click(button='left')
        pyautogui.typewrite(str(value[i]), interval=0.1)

        # Add a delay to ensure each action is completed before moving to the next position
        time.sleep(0.5)

def find_blue_balls():
    # Capture the screen
    screen = np.array(pyautogui.screenshot())

    roi = screen[300:1000, 60:840, 2]

    _, mask = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blue_balls_coords = []
    # Iterate through contours and find blue balls with size > 100
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            # Calculate centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Append centroid coordinates to the list
                blue_balls_coords.append((cX, cY))

    return blue_balls_coords

time.sleep(3)
find_blue_balls()

# move_mouse_and_input_floats(1.23, 4.56, 7.89)
