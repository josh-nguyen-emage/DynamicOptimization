import math
import cv2
import gym
import numpy as np
from gym import spaces

def abs_sub_images(image1, image2, pos1, pos2):
    # Đọc ảnh
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    # Tính toán ma trận chuyển đổi affine cho từng ảnh
    M1 = cv2.getRotationMatrix2D((pos1['x'], pos1['y']), pos1['angle'], pos1['zoom'])
    M2 = cv2.getRotationMatrix2D((pos2['x'], pos2['y']), pos2['angle'], pos2['zoom'])

    # Áp dụng phép biến đổi affine
    img1_transformed = cv2.warpAffine(img1, M1, (img1.shape[1], img1.shape[0]))
    img2_transformed = cv2.warpAffine(img2, M2, (img2.shape[1], img2.shape[0]))

    # Tính toán giá trị tuyệt đối khác nhau
    diff = cv2.absdiff(img1_transformed, img2_transformed)

    # Tính tổng giá trị pixel của ảnh kết quả
    total_pixels = np.sum(diff)

    return total_pixels

def data2pos(data):
    data = np.clip(data,0,1)
    pos1 = {'x': data[0]*1000, 'y': data[1]*1000, 'angle': data[2]*90, 'zoom': data[3]*2}
    pos2 = {'x': data[4]*1000, 'y': data[5]*1000, 'angle': data[6]*90, 'zoom': data[7]*2}

    return pos1, pos2


class DTW_simulator(gym.Env):
    def __init__(self) -> None:
        super(DTW_simulator, self).__init__()
        self.action_space = spaces.Discrete(8, )

        lowerBoundLidar = np.full((8,), 0, dtype="float32")
        upperBoundLidar = np.full((8,), 1, dtype="float32")     

        self.observation_space = spaces.Box(
            low=lowerBoundLidar,
            high=upperBoundLidar
        )

        self.action_space = spaces.Box(
            low=lowerBoundLidar,
            high=upperBoundLidar
        )

        self.runCounter = 0


    def reset(self):
        returnSpace = []
        self.runCounter = 0
        for _ in range(8):
            returnSpace.append(np.random.rand())
        return returnSpace

    def step(self, action):        
        self.runCounter += 1
        pos1, pos2 = data2pos(action)
        image1_path = '1.jpg'
        image2_path = '2.jpg'
        total_diff_pixels = abs_sub_images(image1_path, image2_path, pos1, pos2)
        reward = 10 - math.log10(total_diff_pixels)

        print("Reward: ",reward)

        completed = self.runCounter > 16
            
        return action, reward, completed, {}

    def render(self, mode="human", close=False):
        pass