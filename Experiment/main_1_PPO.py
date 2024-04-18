from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import os
import time
import random
from randomCode import DTW_simulator

#                            _
#                         _ooOoo_
#                        o8888888o
#                        88" . "88
#                        (| -_- |)
#                        O\  =  /O
#                     ____/`---'\____
#                   .'  \\|     |//  `.
#                  /  \\|||  :  |||//  \
#                 /  _||||| -:- |||||_  \
#                 |   | \\\  -  /'| |   |
#                 | \_|  `\`---'//  |_/ |
#                 \  .-\__ `-. -'__/-.  /
#               ___`. .'  /--.--\  `. .'___
#            ."" '<  `.___\_<|>_/___.' _> \"".
#           | | :  `- \`. ;`. _/; .'/ /  .' ; |
#           \  \ `-.   \_\_`. _.'_/_/  -' _.' /
# ===========`-.`___`-.__\ \___  /__.-'_.'_.-'================
#                         `=--=-'                    

if __name__ == '__main__':  

    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    env = DTW_simulator()
    env.reset()

    model = RecurrentPPO("MlpLstmPolicy", env, n_epochs = 16)
    
    i = 0
    while True:
        i += 1
        model.learn(total_timesteps=16)
        # model.save(f"{GAME_SETTING.MODEL_DIR}/{TIMESTEPS*i}")
        print(" ================ Epoch",i,"=======================")
        

    env.close()