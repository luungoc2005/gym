import random
import numpy as np
import os

from gym.envs.chrome_dino.chrome_dino_env import ChromeDinoEnv
from PIL import Image
import cv2

env = ChromeDinoEnv(
    chromedriver_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "chromedriver"
    )
)

obs = env.reset()
for i in range(250):
    obs, rewards, dones, info = env.step(random.randint(0, 1))

Image.fromarray(
    np.reshape(obs[:,:,-1], (120, 120)) * 255
).convert('RGB').save("test.jpg")