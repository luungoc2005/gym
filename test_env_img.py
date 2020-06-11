import random
import numpy as np
from gym.envs.chrome_dino.chrome_dino_env import ChromeDinoEnv
from PIL import Image
import cv2

env = ChromeDinoEnv(
    chromedriver_path="/media/luungoc2005/Data/Projects/Samples/gym/chromedriver"
)

obs = env.reset()
for i in range(250):
    obs, rewards, dones, info = env.step(random.randint(0, 2))

Image.fromarray(
    np.reshape(obs, (80, 80)) * 255
).convert('RGB').save("test.jpg")