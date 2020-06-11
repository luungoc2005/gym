import numpy as np
import os
import gym
from gym import error, spaces

from io import BytesIO
from PIL import Image
import base64
import cv2

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time

class ChromeDinoEnv(gym.Env):

    def __init__(self,
            screen_width: int=80,
            screen_height: int=80,
            chromedriver_path: str="chromedriver"
        ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.chromedriver_path = chromedriver_path

        self.action_space = spaces.Discrete(3) # do nothing, up, down
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(self.screen_width, self.screen_height, 1), 
            dtype=np.uint8
        )

        _chrome_options = webdriver.ChromeOptions()
        _chrome_options.add_argument("--mute-audio")

        self._driver = webdriver.Chrome(
            executable_path=self.chromedriver_path,
            chrome_options=_chrome_options
        )

    def reset(self):
        self._driver.get('chrome://dino')
        WebDriverWait(self._driver, 10).until(
            EC.presence_of_element_located((
                By.CLASS_NAME, 
                "runner-canvas"
            ))
        )

        # trigger game start
        self._driver.find_element_by_tag_name("body").send_keys(Keys.SPACE)

        return self._next_observation()

    def _get_image(self):
        LEADING_TEXT = "data:image/png;base64,"
        _img = self._driver.execute_script(
            "return document.querySelector('canvas.runner-canvas').toDataURL()"
        )
        _img = _img[len(LEADING_TEXT):]
        return np.array(
            Image.open(BytesIO(base64.b64decode(_img)))
        )

    def _next_observation(self):
        image = cv2.cvtColor(self._get_image(), cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.screen_width, self.screen_height))
        image = np.reshape(image, (self.screen_width, self.screen_height, 1))
        # Thresholding
        # image[image > .5] = 1
        # image[image < .5] = 0

        return image

    def _get_reward(self):
        score_str = ''.join(
            self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        )
        return int(score_str)

    def _get_done(self):
        return not self._driver.execute_script("return Runner.instance_.playing")

    def step(self, action):
        # perform action
        if action == 0: # do nothing
            pass
        elif action == 1: # jump
            self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
        else: # duck
            self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)
        
        obs = self._next_observation()
        reward = self._get_reward()
        done = self._get_done()

        time.sleep(.18)

        return obs, reward, done, {}

    def render(self, mode='human'):
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
