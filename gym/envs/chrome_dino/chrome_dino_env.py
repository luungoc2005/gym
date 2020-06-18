import numpy as np
import os
import gym
from gym import error, spaces
from collections import deque

from io import BytesIO
from PIL import Image
import base64
import cv2

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time

class ChromeDinoEnv(gym.Env):

    def __init__(self,
            screen_width: int=120,
            screen_height: int=120,
            chromedriver_path: str="chromedriver"
        ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.chromedriver_path = chromedriver_path

        self.action_space = spaces.Discrete(3) # do nothing, up, down
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(self.screen_width, self.screen_height, 4), 
            dtype=np.uint8
        )

        _chrome_options = webdriver.ChromeOptions()
        _chrome_options.add_argument("--mute-audio")
        # _chrome_options.add_argument("--disable-gpu")
        # _chrome_options.add_argument("--headless")

        self._driver = webdriver.Chrome(
            executable_path=self.chromedriver_path,
            chrome_options=_chrome_options
        )
        self.current_key = None
        self.state_queue = deque(maxlen=4)

        self.actions_map = [
            Keys.ARROW_RIGHT, # do nothing
            Keys.ARROW_UP, # jump
            Keys.ARROW_DOWN # duck
        ]
        action_chains = ActionChains(self._driver)
        self.keydown_actions = [action_chains.key_down(item) for item in self.actions_map]
        self.keyup_actions = [action_chains.key_up(item) for item in self.actions_map]

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
        image = image[:500, :480] # cropping
        image = cv2.resize(image, (self.screen_width, self.screen_height))

        # Thresholding
        # image[image > 127] = 255
        # image[image < 127] = 0
        self.state_queue.append(image)

        if len(self.state_queue) < 4:
            return np.stack([image] * 4, axis=-1)
        else:
            return np.stack(self.state_queue, axis=-1)

        return image

    def _get_score(self):
        score_str = ''.join(
            self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        )
        return int(score_str)

    def _get_done(self):
        return not self._driver.execute_script("return Runner.instance_.playing")

    def step(self, action: int):
        # perform action
        # if self.current_key != action:
        #     if self.current_key is not None:
        #             self.keyup_actions[self.current_key].perform()

        #     self.current_key = action

        #     self.keydown_actions[self.current_key].perform()
        self._driver.find_element_by_tag_name("body") \
            .send_keys(self.actions_map[action])

        obs = self._next_observation()

        done = self._get_done()
        reward = .01 if not done else -1

        # self._driver.implicitly_wait(.4)
        time.sleep(.015)

        return obs, reward, done, {"score": self._get_score()}

    def render(self, mode: str='human'):
        img = cv2.cvtColor(self._get_image(), cv2.COLOR_BGR2RGB)
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
