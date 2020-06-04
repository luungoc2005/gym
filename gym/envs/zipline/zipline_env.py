import numpy as np
import os
import gym

from gym import error, spaces
from gym import utils
from gym.utils import seeding

import subprocess
import redis
import json

ZIPLINE_PYTHON_PATH = "/Users/luungoc2005/miniconda3/envs/trading-bot-base/bin/zipline"
ALGO_PATH = "/Users/luungoc2005/Documents/Samples/pipeline-live/zipline_algos/gym_algo.py"

PUBSUB_CHANNEL = "zipline-env"
redis_host = redis.Redis(host='localhost', port=6379, db=0)
pubsub = redis_host.pubsub()
pubsub.subscribe(PUBSUB_CHANNEL)

class ZiplineEnv(gym.Env):

    def __init__(
        self,
        tickers=['U11', 'BS6', 'A17U', 'G13', 'O39', 'Z74', 'C61U', 'C31', 'C38U', 'D05'],
        data_frequency="daily",
        capital_base="1000",
        trading_calendar="XSES",
        bundle_name="sgx_stocks",
        start_date="2018-12-1",
        end_date="2020-5-24",
        env_output="env_result.pickle",
        lookback_window=30,
        max_steps=256,
    ):
        self.tickers = tickers
        self.data_frequency = data_frequency
        self.capital_base = capital_base
        self.trading_calendar = trading_calendar
        self.bundle_name = bundle_name
        self.start_date = start_date
        self.end_date = end_date
        self.env_output = env_output
        self.lookback_window = lookback_window
        self.max_steps = max_steps

        self.action_space = spaces.Box(
            low=0, 
            high=1,
            shape=(len(self.tickers) + 1,),
            dtype=np.float16,
        )
        self.observation_space = spaces.Box(
            low=0, high=1000, # how to do high value??? 
            # (OHLC + current portfolio ratio) * days * (positions + cash)
            shape=(len(self.tickers) + 1, self.lookback_window, 5),
            dtype=np.float16
        )
        self.p_zipline = None


    def reset(self):
        if self.p_zipline is not None:
            self.p_zipline.terminate()

        self.p_zipline = subprocess.Popen([
            ZIPLINE_PYTHON_PATH, "run",
            "-f", ALGO_PATH,
            "--data-frequency", self.data_frequency,
            "--capital-base", self.capital_base,
            "--trading-calendar", self.trading_calendar,
            "-b", self.bundle_name,
            "-s", self.start_date,
            "-e", self.end_date,
            "-o", self.env_output,
        ])

        finished = False
        while not finished:
            for message in pubsub.listen():
                try:
                    # print(message)
                    data = json.loads(message["data"].decode('utf-8'))
                    if data["message"] == "algo_init":
                        redis_host.publish(PUBSUB_CHANNEL, json.dumps({
                            "message": "set_options",
                            "data": {
                                "tickers": self.tickers,
                                "lookback_window": self.lookback_window,
                            },
                        }))
                        finished = True
                        break
                except:
                    pass

        self.current_step = 0
        self.positions_value = 0
        self.portfolio_value = 0
        self.profit = 0
        self.actions_history = []

        return self._next_observation()


    def _next_observation(self):
        redis_host.publish(PUBSUB_CHANNEL, json.dumps({
            "message": "get_state",
        }))
        while True:
            for message in pubsub.listen():
                try:
                    # print(message)
                    data = json.loads(message["data"].decode('utf-8'))
                    if data["message"] == "algo_state":
                        return data["data"]
                except:
                    pass

    def _get_reward(self):
        redis_host.publish(PUBSUB_CHANNEL, json.dumps({
            "message": "get_reward",
        }))
        while True:
            for message in pubsub.listen():
                try:
                    # print(message)
                    data = json.loads(message["data"].decode('utf-8'))
                    if data["message"] == "algo_reward":
                        return data["data"]
                except:
                    pass


    def step(self, action):
        redis_host.publish(PUBSUB_CHANNEL, json.dumps({
            "message": "set_action",
            "data": action.tolist()
        }))

        reward_raw = self._get_reward()
        obs = self._next_observation()

        reward = reward_raw["reward"]
        self.positions_value = reward_raw["positions_value"]
        self.portfolio_value = reward_raw["portfolio_value"]
        self.profit = reward_raw["profit"]

        # done = False
        # if self.p_zipline is not None:
        #     try:
        #         self.p_zipline.wait(timeout=300)
        #     except subprocess.TimeoutExpired:
        #         if self.p_zipline.returncode is None:
        #             done = True

        self.current_step += 1
        self.actions_history.append(action)

        return obs, reward, self.current_step == self.max_steps, {}

    def render(self, mode='human', close=False):
        print('Step: {}'.format(self.current_step))
        print('Positions value: {}'.format(self.positions_value))
        print('Portfolio value: {}'.format(self.portfolio_value))
        print('Profit: {}'.format(self.profit))
        print('---')
