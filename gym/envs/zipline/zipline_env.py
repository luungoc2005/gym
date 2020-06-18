import numpy as np
import pandas as pd
import os
import gym

from gym import error, spaces
from gym import utils
from gym.utils import seeding

import subprocess
import redis
import json
import uuid

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
        do_normalize=True,
        do_record=False,
        communication_mode="redis",
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
        self.do_normalize = do_normalize
        self.do_record = do_record
        self.env_id = uuid.uuid4().hex
        self.communication_mode = communication_mode

        assert communication_mode in ["redis", "pipe"]

        self.rewards_history = []
        self.actions_history = []
        self.obs_history = []

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
        self.fig = None


    def reset(self):
        if self.p_zipline is not None:
            self.p_zipline.terminate()

        env_dict = os.environ
        env_dict["ENV_ID"] = self.env_id

        if self.communication_mode == "redis":
            env_dict["COMM_MODE"] = "redis"
        else:
            env_dict["COMM_MODE"] = "pipe"

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
            ],
            env=env_dict,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1
        )

        finished = False
        while not finished:
            for data in self._get_message():
                if data["env_id"] == self.env_id and data["message"] == "algo_init":
                    self._send_message({
                        "env_id": self.env_id,
                        "message": "set_options",
                        "data": {
                            "tickers": self.tickers,
                            "lookback_window": self.lookback_window,
                            "start_date": self.start_date,
                            "end_date": self.end_date,
                            "do_normalize": self.do_normalize,
                            "do_record": self.do_record,
                        },
                    })
                    finished = True
                    break

        self.current_step = 0
        self.positions_value = 0
        self.portfolio_value = 0
        self.profit = 0
        self.actions_history = []
        self.rewards_history = []
        self.obs_history = []
        self.current_date = ""

        return self._next_observation()

    def _send_message(self, data: object):
        if self.communication_mode == "redis":
            redis_host.publish(PUBSUB_CHANNEL, json.dumps(data))
        else:
            self.p_zipline.stdin.write((json.dumps(data) + '\n').encode('utf-8'))
            self.p_zipline.stdin.flush()

    def _get_message(self):
        while True:
            message = pubsub.get_message() if self.communication_mode == "redis" \
                else self.p_zipline.stdout.readline()

            try:
                if self.communication_mode == "redis":
                    data = json.loads(message["data"].decode('utf-8'))
                else:
                    data = json.loads(message.decode('utf-8'))
                yield data
            except GeneratorExit:
                return
            except:
                pass

    def _next_observation(self):
        self._send_message({
            "env_id": self.env_id,
            "message": "get_state",
        })
        while True:
            for data in self._get_message():
                if data["env_id"] == self.env_id and data["message"] == "algo_state":
                    return data["data"]

    def _get_reward(self):
        self._send_message({
            "env_id": self.env_id,
            "message": "get_reward",
        })
        while True:
            for data in self._get_message():
                if data["env_id"] == self.env_id and data["message"] == "algo_reward":
                    return data["data"]


    def step(self, action):
        self._send_message({
            "env_id": self.env_id,
            "message": "set_action",
            "data": action.tolist()
        })

        reward_raw = self._get_reward()
        obs = self._next_observation()

        reward = reward_raw["reward"]
        self.current_date = reward_raw["date"]
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
        self.rewards_history.append(reward_raw)
        self.obs_history.append(obs)

        return obs, reward, self.current_step == self.max_steps, {}

    def render(self, mode='text', window_size=40):
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        from mpl_finance import candlestick_ochl

        import math

        if self.fig is None:
            self.fig = plt.figure()

            plt_rows = math.ceil(len(self.tickers) / 2) + 3
            self.positions_value_ax = plt.subplot2grid(
                (plt_rows, 2), (0, 0), 
                rowspan=2, colspan=2
            )
            self.positions_value_ax.xaxis.set_visible(False)

            self.tickers_axes = []
            for ticker_ix, ticker in enumerate(self.tickers):
                ticker_x = ticker_ix % 2
                ticker_y = ticker_ix // 2 + 2
                ticker_ax = plt.subplot2grid(
                    (plt_rows, 2), (ticker_y, ticker_x), 
                    rowspan=1, colspan=1, 
                    sharex=self.positions_value_ax
                )
                ticker_ax.text(0.5, 0.5, ticker, va="center", ha="center")

                ticker_ax.yaxis.set_visible(False)
                ticker_ax.xaxis.set_visible(False)

                self.tickers_axes.append(ticker_ax)

            plt.show(block=False)
        
        raw_rewards = self.rewards_history[-window_size:]
        raw_rewards = [item for item in raw_rewards if item["date"] != ""]

        if len(raw_rewards) > 0:
            dates = [pd.to_datetime(item["date"]) for item in raw_rewards]

            # render positions value
            self.positions_value_ax.clear()
            self.positions_value_ax.plot_date(
                dates,
                [item["portfolio_value"] for item in raw_rewards],
                '-',
                label="Portfolio Value"
            )

            if not pd.isnull(dates[-1]):
                self.positions_value_ax.annotate(
                    '{0:.2f}'.format(self.portfolio_value),     
                    (dates[-1], self.portfolio_value),
                    xytext=(dates[-1], self.portfolio_value),
                    bbox=dict(boxstyle='round', fc='w', ec='k', lw=1),
                    color="black",
                    fontsize="small"
                )

            plt.setp(self.positions_value_ax.get_xticklabels(), visible=False)

            # ticker prices

            UP_COLOR = '#27A59A'
            DOWN_COLOR = '#EF534F'
            UP_TEXT_COLOR = '#73D3CC'
            DOWN_TEXT_COLOR = '#DC2C27'

            dates_ts = mdates.date2num([item.to_pydatetime() for item in dates])

            for ticker_ix, ticker in enumerate(self.tickers):
                ticker_ax = self.tickers_axes[ticker_ix]
                ticker_ax.clear()

                # chronologically ascending
                values_history = [
                    item[ticker_ix][0]
                    for item in self.obs_history[-len(dates):]
                ]

                # ['close', 'high', 'low', 'open'] - alphabetical
                candlesticks = zip(
                    dates_ts,
                    [item[3] for item in values_history],
                    [item[0] for item in values_history],
                    [item[1] for item in values_history],
                    [item[2] for item in values_history],
                )

                candlestick_ochl(
                    ticker_ax, candlesticks, 
                    width=0.8,
                    colorup=UP_COLOR, 
                    colordown=DOWN_COLOR
                )

                last_state = self.obs_history[-1][ticker_ix][0]
                last_high = last_state[1]
                last_close = last_state[0]
                # print(dates[-1])
                # print(ticker)
                # print(last_state)

                ticker_ax.annotate(
                    '{0:.2f}'.format(last_close),
                    (dates[-1], last_close),
                    xytext=(dates[-1], last_high),
                    bbox=dict(boxstyle='round', fc='w', ec='k', lw=1),
                    color="black",
                    fontsize="small"
                )

                # actions
                TRADE_THRESHOLD = .03 # 3% of portfolio value
                trades = [item[ticker_ix] for item in self.actions_history[-window_size:-1]]
                filtered_trades = []

                for trade_ix, trade_value in enumerate(trades):
                    if trade_ix == 0:
                        filtered_trades.append((dates[trade_ix], trade_value))
                    else:
                        if abs(trade_value - trades[trade_ix - 1]) > TRADE_THRESHOLD:
                            filtered_trades.append((dates[trade_ix], trade_value - trades[trade_ix - 1]))

                colors = [
                    UP_TEXT_COLOR if item[1] >= 0 else DOWN_TEXT_COLOR
                    for item in filtered_trades
                ]
                for ix, item in enumerate(filtered_trades):
                    ticker_ax.annotate(
                        '{0:.2f}'.format(item[1]), 
                        item, # date, price
                        xytext=item,
                        color=colors[ix],
                        fontsize=8,
                        arrowprops={"color": colors[ix]}
                    )

        plt.pause(0.001)
        
        print('Step: {}'.format(self.current_step))
        print('Date: {}'.format(self.current_date))
        print('Positions value: {}'.format(self.positions_value))
        print('Portfolio value: {}'.format(self.portfolio_value))
        print('Profit: {}'.format(self.profit))
        print('---')
