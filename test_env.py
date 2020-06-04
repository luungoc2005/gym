import gym
import json
import datetime as dt

from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from gym.envs.zipline.zipline_env import ZiplineEnv

env = DummyVecEnv([lambda: ZiplineEnv()])

model = PPO2(
    MlpLstmPolicy, 
    env,
    verbose=1, 
    n_steps=256, 
    nminibatches=1,
    tensorboard_log="./.tb_zipline_env/",
)
model.learn(total_timesteps=200000)

obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()