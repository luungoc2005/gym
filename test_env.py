import gym
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.a2c import CnnPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from gym.envs.chrome_dino.chrome_dino_env import ChromeDinoEnv

if __name__ == '__main__':
    env_lambda = lambda: Monitor(ChromeDinoEnv(
        chromedriver_path="/media/luungoc2005/Data/Projects/Samples/gym/chromedriver"
    ))
    do_train = True
    num_cpu = 4
    save_path = "chrome_dino_a2c_cnn"
    env = SubprocVecEnv([env_lambda for i in range(num_cpu)])

    if do_train:
        checkpoint_callback = CheckpointCallback(
            save_freq=5000, 
            save_path='./.checkpoints/',
            name_prefix=save_path,
        )
        model = A2C(
            CnnPolicy,
            env,
            verbose=1,
            tensorboard_log="./.tb_chromedino_env/",
        )
        model.learn(total_timesteps=500000, callback=checkpoint_callback)
        model.save(save_path)
    else:
        model = A2C.load(save_path)
        model.set_env(env)

        obs = env.reset()
        for _ in range(1000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            # env.env_method("render", indices=[0])
            env.render()