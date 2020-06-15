import gym
import numpy as np
import os

from stable_baselines import DQN
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import CheckpointCallback

from gym.envs.chrome_dino.chrome_dino_env import ChromeDinoEnv

if __name__ == '__main__':
    env_lambda = lambda: ChromeDinoEnv(
        screen_width=96,
        screen_height=96,
        chromedriver_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "chromedriver"
        )
    )
    do_train = True
    save_path = "chrome_dino_dqn_cnn"
    env = DummyVecEnv([env_lambda])

    if do_train:
        checkpoint_callback = CheckpointCallback(
            save_freq=100000, 
            save_path='./.checkpoints/',
            name_prefix=save_path,
        )
        model = DQN(
            CnnPolicy,
            env,
            verbose=1,
            tensorboard_log="./.tb_chromedino_env/",
        )
        model.learn(total_timesteps=2000000, callback=checkpoint_callback)
        model.save(save_path)

    model = DQN.load(save_path, env=env)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        # env.env_method("render", indices=[0])
        env.render(mode="human")