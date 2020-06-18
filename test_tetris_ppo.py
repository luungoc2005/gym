import retro

import gym
import numpy as np
import os
import imageio

from tqdm import tqdm

from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.callbacks import CheckpointCallback

if __name__ == '__main__':
    env_lambda = lambda: retro.make("TetrisAttack-Snes")
    do_train = True
    num_cpu = 4
    save_path = "tetris_attack_ppo_cnn"
    env = SubprocVecEnv([env_lambda for i in range(num_cpu)])

    if do_train:
        checkpoint_callback = CheckpointCallback(
            save_freq=int(10e6), 
            save_path='./.checkpoints/',
            name_prefix=save_path,
        )
        model = PPO2(
            CnnPolicy,
            env,
            verbose=1,
            tensorboard_log="./.tb_tetris_attack/",
        )
        model.learn(
            total_timesteps=int(100e6), 
            callback=[checkpoint_callback]
        )
        model.save(save_path)
    
    model = PPO2.load(save_path, env=env)

    # images = []

    obs = env.reset()
    # img = model.env.render(mode='rgb_array')

    while True:
        # images.append(img)
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render(mode='human')

        if any(dones):
            break
        # env.env_method("render", indices=[0])
        # img = env.render(mode='rgb_array')

    # imageio.mimsave('f1.gif', [np.array(img) for i, img in enumerate(images[:-500])], fps=15)

    exit()