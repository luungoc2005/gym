import retro

import gym
import numpy as np
import os
import imageio

from tqdm import tqdm

from stable_baselines import A2C
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.callbacks import CheckpointCallback

if __name__ == '__main__':
    env_lambda = lambda: retro.make("F1-Genesis")
    do_train = True
    num_cpu = 4
    save_path = "f1_genesis_a2c_cnn"
    env = SubprocVecEnv([env_lambda for i in range(num_cpu)])

    if do_train:
        checkpoint_callback = CheckpointCallback(
            save_freq=int(1e6), 
            save_path='./.checkpoints/',
            name_prefix=save_path,
        )
        model = A2C(
            CnnPolicy,
            env,
            verbose=1,
            tensorboard_log="./.tb_f1_genesis/",
        )
        model.learn(
            total_timesteps=int(20e6), 
            callback=[checkpoint_callback]
        )
        model.save(save_path)
    
    model = A2C.load(save_path, env=env)

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