import gym
import numpy as np
import os
import imageio

from tqdm import tqdm

from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
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
    do_train = False
    num_cpu = 4
    save_path = "chrome_dino_ppo_cnn"
    env = SubprocVecEnv([env_lambda for i in range(num_cpu)])

    if do_train:
        checkpoint_callback = CheckpointCallback(
            save_freq=200000, 
            save_path='./.checkpoints/',
            name_prefix=save_path,
        )
        model = PPO2(
            CnnPolicy,
            env,
            verbose=1,
            tensorboard_log="./.tb_chromedino_env/",
        )
        model.learn(
            total_timesteps=2000000, 
            callback=[checkpoint_callback]
        )
        model.save(save_path)
    
    model = PPO2.load(save_path, env=env)

    images = []

    obs = env.reset()
    img = model.env.render(mode='rgb_array')

    for i in tqdm(range(1000)):
        images.append(img)
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        # env.env_method("render", indices=[0])
        img = env.render(mode='rgb_array')

    imageio.mimsave('dino.gif', [np.array(img) for i, img in enumerate(images[:-500])], fps=15)

    exit()