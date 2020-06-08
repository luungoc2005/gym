import gym
import json
import datetime as dt
import os

from stable_baselines.common import make_vec_env
from stable_baselines.sac.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines import PPO2, SAC

from gym.envs.zipline.zipline_env import ZiplineEnv

class CheckpointCallback(BaseCallback):

    def __init__(self, save_freq: int, save_path: str, name_prefix='rl_model', vec_norm=None, verbose=0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.vec_norm = vec_norm

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, '{}_{}_steps'.format(self.name_prefix, self.num_timesteps))
            vec_norm_path = os.path.join(self.save_path, '{}_{}_steps_norm'.format(self.name_prefix, self.num_timesteps))
            self.model.save(path)
            if self.vec_norm is not None:
                self.vec_norm.save(vec_norm_path)
            if self.verbose > 1:
                print("Saving model checkpoint to {}".format(path))
        return True

SAVE_PATH = "sac_zipline_sortino_lookback_21"
STATS_PATH = "sac_zipline_sortino_lookback_21_norm.pkl"

ENV_ARGS = {
    "start_date": "2018-12-1",
    "end_date": "2020-5-24",
    "lookback_window": 21,
}
N_ENVS = 1
DO_TRAIN = False

if __name__ == "__main__":
    if DO_TRAIN:
        env = DummyVecEnv([lambda: ZiplineEnv(**ENV_ARGS, max_steps=256)] * N_ENVS)
        # env = VecNormalize(env, norm_obs=True, norm_reward=True,
        #     clip_obs=10.)

        # model = PPO2(
        #     MlpLstmPolicy, 
        #     env,
        #     verbose=1, 
        #     n_steps=256,
        #     nminibatches=N_ENVS,
        #     tensorboard_log="./.tb_zipline_env/",
        # )

        # model = PPO2(
        #     MlpPolicy, 
        #     env,
        #     verbose=1, 
        #     n_steps=256,
        #     nminibatches=N_ENVS,
        #     tensorboard_log="./.tb_zipline_env/",
        # )
        model = SAC(
            LnMlpPolicy, 
            env,
            verbose=1,
            tensorboard_log="./.tb_zipline_env/",
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, 
            save_path='./.checkpoints/',
            name_prefix='sac_zipline_sortino',
        )

        model.learn(total_timesteps=1000000, callback=checkpoint_callback)
        model.save(SAVE_PATH)
        # env.save(STATS_PATH)

    # evaluate
    # model = PPO2.load(SAVE_PATH)
    model = SAC.load(SAVE_PATH)

    env = DummyVecEnv([lambda: ZiplineEnv(**ENV_ARGS, 
        max_steps=512, 
        do_record=True
    )] * N_ENVS)
    model.set_env(env)

    if os.path.isfile(STATS_PATH):
        env = VecNormalize.load(STATS_PATH, env)
        env.training = False
        env.norm_reward = False

    obs = env.reset()

    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if i == 256:
            print("--- Live trade starts at")
            print(rewards[0])
        # env.env_method("render", indices=[0])
        env.render()
        if done:
            exit()