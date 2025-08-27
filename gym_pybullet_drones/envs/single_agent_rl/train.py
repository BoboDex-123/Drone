
import time
import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from gym_pybullet_drones.envs.single_agent_rl.DynamicAviary import DynamicAviary
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):

    filename = os.path.join(output_folder, 'save-flight.zip')

    if not os.path.exists(filename):
        # Note: training is done without the GUI and without recording
        train_env = make_vec_env(DynamicAviary, n_envs=1, env_kwargs=dict(gui=False, record=False))
        model = PPO('MlpPolicy', train_env, verbose=1)
        model.learn(total_timesteps=10000) # Lowered for faster training
        model.save(filename)

    if os.path.exists(filename):
        # Run the evaluation with the GUI and record the video
        eval_env = DynamicAviary(gui=gui, record=record_video)
        model = PPO.load(filename)

        obs, info = eval_env.reset(seed=42)
        start = time.time()
        for i in range(3 * eval_env.PYB_FREQ):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            eval_env.render()
            sync(i, start, eval_env.CTRL_TIMESTEP)
            if terminated:
                obs, info = eval_env.reset(seed=42)
        eval_env.close()

if __name__ == '__main__':
    run(record_video=True, gui=True)
