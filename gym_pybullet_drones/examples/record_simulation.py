"""Script to record a video of a trained model in a custom environment.

Example
-------
In a terminal, run as:

    $ python record_simulation.py --model_path <path/to/your/model.zip> --env_file <your_env_file.py> --env_class <YourEnvClass>

"""
import os
import time
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import importlib.util

# Import your custom environment
# from <your_env_file> import <YourEnvClass>

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('one_d_rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'

def run(model_path, env_file, env_class, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO):

    # Dynamically import the custom environment
    spec = importlib.util.spec_from_file_location(env_class, env_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    CustomAviary = getattr(module, env_class)

    #### Load the trained model ########################################
    if os.path.isfile(model_path):
        model = PPO.load(model_path)
    else:
        print(f"[ERROR]: no model found at the specified path {model_path}")
        return

    #### Show (and record a video of) the model's performance ##
    test_env = CustomAviary(gui=gui,
                           obs=DEFAULT_OBS,
                           act=DEFAULT_ACT,
                           record=record_video)

    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=1, # Assuming a single drone for simplicity
                output_folder=output_folder,
                colab=colab
                )

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            logger.log(drone=0,
                timestamp=i/test_env.CTRL_FREQ,
                state=np.hstack([obs2[0:3],
                                    np.zeros(4),
                                    obs2[3:15],
                                    act2
                                    ]),
                control=np.zeros(12)
                )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Script to record a video of a trained model.')
    parser.add_argument('--model_path',       required=True, type=str,           help='Path to the trained model file (e.g., results/save-08.05.2025_20.19.22/best_model.zip)')
    parser.add_argument('--env_file',         required=True, type=str,           help='Python file containing your custom environment (e.g., my_custom_env.py)')
    parser.add_argument('--env_class',        required=True, type=str,           help='Class name of your custom environment (e.g., MyCustomAviary)')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: True)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
