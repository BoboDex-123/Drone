import time
import argparse
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.MazeAviary import MazeAviary
import os

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 120

def run(gui=DEFAULT_GUI, record_video=DEFAULT_RECORD_VIDEO, simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ, control_freq_hz=DEFAULT_CONTROL_FREQ_HZ, duration_sec=DEFAULT_DURATION_SEC, local=True):

    filename = os.path.join(DEFAULT_OUTPUT_FOLDER, 'save-maze-ppo')
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    train_env = make_vec_env(MazeAviary,
                             env_kwargs=dict(gui=False, record=False, ctrl_freq=control_freq_hz),
                             n_envs=1,
                             seed=0
                             )

    eval_env = MazeAviary(gui=gui, record=record_video, ctrl_freq=control_freq_hz)

    #### Train the model #######################################
    model = PPO('CnnPolicy',
                train_env,
                verbose=1)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=np.inf,
                                                     verbose=1)
    
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)

    model.learn(total_timesteps=int(1e3) if local else int(1e2), # shorter training in GitHub Actions pytest
                callback=eval_callback,
                log_interval=100)

    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################

    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

    #### Show (and record a video of) the model's performance ##
    test_env = MazeAviary(gui=gui, record=record_video)
    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range(0, int(duration_sec * test_env.CTRL_FREQ)):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        print(f"Detected Allies: {info['detected_allies']}, Detected Enemies: {info['detected_enemies']}", end='\r')
        test_env.render()
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs, info = test_env.reset(seed=42, options={})
    test_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Maze Aviary')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int, help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int, help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int, help='Duration of the simulation in seconds (default: 120)', metavar='')
    ARGS = parser.parse_args()
    run(**vars(ARGS))