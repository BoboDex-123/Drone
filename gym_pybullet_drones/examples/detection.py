import time
import argparse
import numpy as np

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs import DetectionAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 60
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('pid')

def run(gui=DEFAULT_GUI, record_video=DEFAULT_RECORD_VIDEO, simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ, control_freq_hz=DEFAULT_CONTROL_FREQ_HZ, duration_sec=DEFAULT_DURATION_SEC, obs=DEFAULT_OBS, act=DEFAULT_ACT):
    env = DetectionAviary(gui=gui, record=record_video, obs=obs, act=act)
    obs, info = env.reset(seed=42, options={})
    print(f"Total Friends: {env.NUM_FRIENDS}, Total Enemies: {env.NUM_ENEMIES}")

    start = time.time()
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        # Fly in a circle
        radius = 3
        angle = (i / (duration_sec * env.CTRL_FREQ)) * 2 * np.pi * 2 # 2 circles
        target_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), 1.0])

        action = np.array([target_pos])
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Detected Friends: {info['detected_friends']}, Detected Enemies: {info['detected_enemies']}", end='\r')

        env.render()
        sync(i, start, env.CTRL_TIMESTEP)
        if terminated:
            obs, info = env.reset(seed=42, options={})
            print(f"\nTotal Friends: {env.NUM_FRIENDS}, Total Enemies: {env.NUM_ENEMIES}")

    print(f"\nFinal Detected Friends: {info['detected_friends']}, Final Detected Enemies: {info['detected_enemies']}")
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detection Aviary')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int, help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int, help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int, help='Duration of the simulation in seconds (default: 60)', metavar='')
    ARGS = parser.parse_args()
    run(**vars(ARGS))