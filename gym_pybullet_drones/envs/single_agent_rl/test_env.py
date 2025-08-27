import time
import numpy as np
from gym_pybullet_drones.envs.single_agent_rl.DynamicAviary import DynamicAviary
from gym_pybullet_drones.utils.utils import sync

if __name__ == "__main__":
    env = DynamicAviary(gui=True, record=False)
    obs, info = env.reset(seed=42)
    start = time.time()
    for i in range(3 * env.PYB_FREQ):
        action = env.action_space.sample() # Take random actions
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        sync(i, start, env.CTRL_TIMESTEP)
        if terminated:
            obs, info = env.reset(seed=42)
    env.close()