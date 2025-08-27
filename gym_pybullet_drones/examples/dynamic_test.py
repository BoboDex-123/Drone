from gym_pybullet_drones.envs.single_agent_rl.DynamicAviary import DynamicAviary
import time

if __name__ == "__main__":
    env = DynamicAviary(gui=True)
    obs = env.reset()

    # Run the simulation until the episode is done
    done = False
    while not done:
        action = env.action_space.sample()  # Random actions
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        time.sleep(1./240.) # This is important for real-time visualization

    env.close()
    
