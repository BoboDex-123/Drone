from gymnasium.envs.registration import register

register(
    id='dynamic-aviary-v0',
    entry_point='gym_pybullet_drones.envs.single_agent_rl.DynamicAviary:DynamicAviary',
    max_episode_steps=1000,
)
register(
    id='ctrl-aviary-v0',
    entry_point='gym_pybullet_drones.envs:CtrlAviary',
)

register(
    id='velocity-aviary-v0',
    entry_point='gym_pybullet_drones.envs:VelocityAviary',
)

register(
    id='hover-aviary-v0',
    entry_point='gym_pybullet_drones.envs:HoverAviary',
)

register(
    id='multihover-aviary-v0',
    entry_point='gym_pybullet_drones.envs:MultiHoverAviary',
)
