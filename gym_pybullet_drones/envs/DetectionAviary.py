import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class DetectionAviary(BaseRLAviary):
    """Multi-drone environment for target detection."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: str='kin',
                 act: str='rpm',
                 num_friends: int=None,
                 num_enemies: int=None
                 ):
        """Initialization of a multi-drone environment for target detection."""
        if num_friends is None:
            self.NUM_FRIENDS = np.random.randint(1, 6)
        else:
            self.NUM_FRIENDS = num_friends
        if num_enemies is None:
            self.NUM_ENEMIES = np.random.randint(1, 6)
        else:
            self.NUM_ENEMIES = num_enemies
            
        self.friend_ids = []
        self.enemy_ids = []
        self.detected_friends = set()
        self.detected_enemies = set()
        self.EPISODE_LEN_SEC = 20
        if initial_xyzs is None:
            initial_xyzs = np.array([[0, 0, 1.0] for i in range(num_drones)])
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################

    def reset(self, seed: int = None, options: dict = None):
        """Resets the environment."""
        self.detected_friends = set()
        self.detected_enemies = set()
        return super().reset(seed=seed, options=options)

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment."""
        for i in range(self.NUM_FRIENDS):
            friend_pos = [np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0.1]
            friend_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=0.1,
                    rgbaColor=[0, 0, 1, 1]
                ),
                basePosition=friend_pos,
                physicsClientId=self.CLIENT
            )
            self.friend_ids.append(friend_id)

        for i in range(self.NUM_ENEMIES):
            enemy_pos = [np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0.1]
            enemy_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=0.1,
                    rgbaColor=[1, 0, 0, 1]
                ),
                basePosition=enemy_pos,
                physicsClientId=self.CLIENT
            )
            self.enemy_ids.append(enemy_id)

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value."""
        reward = 0
        state = self._getDroneStateVector(0)
        for friend_id in self.friend_ids:
            friend_pos, _ = p.getBasePositionAndOrientation(friend_id, physicsClientId=self.CLIENT)
            dist_to_friend = np.linalg.norm(state[0:3] - friend_pos)
            if dist_to_friend < 0.2 and friend_id not in self.detected_friends:
                reward += 100
                self.detected_friends.add(friend_id)

        for enemy_id in self.enemy_ids:
            enemy_pos, _ = p.getBasePositionAndOrientation(enemy_id, physicsClientId=self.CLIENT)
            dist_to_enemy = np.linalg.norm(state[0:3] - enemy_pos)
            if dist_to_enemy < 0.2 and enemy_id not in self.detected_enemies:
                reward -= 100
                self.detected_enemies.add(enemy_id)
        return reward

    ################################################################################

    def _computeTerminated(self):
        """Computes the current terminated value."""
        state = self._getDroneStateVector(0)
        if state[2] < 0.1:
            return True
        return False

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value."""
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s)."""
        return {"detected_friends": len(self.detected_friends), "detected_enemies": len(self.detected_enemies)}

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment."""
        state = self._getDroneStateVector(0)
        
        # Find nearest undetected friend and enemy
        friend_positions = [p.getBasePositionAndOrientation(f_id, physicsClientId=self.CLIENT)[0] for f_id in self.friend_ids if f_id not in self.detected_friends]
        enemy_positions = [p.getBasePositionAndOrientation(e_id, physicsClientId=self.CLIENT)[0] for e_id in self.enemy_ids if e_id not in self.detected_enemies]

        dist_to_friends = [np.linalg.norm(state[0:3] - pos) for pos in friend_positions]
        dist_to_enemies = [np.linalg.norm(state[0:3] - pos) for pos in enemy_positions]

        nearest_friend_pos = friend_positions[np.argmin(dist_to_friends)] if len(dist_to_friends) > 0 else [0,0,0]
        nearest_enemy_pos = enemy_positions[np.argmin(dist_to_enemies)] if len(dist_to_enemies) > 0 else [0,0,0]

        return np.hstack([state, nearest_friend_pos - state[0:3], nearest_enemy_pos - state[0:3]]).reshape(26)

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment."""
        return spaces.Box(low=np.array([-np.inf]*26),
                           high=np.array([np.inf]*26),
                           dtype=np.float32
                           )