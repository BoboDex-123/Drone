import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class MazeAviary(BaseAviary):
    """Single agent RL problem: navigate a maze."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        """
        self.EPISODE_LEN_SEC = 120
        self.maze = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ])
        self.OBS_TYPE = ObservationType.KIN
        self.ACT_TYPE = ActionType.PID
        if self.ACT_TYPE in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(1)]
            else:
                print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=True,
                         vision_attributes=True
                         )
        self.allies = []
        self.enemies = []
        self.detected_allies = set()
        self.detected_enemies = set()
        self.previous_detected_allies = 0
        self.previous_detected_enemies = 0

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment."""
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i, j] == 1:
                    p.loadURDF("cube_small.urdf",
                               [i, j, 0.5],
                               p.getQuaternionFromEuler([0, 0, 0]),
                               useFixedBase=True,
                               physicsClientId=self.CLIENT
                               )

    def _addAgents(self):
        # Add allies
        self.allies.append(p.loadURDF("sphere_small.urdf", [1, 1, 0.5], useFixedBase=True, physicsClientId=self.CLIENT))
        p.changeVisualShape(self.allies[-1], -1, rgbaColor=[0, 0, 1, 1])

        # Add enemies
        self.enemies.append(p.loadURDF("sphere_small.urdf", [3, 3, 0.5], useFixedBase=True, physicsClientId=self.CLIENT))
        p.changeVisualShape(self.enemies[-1], -1, rgbaColor=[1, 0, 0, 1])


    def reset(self, seed: int = None, options: dict = None):
        obs, info = super().reset(seed=seed, options=options)
        self._addAgents()
        self.detected_allies = set()
        self.detected_enemies = set()
        self.previous_detected_allies = 0
        self.previous_detected_enemies = 0
        return obs, info

    ################################################################################
    
    def _actionSpace(self):
        if self.ACT_TYPE==ActionType.PID:
            size = 3
        else:
            print("[ERROR] in MazeAviary._actionSpace()")
            exit()
        act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        rpm = np.zeros((self.NUM_DRONES,4))
        for k in range(action.shape[0]):
            target = action[k, :]
            if self.ACT_TYPE == ActionType.PID:
                state = self._getDroneStateVector(k)
                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target,
                    step_size=1,
                    )
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=next_pos
                                                        )
                rpm[k,:] = rpm_k
            else:
                print("[ERROR] in MazeAviary._preprocessAction()")
                exit()
        return rpm

    ################################################################################

    def _observationSpace(self):
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.KIN:
            return spaces.Box(low=0, high=255, shape=(self.IMG_RES[1], self.IMG_RES[0], 1), dtype=np.uint8)
        else:
            print("[ERROR] in MazeAviary._observationSpace()")
    
    ################################################################################

    def _computeObs(self):
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                self.rgb[0], self.dep[0], self.seg[0] = self._getDroneImages(0,
                                                                                 segmentation=False
                                                                                 )
                #### Printing observation to PNG frames example ############
                if self.RECORD:
                    self._exportImage(img_type=ImageType.RGB,
                                      img_input=self.rgb[0],
                                      path=self.ONBOARD_IMG_PATH,
                                      frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                      )
            return self.rgb[0]
        elif self.OBS_TYPE == ObservationType.KIN:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                self.rgb[0], self.dep[0], self.seg[0] = self._getDroneImages(0,
                                                                                 segmentation=False
                                                                                 )
                #### Printing observation to PNG frames example ############
                if self.RECORD:
                    self._exportImage(img_type=ImageType.DEP,
                                      img_input=self.dep[0],
                                      path=self.ONBOARD_IMG_PATH,
                                      frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                      )
            return self.dep[0].reshape(self.IMG_RES[1], self.IMG_RES[0], 1)
        else:
            print("[ERROR] in MazeAviary._computeObs()")

    def _computeReward(self):
        reward = 0
        # Reward for detecting new allies and enemies
        if len(self.detected_allies) > self.previous_detected_allies:
            reward += 10
        if len(self.detected_enemies) > self.previous_detected_enemies:
            reward += 10
        
        self.previous_detected_allies = len(self.detected_allies)
        self.previous_detected_enemies = len(self.detected_enemies)

        # Penalty for collision
        if self.step_counter > 0 and len(p.getContactPoints(self.DRONE_IDS[0])) > 0:
            reward -= 100

        # Penalty for each step
        reward -= 0.1
        return reward

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value."""
        if len(p.getContactPoints(self.DRONE_IDS[0])) > 0:
            return True
        return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value."""
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s)."""
        return {"detected_allies": len(self.detected_allies), "detected_enemies": len(self.detected_enemies)}