import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from sympy import false

class DynamicAviary(BaseRLAviary):
    """Single agent RL problem: hover at a static target."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3 dimensional)

        """
        self.TARGET_POS = np.array([0,0,1])
        self.EPISODE_LEN_SEC = 8
        self.obstacle_ids = []
        super().__init__(drone_model=drone_model,
                         num_drones=1,
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

    def _addObstacles(self):
        """Add obstacles to the environment."""
        for i in range(5):
            obstacle = p.loadURDF("cube.urdf",
                                basePosition=[np.random.uniform(-2, 2),
                                              np.random.uniform(-2, 2),
                                              np.random.uniform(0.5, 2.5)],
                                useFixedBase=False)
            self.obstacle_ids.append(obstacle)

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        # Simple reward based on distance to target
        reward = -np.linalg.norm(self.TARGET_POS - state[0:3])
        # Bonus for being close to the target
        if np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.1:
            reward += 10
        return reward

    ################################################################################

    def _computeTerminated(self):
        """Computes the current terminated value.

        Returns
        -------
        bool
            Whether the episode is terminated.

        """
        return False
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.01:
            return True
        return False

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the episode is truncated.

        """
        return False
        state = self._getDroneStateVector(0)
        if any(abs(state[i]) > 2 for i in range(3)) or state[2] < 0.1:
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            A dict containing information about the simulation state.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        spaces.Box
            A Box of shape (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### KINEMATIC INFORMATION (state)
            # state      - [x, y, z, q0, q1, q2, q3, r, p, y, vx, vy, vz, wx, wy, wz, last_action_...];
            # obs_lower_bound, obs_upper_bound
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo])
            obs_upper_bound = np.array([hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi])
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
        else:
            print("[ERROR] in DynamicAviary._observationSpace()")

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of shape (4,) depending on the action type.

        """
        if self.ACT_TYPE == ActionType.RPM:
            #### RPM
            size = 4
            act_lower_bound = np.zeros(size)
            act_upper_bound = np.ones(size) * self.MAX_RPM
            return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
        elif self.ACT_TYPE == ActionType.ONE_D_RPM:
            #### ONE_D_RPM
            size = 1
            act_lower_bound = np.zeros(size)
            act_upper_bound = np.ones(size) * self.MAX_RPM
            return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
        else:
            print("[ERROR] in DynamicAviary._actionSpace()")

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameters
        ----------
        action : ndarray
            The input action for one or more drones, to be translated into RPMs.

        Returns
        -------
        ndarray
            (4,)-shaped array of ints containing to clipped RPMs
            (i.e., clipping based on MAX_RPM).

        """
        if self.ACT_TYPE == ActionType.RPM:
            return np.clip(action, 0, self.MAX_RPM)
        elif self.ACT_TYPE == ActionType.ONE_D_RPM:
            return np.repeat(np.clip(action, 0, self.MAX_RPM), 4)
        else:
            print("[ERROR] in DynamicAviary._preprocessAction()")

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.KIN:
            state = self._getDroneStateVector(0)
            return state[0:12]
        else:
            print("[ERROR] in DynamicAviary._computeObs()")