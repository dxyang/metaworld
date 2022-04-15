import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerReachEnvV2(SawyerXYZEnv):
    """
    Motivation for V2:
        V1 was very difficult to solve because the observation didn't say where
        to move (where to reach).
    Changelog from V1 to V2:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/15/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
        - (6/15/20) Separated reach-push-pick-place into 3 separate envs.
    """
    def __init__(self):
        goal_low = (-0.1, 0.5, 0.05)
        goal_high = (0.1, 0.8, 0.3)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0., 0.6, 0.02]),
            'hand_init_pos': np.array([0., 0.6, 0.2]),
        }

        self.goal = np.array([-0.1, 0.8, 0.2])

        self.obj_init_angle = self.init_config['obj_init_angle']
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.num_resets = 0

    @property
    def model_name(self):
        if self.use_franka:
            return full_v2_path_for('franka_xyz/franka_reach_v2.xml')
        else:
            return full_v2_path_for('sawyer_xyz/sawyer_reach_v2.xml')

    # @property
    # # TODO: dxy - decide how to adadpt this to different tasks
    # def observation_space(self):
    #     goal_low = np.zeros(3) if self._partially_observable \
    #         else self.goal_space.low
    #     goal_high = np.zeros(3) if self._partially_observable \
    #         else self.goal_space.high

    #     return Box(
    #         np.hstack((self._HAND_SPACE.low, goal_low)),
    #         np.hstack((self._HAND_SPACE.high, goal_high))
    #     )


    @_assert_task_is_set
    def evaluate_state(self, obs, action):

        reward, reach_dist, in_place = self.compute_reward(action, obs)
        success = float(reach_dist <= 0.05)

        info = {
            'success': success,
            'near_object': reach_dist,
            'grasp_success': 1.,
            'grasp_reward': reach_dist,
            'in_place_reward': in_place,
            'obj_to_target': reach_dist,
            'unscaled_reward': reward,
        }

        return reward, info

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def _get_quat_objects(self):
        return Rotation.from_matrix(
            self.data.get_geom_xmat('objGeom')
        ).as_quat()

    def fix_extreme_obj_pos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not
        # aligned. If this is not done, the object could be initialized in an
        # extreme position
        diff = self.get_body_com('obj')[:2] - \
               self.get_body_com('obj')[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        # The convention we follow is that body_com[2] is always 0,
        # and geom_pos[2] is the object height
        return [
            adjusted_pos[0],
            adjusted_pos[1],
            self.get_body_com('obj')[-1]
        ]

    def _get_obs(self):
        full_obs = super()._get_obs()
        eef_xyz = full_obs[:3]
        goal = full_obs[-3:]
        obs = np.concatenate([eef_xyz, goal])

        return obs


    def reset_model(self):
        self._reset_hand()

        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.fix_extreme_obj_pos(self.init_config['obj_init_pos'])
        self.obj_init_angle = self.init_config['obj_init_angle']

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
            while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
                goal_pos = self._get_state_rand_vec()
                self._target_pos = goal_pos[3:]
            self._target_pos = goal_pos[-3:]
            self.obj_init_pos = goal_pos[:3]

        self._set_obj_xyz(self.obj_init_pos)
        self.num_resets += 1

        return self._get_obs()

    def compute_reward(self, actions, obs):
        _TARGET_RADIUS = 0.05
        eef = self.get_endeff_pos()
        # obj = obs[4:7]
        # tcp_opened = obs[3]
        target = self._target_pos

        eef_to_target = np.linalg.norm(eef - target)
        # obj_to_target = np.linalg.norm(obj - target)

        in_place_margin = (np.linalg.norm(self.hand_init_pos - target))
        in_place = reward_utils.tolerance(eef_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='long_tail',)

        # dxy: clamping probably isn't necessary?
        reward = -eef_to_target
        return [reward, eef_to_target, eef_to_target]

    def set_goal(self, goal):
        self._target_pos = goal.copy()