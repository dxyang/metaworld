import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move

# going_up_hack = False
class SawyerDisassembleV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        # if np.allclose(obs[:3], np.array([0, 0.4, 0.2]), rtol=0.1, atol=0.1):
        #     global going_up_hack
        #     going_up_hack = False

        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'wrench_pos': obs[4:7],
            'peg_pos': obs[-3:],
            'unused_info': obs[7:-3],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=10.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_wrench = o_d['wrench_pos'] + np.array([-.02, .0, .01])
        pos_peg = o_d['peg_pos'] + np.array([.12, .0, .14])

        # If XY error is greater than 0.02, place end effector above the wrench
        if np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02:
            return pos_wrench + np.array([0., 0., 0.1])
        # Once XY error is low enough, drop end effector down on top of wrench
        elif abs(pos_curr[2] - pos_wrench[2]) > 0.03:
            return pos_wrench
        # Move upwards
        else:
            return pos_curr + np.array([.0, .0, .1])

        # global going_up_hack

        # if not going_up_hack:
        #     # If XY error is greater than 0.02, place end effector above the wrench
        #     if np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02:
        #         return pos_wrench + np.array([0., 0., 0.1])
        #     # Once XY error is low enough, drop end effector down on top of wrench
        #     elif abs(pos_curr[2] - pos_wrench[2]) > 0.035:
        #         return pos_wrench
        #     # Move upwards
        #     else:
        #         going_up_hack = True

        # if going_up_hack:
        #     return pos_curr + np.array([.0, .0, .1])

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_wrench = o_d['wrench_pos'] + np.array([-.02, .0, .01])

        if np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02 or abs(pos_curr[2] - pos_wrench[2]) > 0.07:
            return 0.
        else:
            return 0.8
