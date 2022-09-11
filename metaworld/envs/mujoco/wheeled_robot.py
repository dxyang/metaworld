from gym.envs import register
from gym.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from gym import utils
# from lgpl.envs.env_utils import get_asset_xml
import gym
import os

class WheeledEnv(MujocoEnv, utils.EzPickle):

    FILE = 'wheeled.xml'

    def __init__(self, sample_goal_during_reset=True, sample_start_during_reset=True):
        self.goal = np.zeros((2,))
        self.frame_skip = 10
        self.sample_goal_during_reset = sample_goal_during_reset
        self.sample_start_during_reset = sample_start_during_reset
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets_v2/wheeled.xml')
        MujocoEnv.__init__(self, full_path, 10)
        utils.EzPickle.__init__(self)

    def reset_model(self, reset_args=None):
        curr_qpos = np.zeros((6,))
        curr_qvel = np.zeros((6,))
        if self.sample_goal_during_reset:
            radius = 2.0 
            angle = np.random.uniform(0, np.pi)
            xpos = radius*np.cos(angle)
            ypos = radius*np.sin(angle)
            self.goal = np.array([xpos, ypos])

        if self.sample_start_during_reset:
            xpos = np.random.uniform(-2., 2.)
            ypos = np.random.uniform(-2., 2.)
            curr_qpos[:2] = np.array([xpos, ypos])
    
        body_pos = self.sim.model.body_pos.copy()
        body_pos[-1][:2] = self.goal
        self.sim.model.body_pos[:] = body_pos
        self.set_state(curr_qpos, curr_qvel)
        self.sim.forward()
        return self.get_current_obs()

    def set_goal(self, goal):
        self.goal = goal
        return self.get_current_obs()
        
    def get_current_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            self.goal
        ]).reshape(-1)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        next_obs = self.get_current_obs()
        reward = -np.linalg.norm(next_obs[:2] - self.goal)
        done = False
        info = {}
        return next_obs, reward, done, info