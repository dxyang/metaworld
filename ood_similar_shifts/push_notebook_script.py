import math
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn
import numpy as np
import matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt
import copy
# from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import time
plt.ion()   # interactive mode - o
# from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#break bc.
import argparse
import pdb
import functools
import os
import random
import matplotlib
# matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
import gym
from gym.spaces import Box
import copy

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import test_cases_latest_nonoise
from render_utils import trajectory2vid
from networks import *
import time

from stable_baselines3 import SAC

def collect_data(env, policy, num_trajs=1000, mode='in_dist', device=None, render=False):
    """Collect expert rollouts from the environment, store it in some datasets"""
    trajs = []
    obj_poses = []
    goal_poses = []
    for tn in range(num_trajs):
        # print("Traj num %d"%tn)
        env.reset()
        if mode == 'in_dist':
            goal_pos = np.random.uniform(low=np.array([0, 0.5, 0.01]),
                                         high=np.array([0.1, 0.6, 0.01]), size=(3,))
        elif mode == 'ood':
            goal_pos = np.random.uniform(low=np.array([-0.1, 0.5, 0.01]),
                                         high=np.array([0., 0.6, 0.01]), size=(3,))
        o, obj_pos, goal_pos = env.reset_model_ood(mode, 0, 0.5, None, True, goal_pos=goal_pos)

        traj = {'obs': [],'action': [], 'next_obs': [], 'done': [], 'reward': [], 'tcp': []}
#         print(goal_pos)
        for _ in range(horizon):
            if hasattr(policy,'predict'): #sac
                ac, _ = policy.predict(o, deterministic=True)
            elif hasattr(policy,'get_action'):
                ac = policy.get_action(o)
            else:
                t1s = torch.Tensor(o[None]).to(device)
                ac = policy(t1s).cpu().detach().numpy()[0]
            no, r, _, info = env.step(ac)
            traj['obs'].append(o.copy())
            traj['action'].append(ac.copy())
            traj['next_obs'].append(no.copy())
            traj['done'].append(info['success'])
            traj['reward'].append(info['in_place_reward'])
            traj['tcp'].append(env.tcp_center)
            o = no
#             time.sleep(0.01)
            if render == True:
                env.render()

        traj['obs'] = np.array(traj['obs'])
        traj['action'] = np.array(traj['action'])
        traj['next_obs'] = np.array(traj['next_obs'])
        traj['done'] = np.array(traj['done'])
        # traj['tcp'] = env.tcp_center #gripper at end of trajectory TODO not good indicator for all tasks
        traj['tcp'] = np.array(traj['tcp'])
        trajs.append(traj)
        obj_poses.append(obj_pos)
        goal_poses.append(goal_pos)
    return trajs, obj_poses, goal_poses

# Utilities for defining neural nets
def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None
    ):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def plot_3d_viz(trajs, tag, nplot=20, threeD=True, gripper=False):
    colors = sns.color_palette("hls", nplot)
    plot_idx = random.sample(range(len(trajs)), k=nplot) #random
    fig = plt.figure()
    if threeD:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0.4, 1.0)
    end_dists = []
    tcp_dists = []
    obj_dists = []
    for colorid, idx in enumerate(plot_idx):
        t = np.array(trajs[idx]['obs'])
        tcp = np.array(trajs[idx]['tcp'])
        if threeD:
            if gripper:
                # ax.plot3D(t[:,0], t[:,1], t[:,2], color=colors[colorid], linestyle=':') #end effector traj
                ax.plot3D(tcp[:,0], tcp[:,1], tcp[:,2], color=colors[colorid], linestyle=':') #tcp
            else:
                ax.plot3D(t[:,4], t[:,5], t[:,6], color=colors[colorid], linestyle=':') #obj
            ax.scatter3D(t[-1,-3], t[-1,-2], t[-1,-1], color=colors[colorid], marker='x') #gt goal
        else:
            ax.plot(t[:,0], t[:,1], color=colors[colorid], linestyle=':') #end effector traj
            ax.scatter(t[-1,-3], t[-1,-2], color=colors[colorid], marker='x') #gt goal
        end_dists.append(np.linalg.norm(t[-1][:3] - t[-1][-3:])) #dist between end effector and gt goal
        tcp_dists.append(np.linalg.norm(tcp[-1][:3] - t[-1][-3:])) #dist between tcp and gt goal
        obj_dists.append(np.linalg.norm(t[-1][4:7] - t[-1][-3:])) #dist between obj and gt goal
    end_dists = np.array(end_dists)
    tcp_dists = np.array(tcp_dists)
    obj_dists = np.array(obj_dists)
    plt.title(str(round(np.mean(end_dists),4))+' '+str(round(np.mean(tcp_dists),4))+' '+str(round(np.mean(obj_dists),4)))
    fig.savefig(tag)


gen_type = 'goal'
horizon = 100
env_name = 'push-v2'
reach_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name+"-goal-observable"]
env = reach_goal_observable_cls(seed=0)

base_fig_dir = os.path.join('figs',env_name,'notebook_script3D')
if not os.path.exists(base_fig_dir):
    os.makedirs(base_fig_dir)

env.random_init = False
env._freeze_rand_vec = False
vae_env = reach_goal_observable_cls(seed=0)
vae_env.random_init = True
vae_env._freeze_rand_vec = False
obs_size = env.observation_space.shape[0]
ac_size = env.action_space.shape[0]
goal_size = env.goal_space.shape[0]
obj_size = len(env.obj_init_pos)

n_expert_traj = 100
n_vae_traj = 1000
n_eval_traj = 10
num_epochs_bc = 500
num_epochs_vae = 50
num_epochs_res = 500

# Data collection
print('EXPERT')
#for each task, create 1000 expert demos
expert_policy = functools.reduce(lambda a,b : a if a[0] == env_name else b, test_cases_latest_nonoise)[1]

######
#SAC expert policy
expert_path = os.path.join('expert_policy', env_name)
if not os.path.exists(expert_path):
    os.makedirs(expert_path)
    #train model
expert_policy_path = os.path.join(expert_path,"sac_"+env_name)
# if not os.path.exists(expert_policy_path):
if False:    
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save(expert_policy_path)
    del model # remove to demonstrate saving and loading
    expert_policy = SAC.load(expert_policy_path)
######


trajs_unprocessed, traj_obj_poses, traj_goal_poses = collect_data(env, expert_policy, num_trajs=n_expert_traj, mode='in_dist') #1000
trajs_unprocessed_ood, traj_obj_poses_ood, traj_goal_poses_ood = collect_data(env, expert_policy, num_trajs=n_expert_traj, mode='ood') #1000

fig = plt.figure()
plt.scatter(np.array(traj_goal_poses)[:, 0], np.array(traj_goal_poses)[:, 1], color='r')
plt.scatter(np.array(traj_goal_poses_ood)[:, 0], np.array(traj_goal_poses_ood)[:, 1], color='b')
plt.savefig(os.path.join(base_fig_dir, 'goals.png'))

# Post process
trajs = []
reduced_state_space = True
for traj in trajs_unprocessed:
    traj_new = {'obs': [], 'action': [], 'next_obs': [], 'tcp': []}
    if reduced_state_space:
        traj_new['obs'] = np.concatenate([traj['obs'][:, :7].copy(),
                                          traj['obs'][:, -3:].copy()], axis=-1).copy()
        traj_new['action'] = traj['action'].copy()
        traj_new['next_obs'] = np.concatenate([traj['next_obs'][:, :7].copy(),
                                               traj['next_obs'][:, -3:].copy()], axis=-1).copy()
    else:
        traj_new['obs'] = traj['obs'].copy()
        traj_new['action'] = traj['action'].copy()
        traj_new['next_obs'] = traj_new['next_obs'].copy()
    
    traj_new['tcp'] = traj['tcp'].copy()

    trajs.append(traj_new)

# fig = plt.figure()
# plt.xlim(-0.5, 0.5)
# plt.ylim(0.4, 1.0)
# for traj in trajs:
#     plt.plot(traj['obs'][:, 0], traj['obs'][:, 1])
#     plt.scatter(traj['obs'][-1, -3], traj['obs'][-1, -2])
# plt.savefig(os.path.join(base_fig_dir, 'expert_gripper.png'))
plot_3d_viz(trajs, os.path.join(base_fig_dir, 'expert_tcp.png'), nplot=20, threeD=True, gripper=True)

# fig = plt.figure()
# plt.xlim(-0.5, 0.5)
# plt.ylim(0.4, 1.0)
# for traj in trajs:
#     plt.plot(traj['obs'][:, 4], traj['obs'][:, 5])
#     plt.scatter(traj['obs'][-1, -3], traj['obs'][-1, -2])
# plt.savefig(os.path.join(base_fig_dir, 'expert_obj.png'))
plot_3d_viz(trajs, os.path.join(base_fig_dir, 'expert_obj.png'), nplot=20, threeD=True)


def get_latent(x):
    if vae_mode:
        raise NotImplementedError("vae not implemented")
    else:
        return x

# Define the forward model
class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, output_mod=None):
        super().__init__()
        self.trunk = mlp(obs_dim, hidden_dim, action_dim, hidden_depth, output_mod=output_mod)
        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs):
        next_pred = self.trunk(obs)
        return next_pred

hidden_layer_size = 32
hidden_depth = 1
if reduced_state_space:
    obs_size = 10
    ac_size = 4
else:
    obs_size = 39
    ac_size = 4
policy = Policy(obs_size, ac_size, hidden_layer_size, hidden_depth) # 10 dimensional latent
num_tasks = len(trajs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy.to(device)
mode = 'concat'
vae_mode = False
env1 = env


vae_mode = False
kNN_mode = 'learned' # options are learned and kNN
kNN_valid_modes = ['learned', 'kNN', 'uniform'] # TODO: Add in kNN
assert kNN_mode in kNN_valid_modes, "Not implemented this kNN mode"
train_mode = 'all_to_all'
weighted_loss_inout = 'out'
num_nn = 32

# Train standard goal conditioned policy
num_epochs = 10000
batch_size = 32
EPS = 1e-9

criterion = nn.MSELoss()
optimizer = optim.Adam(list(policy.parameters()))

losses = []

idxs = np.array(range(len(trajs)))

num_batches = len(idxs) // batch_size
losses = []
# Train the model with regular SGD
for epoch in range(num_epochs):  # loop over the dataset multiple times
    np.random.shuffle(idxs)
    running_loss = 0.0
    for i in range(num_batches):

        optimizer.zero_grad()

        t1_idx = np.random.randint(len(trajs), size=(batch_size,)) # Indices of first trajectory
        t1_idx_pertraj = np.random.randint(horizon, size=(batch_size,))
        t1_states = np.concatenate([trajs[c_idx]['obs'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
        t1_actions = np.concatenate([trajs[c_idx]['action'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])

        t1_states = torch.Tensor(t1_states).float().to(device)
        t1_actions = torch.Tensor(t1_actions).float().to(device)

        a1_pred = policy(t1_states.to(device)) #first action prediction


        loss = torch.mean(torch.linalg.norm(a1_pred - t1_actions, dim=-1))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 0:    # print every 2000 mini-batches
            # print('[%d, %5d] loss: %.8f' %
            #       (epoch + 1, i + 1, running_loss / 10.))
            losses.append(running_loss/10.)
            running_loss = 0.0
        losses.append(loss.item())

print('Finished Training')
fig = plt.figure()
plt.plot(losses)
plt.savefig(os.path.join(base_fig_dir, 'bc_losses.png'))


num_test_trajs = 20
mode = 'in_dist'
trajs_bc_test = []
obj_poses_bc_test = []
goal_poses_bc_test = []

def process_o(o):
    if reduced_state_space:
        o_proc = np.concatenate([o[:7], o[-3:]]).copy()
    else:
        o_proc = o.copy()
    return o_proc

for tn in range(num_test_trajs):
    env.reset()
    goal_pos =  trajs[tn]['obs'][-1, -3:].copy()
    o, obj_pos, goal_pos = env.reset_model_ood(mode, 0, 0.5, None, True, goal_pos=goal_pos)
    o = process_o(o)
    traj = {'obs': [],'action': [], 'next_obs': [], 'done': [], 'reward': [], 'tcp': []}
    for _ in range(horizon):
        if hasattr(policy,'get_action'):
            ac = policy.get_action(o)
        else:
            t1s = torch.Tensor(o[None]).to(device)
            ac = policy(t1s).cpu().detach().numpy()[0]
        no, r, _, info = env.step(ac)
        no = process_o(no)
        traj['obs'].append(o.copy())
        traj['action'].append(ac.copy())
        traj['next_obs'].append(no.copy())
        traj['done'].append(info['success'])
        traj['reward'].append(info['in_place_reward'])
        traj['tcp'].append(env.tcp_center)
        o = no
#         env.render()
#         time.sleep(0.01)

    traj['obs'] = np.array(traj['obs'])
    traj['action'] = np.array(traj['action'])
    traj['next_obs'] = np.array(traj['next_obs'])
    traj['done'] = np.array(traj['done'])
    # traj['tcp'] = env.tcp_center #gripper at end of trajectory TODO not good indicator for all tasks
    traj['tcp'] = np.array(traj['tcp'])
    trajs_bc_test.append(traj)
    obj_poses_bc_test.append(obj_pos)
    goal_poses_bc_test.append(goal_pos)

# fig = plt.figure()
# plt.xlim(-0.5, 0.5)
# plt.ylim(0.4, 1.0)
# for traj in trajs_bc_test:
#     plt.plot(traj['obs'][:, 0], traj['obs'][:, 1])
#     plt.scatter(traj['obs'][-1, -3], traj['obs'][-1, -2])
# plt.savefig(os.path.join(base_fig_dir, 'bc_gripper.png'))
plot_3d_viz(trajs_bc_test, os.path.join(base_fig_dir, 'bc_tcp.png'), nplot=20, threeD=True, gripper=True)

# fig = plt.figure()
# plt.xlim(-0.5, 0.5)
# plt.ylim(0.4, 1.0)
# for traj in trajs_bc_test:
#     plt.plot(traj['obs'][:, 4], traj['obs'][:, 5])
#     plt.scatter(traj['obs'][-1, -3], traj['obs'][-1, -2])
# plt.savefig(os.path.join(base_fig_dir, 'bc_obj.png'))
plot_3d_viz(trajs_bc_test, os.path.join(base_fig_dir, 'bc_obj.png'), nplot=20, threeD=True)

fig, ax = plt.subplots(4, 1)
for traj in trajs:
    ax[0].plot(traj['action'][:, 0])
    ax[1].plot(traj['action'][:, 1])
    ax[2].plot(traj['action'][:, 2])
    ax[3].plot(traj['action'][:, 3])
plt.savefig(os.path.join(base_fig_dir, 'expert_acts.png'))

num_test_trajs = 10
mode = 'ood'
trajs_bc_test = []
obj_poses_bc_test = []
goal_poses_bc_test = []

for tn in range(num_test_trajs):
    env.reset()
    goal_pos = np.random.uniform(low=np.array([-0.1, 0.5, 0.01]),
                                 high=np.array([0., 0.6, 0.01]), size=(3,))
    o, obj_pos, goal_pos = env.reset_model_ood(mode, 0, 0.5, None, True, goal_pos=goal_pos)
    o = process_o(o)
    traj = {'obs': [],'action': [], 'next_obs': [], 'done': [], 'reward': [], 'tcp': []}
    for _ in range(env.max_path_length):
        if hasattr(policy,'get_action'):
            ac = policy.get_action(o)
        else:
            t1s = torch.Tensor(o[None]).to(device)
            ac = policy(t1s).cpu().detach().numpy()[0]
        no, r, _, info = env.step(ac)
        no = process_o(no)
        traj['obs'].append(o.copy())
        traj['action'].append(ac.copy())
        traj['next_obs'].append(no.copy())
        traj['done'].append(info['success'])
        traj['reward'].append(info['in_place_reward'])
        traj['tcp'].append(env.tcp_center)
        o = no
#         env.render()
#         time.sleep(0.01)

    traj['obs'] = np.array(traj['obs'])
    traj['action'] = np.array(traj['action'])
    traj['next_obs'] = np.array(traj['next_obs'])
    traj['done'] = np.array(traj['done'])
    # traj['tcp'] = env.tcp_center #gripper at end of trajectory TODO not good indicator for all tasks
    traj['tcp'] = np.array(traj['tcp'])
    trajs_bc_test.append(traj)
    obj_poses_bc_test.append(obj_pos)
    goal_poses_bc_test.append(goal_pos)

# fig = plt.figure()
# plt.xlim(-0.5, 0.5)
# plt.ylim(0.4, 1.0)
# for traj in trajs_bc_test:
#     plt.plot(traj['obs'][:, 0], traj['obs'][:, 1]) #gripper
#     plt.plot(traj['obs'][:, 4], traj['obs'][:, 5]) #obj
#     plt.scatter(traj['obs'][-1, -3], traj['obs'][-1, -2])
# plt.savefig(os.path.join(base_fig_dir, 'bc_ood.png'))
plot_3d_viz(trajs_bc_test, os.path.join(base_fig_dir, 'bc_ood_tcp.png'), nplot=10, threeD=True, gripper=True)

# residual_policy = ResidualModel(delta_g_size=2, a_size=2, outshape=2, ctrlshape=(8,8), hidden_depth=2, hidden_size=256)
# residual_policy.to(device)
train_mode = 'all_to_all'
if train_mode == 'all_to_all':
    residual_policy = Policy(8, 4, 32, 1) # 10 dimensional latent
    residual_policy.to(device)

    num_epochs = 20000
    batch_size = 32
    EPS = 1e-9

    optimizer = optim.Adam(list(residual_policy.parameters()))

    losses = []
    idxs = np.array(range(len(trajs)))

    num_batches = len(idxs) // batch_size
    losses = []
    # Train the model with regular SGD
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        np.random.shuffle(idxs)
        running_loss = 0.0
        for i in range(num_batches):

            optimizer.zero_grad()
            t1_idx = np.random.randint(len(trajs), size=(batch_size,)) # Indices of first trajectory
            t2_idx = np.random.randint(len(trajs), size=(batch_size,)) # Indices of second trajectory

            t1_idx_pertraj = np.random.randint(horizon, size=(batch_size,))
            t1_states = np.concatenate([trajs[c_idx]['obs'][t_idx][None]
                                        for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
            t1_actions = np.concatenate([trajs[c_idx]['action'][t_idx][None]
                                         for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])

            t1_states = torch.Tensor(t1_states).float().to(device)
            t1_actions = torch.Tensor(t1_actions).float().to(device)

            a1_pred = policy(t1_states.to(device)) #first action prediction

            t2_states = np.concatenate([trajs[c_idx]['obs'][t_idx][None]
                                        for (c_idx, t_idx) in zip(t2_idx, t1_idx_pertraj)])
            t2_actions = np.concatenate([trajs[c_idx]['action'][t_idx][None]
                                         for (c_idx, t_idx) in zip(t2_idx, t1_idx_pertraj)])

            t2_states = torch.Tensor(t2_states).float().to(device)
            t2_actions = torch.Tensor(t2_actions).float().to(device)

            a2_pred = policy(t2_states.to(device)).detach() #first action prediction


            g_delta_e = np.concatenate([get_latent(trajs[t2_idx_diff]['obs'][-1, -3:])[None] -
                                        get_latent(trajs[t1_idx_diff]['obs'][-1, -3:])[None]
                                         for (t1_idx_diff, t2_idx_diff) in zip(t1_idx, t2_idx)])

            g_delta = g_delta_e
            g_delta_torch = torch.Tensor(g_delta).float().to(device)
            time_idx_in = torch.Tensor(t1_idx_pertraj)[:, None].to(device)
            res_in = torch.cat([a1_pred.detach(), g_delta_torch, time_idx_in], dim=-1)
            residual_pred = residual_policy(res_in)
            a2_pred_residual = residual_pred

    #         residual_pred, _ = residual_policy(a1_pred, g_delta_torch)
    #         a2_pred_residual = residual_pred

            # L2 regression on actions
            loss = torch.mean(torch.linalg.norm(a2_pred_residual - a2_pred, dim=-1))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:    # print every 2000 mini-batches
                # print('[%d, %5d] loss: %.8f' %
                #       (epoch + 1, i + 1, running_loss / 10.))
                losses.append(running_loss/10.)
                running_loss = 0.0
            losses.append(loss.item())

    print('Finished Training')
    fig = plt.figure()
    plt.plot(losses)
    plt.savefig(os.path.join(base_fig_dir, 'res_losses.png'))

all_diffs_e = []
for i in range(100):
    for j in range(100):        
        diff = get_latent(trajs[i]['obs'][-1, -3:])[None] - get_latent(trajs[j]['obs'][-1, -3:])[None]
        all_diffs_e.append(diff.copy())
all_diffs_e = np.concatenate(all_diffs_e)

def run_eval_residual_pergoal_bestworst(quad=1, threeD=True, gripper=True):
    # Check residuals on OOD points
    colors = sns.color_palette("hls", 10)
    size_sample = 10
    train_goals = np.concatenate([t['obs'][-1][-3:][None].copy() for t in trajs])
    fig = plt.figure()
    if threeD:
        ax = plt.axes(projection='3d')
    else:
        plt.cla()
        plt.clf()
        plt.xlim(-0.5, 0.5)
        plt.ylim(0.4, 1.0)

    end_dists = []
    tcp_dists = []
    obj_dists = []
    mode = 'ood'
    all_diffs_extrap_e = []
    for g_idx in range(size_sample):
        #quad3
        if quad == 3:
            goal_pos = np.random.uniform(low=np.array([-0.1, 0.4, 0.01]),
                                        high=np.array([0., 0.5, 0.01]), size=(3,))
        #quad2
        elif quad == 2:
            goal_pos = np.random.uniform(low=np.array([-0.1, 0.5, 0.01]),
                                        high=np.array([0., 0.6, 0.01]), size=(3,))
        #quad1 train
        elif quad == 1:
            goal_pos = np.random.uniform(low=np.array([0, 0.5, 0.01]),
                                        high=np.array([0.1, 0.6, 0.01]), size=(3,))
        #quad IV
        elif quad == 4:
            goal_pos = np.random.uniform(low=np.array([0., 0.4, 0.01]),
                                        high=np.array([0.1, 0.5, 0.01]), size=(3,))
        end_pos = goal_pos.copy()

        traj_curr_goal = []
        dist_curr_goal = []

        for k in range(len(trajs)):
            o = env1.reset()
            o, obj_pos, goal_pos = env.reset_model_ood('ood', 0, 0.5, None, True, goal_pos=goal_pos)
            o = process_o(o)

            #Find closest point
            closest_point_idx = k
            closest_point = train_goals[closest_point_idx].copy()
            closest_traj_obs = trajs[closest_point_idx]['obs'].copy()

            g_diff = end_pos - closest_point
            g_diff_t = torch.Tensor(g_diff[None]).to(device)

            traj = {'obs': [],'action': [], 'next_obs': [], 'tcp': []}
            for i in range(horizon):
                t1s = torch.Tensor(closest_traj_obs[i][None]).to(device) #torch.Tensor(o[None]).to(device)
                ac = policy(t1s).detach()

                t_in = torch.Tensor(np.array([[i]])).to(device)
                res_in = torch.cat([ac, g_diff_t, t_in], dim=-1)
                ac_final = residual_policy(res_in).cpu().detach().numpy()[0]

#                 ac_final = np.concatenate([ac_final, np.array([0., 0.])])
                no, r, d, _ = env1.step(ac_final)
                no = process_o(no).copy()
                traj['obs'].append(o.copy())
                traj['action'].append(ac_final.copy())
                traj['next_obs'].append(no.copy())
                traj['tcp'].append(env1.tcp_center)
                o = no.copy()
#                 env1.render()
            traj_curr_goal.append(traj)
            dist_curr_goal.append(np.linalg.norm(traj['obs'][-1][:3] - end_pos.copy()))

        dist_curr_goal = np.array(dist_curr_goal)
        best_idx = np.argmin(dist_curr_goal)
        best_traj = traj_curr_goal[best_idx]

        worst_idx = np.argmax(dist_curr_goal)
        worst_traj = traj_curr_goal[worst_idx]

        if threeD:
            #tcp
            if gripper:
                ax.plot3D(np.array(best_traj['tcp'])[:, 0], np.array(best_traj['tcp'])[:, 1], np.array(best_traj['tcp'])[:, 2], color=colors[g_idx])
                ax.plot3D(np.array(worst_traj['tcp'])[:, 0], np.array(worst_traj['tcp'])[:, 1], np.array(worst_traj['tcp'])[:, 2], linestyle=':', color=colors[g_idx])                
            #obj
            else:
                ax.plot3D(np.array(best_traj['obs'])[:, 4], np.array(best_traj['obs'])[:, 5], np.array(best_traj['obs'])[:, 6], color=colors[g_idx])
                ax.plot3D(np.array(worst_traj['obs'])[:, 4], np.array(worst_traj['obs'])[:, 5], np.array(worst_traj['obs'])[:, 6], linestyle=':', color=colors[g_idx])
            #goal
            ax.scatter3D([end_pos[0]],[end_pos[1]], [end_pos[2]], color=colors[g_idx], marker='x', s=50)
            ax.scatter3D([trajs[best_idx]['obs'][-1][-3]], [trajs[best_idx]['obs'][-1][-2]], [trajs[best_idx]['obs'][-1][-1]], color=colors[g_idx], marker='^', s=50)
            ax.scatter3D([trajs[worst_idx]['obs'][-1][-3]], [trajs[worst_idx]['obs'][-1][-2]], [trajs[worst_idx]['obs'][-1][-1]], color=colors[g_idx], marker='o', s=50)
        else:
            #obj
            plt.plot(np.array(best_traj['obs'])[:, 4], np.array(best_traj['obs'])[:, 5], color=colors[g_idx])
            plt.plot(np.array(worst_traj['obs'])[:, 4], np.array(worst_traj['obs'])[:, 5], linestyle=':', color=colors[g_idx])
            #goal
            plt.scatter([end_pos[0]],[end_pos[1]], color=colors[g_idx], marker='x', s=50)
            plt.scatter([trajs[best_idx]['obs'][-1][-3]], [trajs[best_idx]['obs'][-1][-2]], color=colors[g_idx], marker='^', s=50)
            plt.scatter([trajs[worst_idx]['obs'][-1][-3]], [trajs[worst_idx]['obs'][-1][-2]], color=colors[g_idx], marker='o', s=50)
#         plt.show()
        end_dists.append(np.linalg.norm(best_traj['obs'][-1][:3] - end_pos.copy())) #dist between end effector and gt goal
        tcp_dists.append(np.linalg.norm(best_traj['tcp'][-1][:3] - end_pos.copy())) #dist between tcp and gt goal
        obj_dists.append(np.linalg.norm(best_traj['obs'][-1][4:7] - end_pos.copy())) #dist between obj and gt goal        
        g_diff = get_latent(end_pos) - get_latent(trajs[best_idx]['obs'][-1][-3:])
        all_diffs_extrap_e.append(g_diff)

    end_dists = np.array(end_dists)
    tcp_dists = np.array(tcp_dists)
    obj_dists = np.array(obj_dists)
        
    all_diffs_extrap_e = np.array(all_diffs_extrap_e)
    print("Average end distance is %.5f"%(np.mean(end_dists)))
    plt.title(str(round(np.mean(end_dists),4))+' '+str(round(np.mean(tcp_dists),4))+' '+str(round(np.mean(obj_dists),4)))
    plt.savefig(os.path.join(base_fig_dir, 'test_residual_reach_worstbest'+str(quad)+'.png'))

    fig = plt.figure()
    if threeD:
        ax = plt.axes(projection='3d')
        ax.scatter3D(all_diffs_e[:, 0], all_diffs_e[:, 1], all_diffs_e[:, 2])
        ax.scatter3D(all_diffs_extrap_e[:, 0], all_diffs_extrap_e[:, 1], all_diffs_extrap_e[:, 2], marker='x', color='r')
    else:
        plt.scatter(all_diffs_e[:, 0], all_diffs_e[:, 1])
        plt.scatter(all_diffs_extrap_e[:, 0], all_diffs_extrap_e[:, 1], marker='x', color='r')
    plt.savefig(os.path.join(base_fig_dir, 'deltas_quad'+str(quad)+'.png'))
    # pdb.set_trace()


run_eval_residual_pergoal_bestworst(quad=1)
run_eval_residual_pergoal_bestworst(quad=2)
run_eval_residual_pergoal_bestworst(quad=3)
run_eval_residual_pergoal_bestworst(quad=4)
