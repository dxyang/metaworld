import math
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn
import numpy as np
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
import copy
from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import time
plt.ion()   # interactive mode - o
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# Import and define the environment
from metameta_envs.wheeled_robot import WheeledEnv

# Load policy for wheeled robot from RL
from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import torch
import pickle

# filename = '/scratch/abhishek_sandbox/rlkit/data/rlkit-wheeled/rlkit-wheeled_2022_02_02_23_28_06_0000--s-0/save_good_itr.pkl'
filename = '/scratch/abhishek_sandbox/rlkit/data/rlkit-wheeled/rlkit-wheeled_2022_02_04_00_13_08_0000--s-0/itr_1000.pkl'
data = torch.load(filename)
policy = data['evaluation/policy']
# env = data['evaluation/env']
env1 = WheeledEnv(sample_goal_during_reset=False)
set_gpu_mode(True)
policy.cuda()
trajs = []
angles = []
num_trajs = 1000
horizon = 200
num_collected = 0
while num_collected < num_trajs:
    print("Taking rollout number %d"%num_collected)
#     radius = 2.0 
#     angle = np.random.uniform(0, np.pi/2)
    xpos = np.random.uniform(-2., 2.) #radius*np.cos(angle)
    ypos = np.random.uniform(-2., 2.) #radius*np.sin(angle)
    env1.set_goal(np.array([xpos, ypos]))
    traj = rollout(
        env1,
        policy,
        max_path_length=horizon,
    )
    trajs.append(traj)
    angle_curr = np.arctan2(traj['observations'][-1, -2], traj['observations'][-1, -1])
    angles.append(angle_curr)
    num_collected += 1

from matplotlib import cm

plt.xlim(-0.5, 2.5)
plt.ylim(-0.5, 2.5)
for traj in trajs:
    angle_curr = np.arctan2(traj['observations'][-1, -2], traj['observations'][-1, -1])
    plt.plot(traj['observations'][:, 0], traj['observations'][:, 1], c=cm.viridis(angle_curr))



# Test model generalization
trajs_gen = []
angles_gen = []
num_trajs = 200
num_collected = 0
while num_collected < num_trajs:
    print("Taking rollout number %d"%num_collected)
    radius = 2.0 
    angle = np.random.uniform(0, 2*np.pi)
    xpos = radius*np.cos(angle)
    ypos = radius*np.sin(angle)
    env1.set_goal(np.array([xpos, ypos]))
    traj = rollout(
        env1,
        policy,
        max_path_length=horizon,
    )
    trajs_gen.append(traj)
    angle_curr = np.arctan2(traj['observations'][-1, -2], traj['observations'][-1, -1])
    angles_gen.append(angle_curr.copy())
    num_collected += 1

from matplotlib import cm

for traj in trajs_gen:
    angle_curr = np.arctan2(traj['observations'][-1, -2], traj['observations'][-1, -1])
    plt.plot(traj['observations'][:, 0], traj['observations'][:, 1], c=cm.viridis(angle_curr/2))   


def get_latent(x):
    if vae_mode:
        raise NotImplementedError('not implemented vae yet')
    else:
        return x

# Define the forward model
class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.trunk = mlp(obs_dim, hidden_dim, action_dim, hidden_depth)
        self.outputs = dict()
        self.apply(weight_init)
    
    def reset(self):
        pass
    
    def forward(self, obs):
        next_pred = self.trunk(obs)
        return next_pred
    
    def get_action(self, obs_np, ):
        actions = self.trunk(torch.Tensor(obs_np)[None].to(device)).cpu().detach().numpy()
        return actions[0, :], {}

hidden_layer_size = 1000
hidden_depth = 3
horizon = 200
obs_size = env1.observation_space.shape[0]
ac_size = env1.action_space.shape[0]
policy = Policy(obs_size, ac_size, hidden_layer_size, hidden_depth) # 10 dimensional latent
num_tasks = len(trajs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy.to(device)
mode = 'concat'
vae_mode = False

# Train standard goal conditioned policy
num_epochs = 500
batch_size = 50
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
        t1_states = np.concatenate([trajs[c_idx]['observations'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
        t1_actions = np.concatenate([trajs[c_idx]['actions'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
   
        t1_states = torch.Tensor(t1_states).float().to(device)
        t1_actions = torch.Tensor(t1_actions).float().to(device)
        
        a1_pred = policy(t1_states.to(device)) #first action prediction
        
        
        loss = torch.mean(torch.linalg.norm(a1_pred - t1_actions, dim=-1))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.8f' %
                  (epoch + 1, i + 1, running_loss / 10.))
            losses.append(running_loss/10.)
            running_loss = 0.0
        losses.append(loss.item())

print('Finished Training')
plt.plot(losses)

# Test model generalization

num_trajs = 100
colors = sns.color_palette("hls", num_trajs)
plt.xlim(-3.0, 3.0)
plt.ylim(-3.0, 3.0)
for j in range(num_trajs):
    print("Taking rollout number %d"%j)
    radius = 2.0 
    angle = np.random.uniform(0, 2*np.pi)
    xpos = radius*np.cos(angle)
    ypos = radius*np.sin(angle)
    env1.set_goal(np.array([xpos, ypos]))
    traj = rollout(
        env1,
        policy,
        max_path_length=horizon,
    )
    plt.scatter(xpos, ypos, marker='x', s=20, color=colors[j])
    plt.plot(traj['observations'][:, 0], traj['observations'][:, 1], color=colors[j])

# Define the forward model for nonlinear hypernet
class TransformPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.trunk = mlp(obs_dim, hidden_dim, action_dim, hidden_depth)
        self.outputs = dict()
    
    # Going forward with passed in parameters
    def forward_parameters(self, in_val, parameters=None):
        if parameters is None: 
            parameters = list(self.parameters())
        
        output = in_val    
        for params_idx in range(0, len(parameters) - 2, 2):
            w = parameters[params_idx]
            b = parameters[params_idx + 1]
            output = F.linear(output, w, b)
            output = F.relu(output)
        w = parameters[-2]
        b = parameters[-1]
        output = F.linear(output, w, b)
        return output

residual_policy = Policy(4, 2, 128, 3) # 10 dimensional latent
residual_policy.to(device)

num_epochs = 1000
batch_size = 100
EPS = 1e-9
    
criterion = nn.MSELoss()
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
        t1_states = np.concatenate([trajs[c_idx]['observations'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
        t1_actions = np.concatenate([trajs[c_idx]['actions'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
   
        t1_states = torch.Tensor(t1_states).float().to(device)
        t1_actions = torch.Tensor(t1_actions).float().to(device)
        
        a1_pred = policy(t1_states.to(device)) #first action prediction
        
        t2_states = np.concatenate([trajs[c_idx]['observations'][t_idx][None] 
                                    for (c_idx, t_idx) in zip(t2_idx, t1_idx_pertraj)])
        t2_actions = np.concatenate([trajs[c_idx]['actions'][t_idx][None] 
                                     for (c_idx, t_idx) in zip(t2_idx, t1_idx_pertraj)])
   
        t2_states = torch.Tensor(t2_states).float().to(device)
        t2_actions = torch.Tensor(t2_actions).float().to(device)
        
        a2_pred = policy(t2_states.to(device)).detach() #first action prediction

        g_delta = np.concatenate([get_latent(trajs[t2_idx_diff]['observations'][-1, -2:])[None] - 
                                    get_latent(trajs[t1_idx_diff]['observations'][-1, -2:])[None]
                                     for (t1_idx_diff, t2_idx_diff) in zip(t1_idx, t2_idx)])
        g_delta_torch = torch.Tensor(g_delta).float().to(device)
        res_in = torch.cat([a1_pred.detach(), g_delta_torch], dim=-1)
        residual_pred = residual_policy(res_in)
        a2_pred_residual = residual_pred
        
        # L2 regression on actions
        loss = torch.mean(torch.linalg.norm(a2_pred_residual - a2_pred, dim=-1))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.8f' %
                  (epoch + 1, i + 1, running_loss / 10.))
            losses.append(running_loss/10.)
            running_loss = 0.0
        losses.append(loss.item())

print('Finished Training')
plt.plot(losses)

def run_eval_residual():
    # Check residuals on OOD points
    colors = sns.color_palette("hls", len(ood_goals))
    size_sample = 5
    train_goals = np.concatenate([t['observations'][-1][-2:][None].copy() for t in trajs])
    train_goals_latent = np.concatenate([get_latent(t['observations'][-1][-2:])[None] for t in trajs])
    
    end_dists = []
    for k in range(len(ood_goals)):
        o = env1.reset()

        end_pos = ood_goals[k].copy()

        #Find closest point 
        closest_point_idx = np.argmin(np.linalg.norm(train_goals_latent - get_latent(end_pos).copy(), axis=-1))
        closest_point = train_goals[closest_point_idx].copy()
        closest_traj_obs = trajs[closest_point_idx]['observations'].copy()

        env1.set_goal(closest_point.copy())
        g_diff = get_latent(end_pos) - get_latent(closest_point)

        o = env1.get_current_obs()
        print(closest_point)
        traj = {'observations': [],'actions': [], 'next_observations': []}
        for i in range(100):
            t1s = torch.Tensor(closest_traj_obs[i][None]).to(device) #torch.Tensor(o[None]).to(device)
            ac = policy(t1s).detach()

            g_diff_t = torch.Tensor(g_diff[None]).to(device)
            res_in = torch.cat([ac, g_diff_t], dim=-1)
            ac_final = residual_policy(res_in).cpu().detach().numpy()[0]

            no, r, d, _ = env1.step(ac_final)
            traj['observations'].append(o.copy())
            traj['actions'].append(ac_final.copy())
            traj['next_observations'].append(no.copy())
            o = no.copy()
        plt.plot(np.array(traj['observations'])[:, 0], np.array(traj['observations'])[:, 1], linestyle=':', color=colors[k])
        plt.scatter([end_pos[0]],[end_pos[1]], color=colors[k], marker='x', s=20)
        closest_dist_sampled = np.linalg.norm(np.array(traj['observations'])[:, :2] - end_pos.copy(), axis=-1).min()
        end_dists.append(closest_dist_sampled)

    end_dists = np.array(end_dists)
    print("Average end distance is %.5f"%(np.mean(end_dists)))
    plt.plot([-3, 3, 3, -3, -3], [3, 3,-3, -3, 3])
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.savefig('test_diff_hypernet.png')

def run_eval_original():
    # Check original policy on OOD points
    colors = sns.color_palette("hls", len(ood_goals))
    size_sample = 5
    end_dists = []
    for k in range(len(ood_goals)):
        o = env1.reset()

        end_pos = ood_goals[k].copy()
        env1.set_goal(end_pos)
        o = env1.get_current_obs()

        traj = {'observations': [],'actions': [], 'next_observations': []}
        for i in range(100):
            t1s = torch.Tensor(o[None]).to(device)
            ac = policy(t1s).cpu().detach().numpy()[0]
            no, r, d, _ = env1.step(ac)

            traj['observations'].append(o.copy())
            traj['actions'].append(ac.copy())
            traj['next_observations'].append(no.copy())
            o = no.copy()
        plt.plot(np.array(traj['observations'])[:, 0], np.array(traj['observations'])[:, 1], linestyle=':', color=colors[k])
        plt.scatter([end_pos[0]],[end_pos[1]], color=colors[k], marker='x', s=20)
        closest_dist_sampled = np.linalg.norm(np.array(traj['observations'])[:, :2] - end_pos.copy(), axis=-1).min()
        end_dists.append(closest_dist_sampled)

    end_dists = np.array(end_dists)
    print("Average end distance is %.5f"%(np.mean(end_dists)))
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.plot([-3, 3, 3, -3, -3], [3, 3,-3, -3, 3])
    

# Sample some OOD goals
ood_goals = []
for k in range(100):
    angle = np.random.uniform(0, np.pi)
    xpos = radius*np.cos(angle)
    ypos = radius*np.sin(angle)
    end_pos = np.array([xpos, ypos])
#     end_pos = np.random.uniform(0, size_sample, size=(2,)) # Random starting point
#     end_pos[0] = -end_pos[0] # Shift to a different quadrant
    ood_goals.append(end_pos.copy())
ood_goals = np.array(ood_goals)

run_eval_residual()
run_eval_original()

#more OOD points (full quadrant flip)
ood_goals = []
for k in range(10):
    end_pos = np.random.uniform(0, size_sample, size=(2,)) # Random starting point
    end_pos[0] = -end_pos[0] # Shift to a different quadrant
    end_pos[1] = -end_pos[1] # Shift to a different quadrant
    ood_goals.append(end_pos.copy())
ood_goals = np.array(ood_goals)

run_eval_residual()
run_eval_original()

all_diffs_e = []
for i in range(100):
    for j in range(100):        
        diff = get_latent(trajs[i]['obs'][-1, 2:4])[None] - get_latent(trajs[j]['obs'][-1, 2:4])[None]
        all_diffs_e.append(diff.copy())
all_diffs_e = np.concatenate(all_diffs_e)
all_diffs_s = []
for i in range(100):
    for j in range(100):
        diff = get_latent(trajs[i]['obs'][0, 0:2])[None] - get_latent(trajs[j]['obs'][0, 0:2])[None]
        all_diffs_s.append(diff.copy())
all_diffs_s = np.concatenate(all_diffs_s)

plt.scatter(all_diffs_e[:, 0], all_diffs_e[:, 1])
plt.scatter(all_diffs_s[:, 0], all_diffs_s[:, 1], marker='x', color='orange')

all_diffs_extrap_e = []
all_diffs_extrap_s = []

train_goals = np.concatenate([t['obs'][-1][2:4][None].copy() for t in trajs])
train_goals_latent = np.concatenate([get_latent(t['obs'][-1][2:4])[None] for t in trajs])

ood_goals = []
for k in range(1000):
    end_pos = np.random.uniform(0, size_sample, size=(2,)) # Random starting point
    end_pos[0] = -end_pos[0] # Shift to a different quadrant
#     end_pos[1] = -end_pos[1] # Shift to a different quadrant
    ood_goals.append(end_pos.copy())
ood_goals = np.array(ood_goals)

for k in range(1000):
    end_pos = ood_goals[k].copy()

    #Find closest point 
    closest_point_idx = np.argmin(np.linalg.norm(train_goals_latent - get_latent(end_pos).copy(), axis=-1))
    closest_point = train_goals[closest_point_idx].copy()
    closest_traj_obs = trajs[closest_point_idx]['obs'].copy()

    g_diff = get_latent(end_pos) - get_latent(closest_point)
    g_diff_s = get_latent(np.zeros((2,))) - get_latent(trajs[closest_point_idx]['obs'][0, :2])
    all_diffs_extrap_e.append(g_diff)
    all_diffs_extrap_s.append(g_diff_s)
    
all_diffs_extrap_e = np.array(all_diffs_extrap_e)
all_diffs_extrap_s = np.array(all_diffs_extrap_s)

plt.scatter(all_diffs_e[:, 0], all_diffs_e[:, 1])
plt.scatter(all_diffs_extrap_e[:, 0], all_diffs_extrap_e[:, 1], color='r')
plt.scatter(all_diffs_extrap_s[:, 0], all_diffs_extrap_s[:, 1], color='g')