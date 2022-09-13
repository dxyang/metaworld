import math
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import time
import datetime
plt.ion()   # interactive mode - o
from io import open
import unicodedata
import string
import re
import random
from torch import optim
from sklearn.neighbors import NearestNeighbors
import argparse
import pdb
import functools
from gym.spaces import Box
import gym
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import test_cases_latest_nonoise
from render_utils import trajectory2vid
from ood_similar_shifts.networks import *
import json
from matplotlib import cm
import sys
sys.path.append('/data/pulkitag/misc/avivn/franka_reach_neurips/adept_envs/adept_envs/franka')
sys.path.append('/data/pulkitag/misc/avivn/franka_reach_neurips/adept_envs/')
from franka_reach import FrankaReachV1
from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import pickle


parser = argparse.ArgumentParser(description='task and ood config')
parser.add_argument('--env-name', type=str, default='FrankaReachV1')
parser.add_argument('--log-path', type=str, default='bilinear')
parser.add_argument('--expert-path', type=str, default='expert_data.pkl')
parser.add_argument('--expert-policy-filename', type=str, default='/data/pulkitag/misc/avivn/franka_reach_neurips/itr_2990.pkl')
parser.add_argument('--goal-ood', default=False, action='store_true')
parser.add_argument('--obj-ood', default=False, action='store_true')
parser.add_argument('--random-hand-init', default=False, action='store_true')
parser.add_argument('--vae', default=False, action='store_true')
parser.add_argument('--expert-break-on-succ', default=False, action='store_true') #if use flag expert demos will stop on success and not max_length. default is to run until max_steps
parser.add_argument('--res-timestep', default=False, action='store_true')
parser.add_argument('--reduced-state-space', default=False)
parser.add_argument('--fourier', default=False, action='store_true')
parser.add_argument('--render', default=False, action='store_true', help='if True will render expert and bc')
parser.add_argument('--debug', default=False, action='store_true') #smaller number or epochs and trajs
parser.add_argument('--quad1low', default=[0., 0.6, 2.])
parser.add_argument('--quad1high', default=[0.5, 0.8, 2.5])
parser.add_argument('--quad2low', default=[-0.5, 0.6, 2.])
parser.add_argument('--quad2high', default=[0., 0.8, 2.5])
parser.add_argument('--quad3low', default=[-0.5, 0.5, 2.])
parser.add_argument('--quad3high', default=[0., 0.6, 2.5])
parser.add_argument('--quad4low', default=[0., 0.5, 2.])
parser.add_argument('--quad4high', default=[0.5, 0.6, 2.5])
parser.add_argument('--horizon', default=200)
parser.add_argument('--expert-n-traj', type=int, default=1000) #1000
parser.add_argument('--eval-n-traj', type=int, default=100)
parser.add_argument('--bc-n-epochs', type=int, default=5000) #5000
parser.add_argument('--bc-hidden-layer-size', type=int, default=32) #1000
parser.add_argument('--bc-hidden-depth', type=int, default=1) #3
parser.add_argument('--bc-batch-size', type=int, default=32)
parser.add_argument('--deep-sets-n-epochs', type=int, default=5000) #5000
parser.add_argument('--deep-sets-hidden-layer-size', type=int, default=32) #1000
parser.add_argument('--deep-sets-hidden-depth', type=int, default=1) #3
parser.add_argument('--deep-sets-batch-size', type=int, default=32)
parser.add_argument('--NN-n-epochs', type=int, default=5000)
parser.add_argument('--NN-hidden-layer-size', type=int, default=32) #1000
parser.add_argument('--NN-hidden-depth', type=int, default=1) #3
parser.add_argument('--NN-batch-size', type=int, default=32)
parser.add_argument('--bilinear-n-epochs', type=int, default=5000)
parser.add_argument('--bilinear-hidden-layer-size', type=int, default=32)
parser.add_argument('--bilinear-hidden-depth', type=int, default=1)
parser.add_argument('--bilinear-batch-size', type=int, default=32)


train_args = parser.parse_args()
train_args_json = vars(train_args)
runtype = 'bc_{}_{}_deep_{}_{}_nn_{}_{}_bilinear_{}_{}_fourier_{}'.format(train_args.bc_hidden_depth, 
                                                                          train_args.bc_hidden_layer_size,
                                                                          train_args.deep_sets_hidden_depth, 
                                                                          train_args.deep_sets_hidden_layer_size,
                                                                          train_args.NN_hidden_depth, 
                                                                          train_args.NN_hidden_layer_size,
                                                                          train_args.bilinear_hidden_depth, 
                                                                          train_args.bilinear_hidden_layer_size,
                                                                          int(train_args.fourier)
                                                                          )
log_path = os.path.join(train_args.log_path, train_args.env_name, runtype, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
expert_data_path = os.path.join(train_args.log_path, train_args.env_name, train_args.expert_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)
with open(os.path.join(log_path, 'train_args.txt'), 'a') as f:
    json.dump(train_args_json, f, indent=2)   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_gpu_mode(True)
env = FrankaReachV1()
env.sample_goal = False
env.sample_state = False

obs_size = env.observation_space.shape[0]
ac_size = env.action_space.shape[0]
goal_size = len(env.goal)
EPS = 1e-9  


def sample_data(quad):
    if quad == 1:
        goal_pos = np.random.uniform(low=train_args.quad1low, high=train_args.quad1high, size=(goal_size,))
        hand_init = np.random.uniform(low=train_args.quad1low, high=train_args.quad1high, size=(goal_size,))
    elif quad == 2:                                    
        goal_pos = np.random.uniform(low=train_args.quad2low, high=train_args.quad2high, size=(goal_size,)) 
        hand_init = np.random.uniform(low=train_args.quad2low, high=train_args.quad2high, size=(goal_size,)) 
    elif quad == 3:
        goal_pos = np.random.uniform(low=train_args.quad3low, high=train_args.quad3high, size=(goal_size,))  
        hand_init = np.random.uniform(low=train_args.quad3low, high=train_args.quad3high, size=(goal_size,))  
    elif quad == 4:
        goal_pos = np.random.uniform(low=train_args.quad4low, high=train_args.quad4high, size=(goal_size,))
        hand_init = np.random.uniform(low=train_args.quad4low, high=train_args.quad4high, size=(goal_size,))
    if not train_args.random_hand_init:
        hand_init = None     
    return hand_init, goal_pos


def collect_data(env, policy, num_trajs=None, device=None, render=False):
    """Collect expert rollouts from the environment, store it in some datasets"""
    trajs = []
    goal_poses = []
    for tn in range(num_trajs):          
        hand_init, goal_pos = sample_data(1)
        env.set_goal(goal_pos)
        traj = rollout(env, policy, max_path_length=train_args.horizon)
        trajs.append(traj)
        goal_poses.append(goal_pos)
    with open(expert_data_path, 'wb') as f:
        pickle.dump(trajs, f)     
    plot_traj(trajs, os.path.join(train_args.log_path, train_args.env_name, 'expert.png'))
    plot_act(trajs, os.path.join(train_args.log_path, train_args.env_name, 'expert_acts.png'))      
    return trajs, goal_poses


def plot_traj(trajs, tag, nplot=20):
    colors = sns.color_palette("hls", nplot)
    plot_idx = random.sample(range(len(trajs)), k=nplot) #random
    fig = plt.figure()  
    ax = plt.axes(projection='3d')    
    end_dists = []    
    for colorid, idx in enumerate(plot_idx):
        t = np.array(trajs[idx]['observations'])        
        end_effector = np.array([traj_info['end_effector'] for traj_info in trajs[idx]['env_infos']])
        ax.plot3D(end_effector[:,0], end_effector[:,1], end_effector[:,2], color=colors[colorid], linestyle=':') #end effector traj
        ax.scatter3D(t[-1,-3], t[-1,-2], t[-1,-1], color=colors[colorid], marker='x') #gt goal    
        end_dists.append(np.linalg.norm(end_effector[-1][:3] - t[-1][-3:])) #dist between end effector and gt goal
    end_dists = np.array(end_dists)
    # plt.title(round(np.mean(end_dists),4))   
    all_returns = [np.sum(traj['rewards']) for traj in trajs]
    plt.title(str(round(np.mean(all_returns),4)) + ' $\pm$ ' + str(round(np.std(all_returns),4)))
    plt.savefig(tag)           


def plot_act(trajs, tag):
    #plot expert acts
    fig, ax = plt.subplots(ac_size, 1)
    for traj in trajs:
        for idx in range(ac_size):
            ax[idx].plot(traj['actions'][:, idx])
    plt.savefig(tag)


def get_latent(x):
    if train_args.vae:
        raise NotImplementedError("vae not implemented")
    else:
        return x    


def process_o(o):
    if train_args.reduced_state_space:
        o_proc = env.reduced_obs(o.copy())
    else:
        o_proc = o.copy()
    return o_proc        


def sample_delta(train_goals, train_deltas, curr_obs):
    """return train traj idx that gives a delta closest to train deltas 
        curr_obs - s,g. s_0 cancels out"""
    #reshape batches of obs deltas
    train_deltas = train_deltas.reshape(-1, obs_size)[:,-goal_size:] 
    #dist from curr g to every g in train
    curr_deltas = np.array(curr_obs[-goal_size:]) - train_goals 
    #for each curr_delta, take min over distance to all train deltas
    delta_dists = [np.min(np.linalg.norm(train_d - train_deltas, axis=1)) for train_d in curr_deltas]
    #sample from deltas in dist
    closest_train_idx = random.choice(np.argsort(delta_dists)[:train_args.expert_n_traj//10]) 
    # print(np.min(np.linalg.norm(curr_deltas, axis=1)), np.linalg.norm(curr_deltas[np.argsort(delta_dists)[train_args.expert_n_traj//10]]))
    return closest_train_idx      


def train_bc(trajs):
    bc_policy = Policy(obs_size, ac_size, train_args.bc_hidden_layer_size, train_args.bc_hidden_depth, train_args.fourier) 
    bc_policy.to(device)    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(bc_policy.parameters()))
    losses = []
    idxs = np.array(range(len(trajs)))
    num_batches = len(idxs) // train_args.bc_batch_size
    for epoch in range(train_args.bc_n_epochs):
        np.random.shuffle(idxs)
        running_loss = 0.0
        for i in range(num_batches):        
            optimizer.zero_grad()
            t1_idx = np.random.randint(len(trajs), size=(train_args.bc_batch_size,)) # Indices of first trajectory
            t1_idx_pertraj = np.random.randint(train_args.horizon, size=(train_args.bc_batch_size,))
            t1_states = np.concatenate([trajs[c_idx]['observations'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
            t1_actions = np.concatenate([trajs[c_idx]['actions'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
            t1_states = torch.Tensor(t1_states).float().to(device)
            t1_actions = torch.Tensor(t1_actions).float().to(device)        
            a1_pred = bc_policy(t1_states.to(device)) 
            loss = torch.mean(torch.linalg.norm(a1_pred - t1_actions, dim=-1))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:    # print every 2000 mini-batches
                # print('[%d, %5d] loss: %.8f' %
                #       (epoch + 1, i + 1, running_loss / 10.))
                losses.append(running_loss/10.)
                running_loss = 0.0
            losses.append(loss.item())
    fig = plt.figure()
    plt.plot(losses)
    plt.savefig(os.path.join(log_path,'bc_losses.png'))
    torch.save(bc_policy, os.path.join(log_path,'bc_policy.pt'))
    return bc_policy


def eval_bc(policy, quad):
    transformed_trajs = []
    goal_poses_test = []
    for tn in range(train_args.eval_n_traj):
        env.reset()
        hand_init, goal_pos = sample_data(quad)
        o = env.set_goal(goal_pos)        
        o = process_o(o)
        traj = {'observations': [],'actions': [], 'next_observations': [], 'dones': [], 'rewards': [], 'env_infos': []}
        for _ in range(train_args.horizon):
            if hasattr(policy,'get_action'):
                ac = policy.get_action(o)
            else:
                t1s = torch.Tensor(o[None]).to(device)
                ac = policy(t1s).cpu().detach().numpy()[0]
            no, r, done, info = env.step(ac)
            no = process_o(no)
            traj['observations'].append(o.copy())
            traj['actions'].append(ac.copy())
            traj['next_observations'].append(no.copy())
            traj['dones'].append(done)
            traj['rewards'].append(r)
            traj['env_infos'].append(info)
            o = no
        traj['observations'] = np.array(traj['observations'])
        traj['actions'] = np.array(traj['actions'])
        traj['next_observations'] = np.array(traj['next_observations'])
        traj['dones'] = np.array(traj['dones'])
        traj['rewards'] = np.array(traj['rewards'])
        transformed_trajs.append(traj)
        goal_poses_test.append(goal_pos)     
    with open(os.path.join(log_path, 'test_bc'+str(quad)+'.pkl'), 'wb') as f:
        pickle.dump(transformed_trajs, f)   
    plot_traj(transformed_trajs, os.path.join(log_path, 'test_bc'+str(quad)+'.png'))            
    return transformed_trajs


def train_deep_sets(trajs):
    deep_sets_policy = DeepSets(obs_size-goal_size, goal_size, ac_size, \
                                train_args.deep_sets_hidden_layer_size, \
                                train_args.deep_sets_hidden_depth, train_args.fourier) 
    deep_sets_policy.to(device)    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(deep_sets_policy.parameters()))
    losses = []
    idxs = np.array(range(len(trajs)))
    num_batches = len(idxs) // train_args.deep_sets_batch_size
    for epoch in range(train_args.deep_sets_n_epochs):
        np.random.shuffle(idxs)
        running_loss = 0.0
        for i in range(num_batches):        
            optimizer.zero_grad()
            t1_idx = np.random.randint(len(trajs), size=(train_args.deep_sets_batch_size,)) # Indices of first trajectory
            t1_idx_pertraj = np.random.randint(train_args.horizon, size=(train_args.deep_sets_batch_size,))
            t1_states = np.concatenate([trajs[c_idx]['observations'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
            t1_actions = np.concatenate([trajs[c_idx]['actions'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
            t1_states = torch.Tensor(t1_states).float().to(device)
            t1_actions = torch.Tensor(t1_actions).float().to(device)   
            a1_pred = deep_sets_policy(t1_states[:,:obs_size-goal_size].to(device), t1_states[:,-goal_size:].to(device)) 
            loss = torch.mean(torch.linalg.norm(a1_pred - t1_actions, dim=-1))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:    # print every 2000 mini-batches
                # print('[%d, %5d] loss: %.8f' %
                #       (epoch + 1, i + 1, running_loss / 10.))
                losses.append(running_loss/10.)
                running_loss = 0.0
            losses.append(loss.item())
    fig = plt.figure()
    plt.plot(losses)
    plt.savefig(os.path.join(log_path,'deep_sets_losses.png'))
    torch.save(deep_sets_policy, os.path.join(log_path,'deep_sets_policy.pt'))    
    return deep_sets_policy


def eval_deep_sets(quad, deep_sets_policy):
    transformed_trajs = []
    goal_poses_test = []
    for tn in range(train_args.eval_n_traj):
        env.reset()
        hand_init, goal_pos = sample_data(quad)
        o = env.set_goal(goal_pos)      
        o = process_o(o)
        traj = {'observations': [],'actions': [], 'next_observations': [], 'dones': [], 'rewards': [], 'env_infos': []}
        for _ in range(train_args.horizon):
            if hasattr(deep_sets_policy,'get_action'):
                ac = deep_sets_policy.get_action(o)
            else:
                t1s = torch.Tensor(o[None]).to(device)
                ac = deep_sets_policy(t1s[:,:obs_size-goal_size], t1s[:,-goal_size:]).cpu().detach().numpy()[0]
            no, r, done, info = env.step(ac)
            no = process_o(no)
            traj['observations'].append(o.copy())
            traj['actions'].append(ac.copy())
            traj['next_observations'].append(no.copy())
            traj['dones'].append(done)
            # traj['reward'].append(info['in_place_reward'])
            traj['rewards'].append(r)
            traj['env_infos'].append(info)
            o = no
        traj['observations'] = np.array(traj['observations'])
        traj['actions'] = np.array(traj['actions'])
        traj['next_observations'] = np.array(traj['next_observations'])
        traj['dones'] = np.array(traj['dones'])
        traj['rewards'] = np.array(traj['rewards'])        
        # traj['tcp'] = env.tcp_center #gripper at end of trajectory TODO not good indicator for all tasks
        transformed_trajs.append(traj)
        goal_poses_test.append(goal_pos)  
    with open(os.path.join(log_path, 'test_deep_sets'+str(quad)+'.pkl'), 'wb') as f:
        pickle.dump(transformed_trajs, f)   
    plot_traj(transformed_trajs, os.path.join(log_path, 'test_deep_sets'+str(quad)+'.png'))            
    return transformed_trajs


def plot_deltas():
    #TODO visualize seen (s,g)-(ds,dg) combs
    all_diffs_e = []
    for i in range(100):
        for j in range(100):        
            diff = get_latent(trajs[i]['observations'][-1, -3:])[None] - get_latent(trajs[j]['observations'][-1, -3:])[None]
            all_diffs_e.append(diff.copy())
    all_diffs_e = np.concatenate(all_diffs_e)


def train_NN(trajs):
    #all to all training 
    nn_policy = Policy(obs_size*2, ac_size, train_args.NN_hidden_layer_size, train_args.NN_hidden_depth, \
                        train_args.fourier) 
    nn_policy.to(device)
    optimizer = optim.Adam(list(nn_policy.parameters()))
    losses = []
    nn_deltas = []
    idxs = np.array(range(len(trajs)))
    num_batches = len(idxs) // train_args.NN_batch_size
    # Train the model with regular SGD
    for epoch in range(train_args.NN_n_epochs):  # loop over the dataset multiple times
        np.random.shuffle(idxs)
        running_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()
            t1_idx = np.random.randint(len(trajs), size=(train_args.NN_batch_size,)) # Indices of first trajectory
            t2_idx = np.random.randint(len(trajs), size=(train_args.NN_batch_size,)) # Indices of second trajectory
            t1_idx_pertraj = np.random.randint(train_args.horizon, size=(train_args.NN_batch_size,))
            t1_states = np.concatenate([trajs[c_idx]['observations'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
            t1_actions = np.concatenate([trajs[c_idx]['actions'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
            t1_states = torch.Tensor(t1_states).float().to(device)
            t1_actions = torch.Tensor(t1_actions).float().to(device)
            t2_states = np.concatenate([trajs[c_idx]['observations'][t_idx][None] for (c_idx, t_idx) in zip(t2_idx, t1_idx_pertraj)])
            t2_actions = np.concatenate([trajs[c_idx]['actions'][t_idx][None] for (c_idx, t_idx) in zip(t2_idx, t1_idx_pertraj)])
            t2_states = torch.Tensor(t2_states).float().to(device)
            t2_actions = torch.Tensor(t2_actions).float().to(device)
            deltas = np.concatenate([get_latent(trajs[t2_idx_diff]['observations'][pertraj])[None] - 
                                    get_latent(trajs[t1_idx_diff]['observations'][pertraj])[None]
                                            for (t1_idx_diff, t2_idx_diff, pertraj) in zip(t1_idx, t2_idx, t1_idx_pertraj)])
            nn_deltas.append(deltas)
            deltas = torch.Tensor(deltas).float().to(device)
            nn_input = torch.cat([t1_states, deltas], dim=-1)
            a2_pred_nn = nn_policy(nn_input) #input: [s,g,ds,dg]
            # L2 regression on actions
            loss = torch.mean(torch.linalg.norm(a2_pred_nn - t2_actions, dim=-1))
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
    fig = plt.figure()
    plt.plot(losses)
    plt.savefig(os.path.join(log_path,'nn_losses.png'))
    torch.save(nn_policy, os.path.join(log_path,'nn_policy.pt')) 
    return nn_policy, np.array(nn_deltas)   


def eval_nn_uniform(quad, nn_policy, trajs, train_deltas):
    # Check residuals on OOD points
    colors = sns.color_palette("hls", train_args.eval_n_traj)
    train_goals = np.concatenate([t['observations'][-1][-3:][None].copy() for t in trajs])
    fig = plt.figure()
    ax = plt.axes(projection='3d')         
    end_dists = []
    all_diffs_extrap_e = []
    transformed_trajs = []
    for g_idx in range(train_args.eval_n_traj):
        hand_init, goal_pos = sample_data(quad)
        end_pos = goal_pos.copy()    
        o = env.reset()
        o = env.set_goal(goal_pos)
        o = process_o(o)
        closest_point_idx = sample_delta(train_goals, train_deltas, o)
        closest_point = train_goals[closest_point_idx].copy()
        closest_traj_obs = trajs[closest_point_idx]['observations'].copy()
        g_diff = end_pos - closest_point
        g_diff_t = torch.Tensor(g_diff[None]).to(device)
        traj = {'observations': [],'actions': [], 'next_observations': [], 'rewards': [], 'env_infos': []}
        for i in range(train_args.horizon):
            t1s = torch.Tensor(closest_traj_obs[i][None]).to(device)
            deltas = torch.Tensor(o[None]).to(device) - t1s     
            nn_input = torch.cat([t1s, deltas], dim=-1)          
            ac_final = nn_policy(nn_input).cpu().detach().numpy()[0]
            no, r, d, info = env.step(ac_final)
            no = process_o(no).copy()
            traj['observations'].append(o.copy())
            traj['actions'].append(ac_final.copy())
            traj['next_observations'].append(no.copy())
            traj['rewards'].append(r)
            traj['env_infos'].append(info)
            o = no.copy()
        transformed_trajs.append(traj)
        #plot
        traj_end_effector = np.array([traj_info['end_effector'] for traj_info in traj['env_infos']])
        ax.plot3D(traj_end_effector[:, 0], traj_end_effector[:, 1], traj_end_effector[:, 2], color=colors[g_idx])        
        ax.scatter3D([end_pos[0]], [end_pos[1]], [end_pos[2]], color=colors[g_idx], marker='x', s=50)
        ax.scatter3D([closest_traj_obs[-1][-3]], [closest_traj_obs[-1][-2]], [closest_traj_obs[-1][-1]], color=colors[g_idx], marker='o', s=50)            
        end_dists.append(np.linalg.norm(traj_end_effector[-1] - end_pos.copy()))  
    end_dists = np.array(end_dists)
    all_diffs_extrap_e = np.array(all_diffs_extrap_e)
    print("Average end distance is %.5f"%(np.mean(end_dists)))
    # plt.title(round(np.mean(end_dists),4))  
    all_returns = [np.sum(traj['rewards']) for traj in transformed_trajs]
    plt.title(str(round(np.mean(all_returns),4)) + ' $\pm$ ' + str(round(np.std(all_returns),4)))
    plt.savefig(os.path.join(log_path, 'test_nn_uniform'+str(quad)+'.png'))
    with open(os.path.join(log_path, 'test_nn_uniform'+str(quad)+'.pkl'), 'wb') as f:
        pickle.dump(transformed_trajs, f) 


def train_bilinear(trajs):
    #all to all training 
    bilinear_policy = BilinearPolicy(obs_size, ac_size, train_args.bilinear_hidden_layer_size, \
                                        train_args.bilinear_hidden_depth, train_args.fourier) 
    bilinear_policy.to(device)
    optimizer = optim.Adam(list(bilinear_policy.parameters()))
    losses = []
    bilinear_deltas = []
    idxs = np.array(range(len(trajs)))
    num_batches = len(idxs) // train_args.bilinear_batch_size
    # Train the model with regular SGD
    for epoch in range(train_args.bilinear_n_epochs):  # loop over the dataset multiple times
        np.random.shuffle(idxs)
        running_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()
            t1_idx = np.random.randint(len(trajs), size=(train_args.bilinear_batch_size,)) # Indices of first trajectory
            t2_idx = np.random.randint(len(trajs), size=(train_args.bilinear_batch_size,)) # Indices of second trajectory
            t1_idx_pertraj = np.random.randint(train_args.horizon, size=(train_args.bilinear_batch_size,))
            t1_states = np.concatenate([trajs[c_idx]['observations'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
            t1_actions = np.concatenate([trajs[c_idx]['actions'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
            t1_states = torch.Tensor(t1_states).float().to(device)
            t1_actions = torch.Tensor(t1_actions).float().to(device)
            t2_states = np.concatenate([trajs[c_idx]['observations'][t_idx][None] for (c_idx, t_idx) in zip(t2_idx, t1_idx_pertraj)])
            t2_actions = np.concatenate([trajs[c_idx]['actions'][t_idx][None] for (c_idx, t_idx) in zip(t2_idx, t1_idx_pertraj)])
            t2_states = torch.Tensor(t2_states).float().to(device)
            t2_actions = torch.Tensor(t2_actions).float().to(device)
            deltas = np.concatenate([get_latent(trajs[t2_idx_diff]['observations'][pertraj])[None] - 
                                    get_latent(trajs[t1_idx_diff]['observations'][pertraj])[None]
                                            for (t1_idx_diff, t2_idx_diff, pertraj) in zip(t1_idx, t2_idx, t1_idx_pertraj)])
            bilinear_deltas.append(deltas)
            deltas = torch.Tensor(deltas).float().to(device)
            a2_pred_bilinear = bilinear_policy(t1_states, deltas) #input: [s,g],[ds,dg]
            # L2 regression on actions
            loss = torch.mean(torch.linalg.norm(a2_pred_bilinear - t2_actions, dim=-1))
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
    fig = plt.figure()
    plt.plot(losses)
    plt.savefig(os.path.join(log_path,'bilinear_losses.png')) 
    torch.save(bilinear_policy, os.path.join(log_path,'bilinear_policy.pt'))     
    return bilinear_policy, np.array(bilinear_deltas)      


def eval_bilinear_uniform(quad, bilinear_policy, trajs, train_deltas):
    # Check residuals on OOD points
    colors = sns.color_palette("hls", train_args.eval_n_traj)
    train_goals = np.concatenate([t['observations'][-1][-3:][None].copy() for t in trajs])
    fig = plt.figure()
    ax = plt.axes(projection='3d')         
    end_dists = []
    transformed_trajs = []    
    all_diffs_extrap_e = []
    for g_idx in range(train_args.eval_n_traj):
        hand_init, goal_pos = sample_data(quad)
        end_pos = goal_pos.copy()      
        o = env.reset()
        o = env.set_goal(goal_pos)
        o = process_o(o)
        closest_point_idx = sample_delta(train_goals, train_deltas, o)
        closest_point = train_goals[closest_point_idx].copy()
        closest_traj_obs = trajs[closest_point_idx]['observations'].copy()
        g_diff = end_pos - closest_point
        g_diff_t = torch.Tensor(g_diff[None]).to(device)
        traj = {'observations': [],'actions': [], 'next_observations': [], 'rewards': [], 'env_infos': []}
        for i in range(train_args.horizon):
            t1s = torch.Tensor(closest_traj_obs[i][None]).to(device)
            deltas = torch.Tensor(o[None]).to(device) - t1s               
            ac_final = bilinear_policy(t1s, deltas).cpu().detach().numpy()[0]
            no, r, d, info = env.step(ac_final)
            no = process_o(no).copy()
            traj['observations'].append(o.copy())
            traj['actions'].append(ac_final.copy())
            traj['next_observations'].append(no.copy())
            traj['rewards'].append(r)
            traj['env_infos'].append(info)
            o = no.copy()
        transformed_trajs.append(traj)
        #plot
        traj_end_effector = np.array([traj_info['end_effector'] for traj_info in traj['env_infos']])
        ax.plot3D(traj_end_effector[:, 0], traj_end_effector[:, 1], traj_end_effector[:, 2], color=colors[g_idx])        
        ax.scatter3D([end_pos[0]], [end_pos[1]], [end_pos[2]], color=colors[g_idx], marker='x', s=50)
        ax.scatter3D([closest_traj_obs[-1][-3]], [closest_traj_obs[-1][-2]], [closest_traj_obs[-1][-1]], color=colors[g_idx], marker='o', s=50)            
        end_dists.append(np.linalg.norm(traj_end_effector[-1] - end_pos.copy())) 
    end_dists = np.array(end_dists)
    all_diffs_extrap_e = np.array(all_diffs_extrap_e)
    print("Average end distance is %.5f"%(np.mean(end_dists)))
    # plt.title(round(np.mean(end_dists),4))      
    all_returns = [np.sum(traj['rewards']) for traj in transformed_trajs]
    plt.title(str(round(np.mean(all_returns),4)) + ' $\pm$ ' + str(round(np.std(all_returns),4)))
    plt.savefig(os.path.join(log_path, 'test_bilinear_uniform'+str(quad)+'.png')) 
    with open(os.path.join(log_path, 'test_bilinear_uniform'+str(quad)+'.pkl'), 'wb') as f:
        pickle.dump(transformed_trajs, f)


# Data collection
print('EXPERT')
#for each task, create 1000 expert demos
if not os.path.exists(expert_data_path):
    data = torch.load(train_args.expert_policy_filename)
    expert_policy = data['evaluation/policy']
    expert_policy.cuda()       
    trajs, traj_goal_poses = collect_data(env, expert_policy, num_trajs=train_args.expert_n_traj, device=device)
else:
    with open(expert_data_path, "rb") as input_file:
        trajs = pickle.load(input_file)  

# #BC
# print('BC')  
# #TRAIN BC  
# bc_policy = train_bc(trajs)
# #TEST BC - pi(s,g)=a
# for quad in [1,2,3]:
#     eval_bc(bc_policy, quad)

# #DEEP SETS - h(f(s)+g(g))=a'
# print('DEEP SETS')
# deep_sets_policy = train_deep_sets(trajs)
# for quad in [1,2,3]:
#     eval_deep_sets(quad, deep_sets_policy)

#NN - f(s,g,ds,dg)=a'
print('NN')
nn_policy, nn_deltas = train_NN(trajs)
for quad in [1,2,3]:
    eval_nn_uniform(quad, nn_policy, trajs, nn_deltas)

#BILINEAR - f(s,g)g(ds,dg)=a'
print('BILINEAR')
#train - all to all 
#TODO timestep?
bilinear_policy, bilinear_deltas = train_bilinear(trajs)
for quad in [1,2,3]:
    eval_bilinear_uniform(quad, bilinear_policy, trajs, bilinear_deltas)




