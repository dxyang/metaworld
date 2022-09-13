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
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import test_cases_latest_nonoise
from render_utils import trajectory2vid
from ood_similar_shifts.networks import *
import json
from matplotlib import cm
from metaworld.envs.mujoco.wheeled_robot import WheeledEnv
import metaworld.envs.mujoco.wheeled_robot 
from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import pickle


parser = argparse.ArgumentParser(description='task and ood config')
parser.add_argument('--env-name', type=str, default='WheeledEnv')
parser.add_argument('--expert-policy-filename', type=str, default='/data/pulkitag/misc/avivn/metaworld/metaworld/envs/mujoco/wheeled_itr_1000.pkl')
parser.add_argument('--goal-ood', default=False, action='store_true')
parser.add_argument('--obj-ood', default=False, action='store_true')
parser.add_argument('--random-init-pos', default=False, action='store_true')
parser.add_argument('--vae', default=False, action='store_true')
parser.add_argument('--expert-break-on-succ', default=False, action='store_true') #if use flag expert demos will stop on success and not max_length. default is to run until max_steps
parser.add_argument('--res-timestep', default=False, action='store_true')
parser.add_argument('--reduced-state-space', default=False)
parser.add_argument('--fourier', default=False, action='store_true')
parser.add_argument('--render', default=False, action='store_true', help='if True will render expert and bc')
parser.add_argument('--debug', default=False, action='store_true') #smaller number or epochs and trajs
parser.add_argument('--quad1low', default=[0., 0.])
parser.add_argument('--quad1high', default=[0.2, 0.2])
parser.add_argument('--quad1', default=[0., np.pi/2])
parser.add_argument('--quad2low', default=[-0.2, 0.])
parser.add_argument('--quad2high', default=[0., 0.2])
parser.add_argument('--quad2', default=[np.pi/2, np.pi])
parser.add_argument('--quad3low', default=[-0.2, -0.2])
parser.add_argument('--quad3high', default=[0., 0.])
parser.add_argument('--quad3', default=[np.pi, (3*np.pi)/2])
parser.add_argument('--quad4low', default=[0., -0.2])
parser.add_argument('--quad4high', default=[0.2, 0.])
parser.add_argument('--quad4', default=[np.pi, 2*np.pi])
parser.add_argument('--radius', default=2)
parser.add_argument('--horizon', default=200)
parser.add_argument('--expert-n-traj', type=int, default=1000) #1000
parser.add_argument('--bc-n-epochs', type=int, default=5000) #5000
parser.add_argument('--bc-hidden-layer-size', type=int, default=32)
parser.add_argument('--bc-hidden-depth', type=int, default=0)
parser.add_argument('--bc-batch-size', type=int, default=32)
parser.add_argument('--bc-n-test-traj', type=int, default=50)
parser.add_argument('--NN-n-epochs', type=int, default=5000)
parser.add_argument('--NN-hidden-layer-size', type=int, default=32)
parser.add_argument('--NN-hidden-depth', type=int, default=0)
parser.add_argument('--NN-batch-size', type=int, default=32)
parser.add_argument('--bilinear-n-epochs', type=int, default=5000)
parser.add_argument('--bilinear-hidden-layer-size', type=int, default=32)
parser.add_argument('--bilinear-hidden-depth', type=int, default=2)
parser.add_argument('--bilinear-batch-size', type=int, default=32)


train_args = parser.parse_args()
train_args_json = json.dumps(vars(train_args))
log_path = os.path.join('bilinear', train_args.env_name, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
if not os.path.exists(log_path):
    os.makedirs(log_path)
with open(os.path.join(log_path, 'train_args.json'), 'w') as f:
    json.dump(train_args_json, f, indent=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_gpu_mode(True)

env = WheeledEnv(sample_goal_during_reset=False, sample_start_during_reset=False)

obs_size = env.observation_space.shape[0]
ac_size = env.action_space.shape[0]
goal_size = len(env.goal)
EPS = 1e-9    

def collect_data(env, policy, num_trajs=None, device=None, render=False):
    """Collect expert rollouts from the environment, store it in some datasets"""
    trajs = []
    angles = []
    goal_poses = []
    for tn in range(num_trajs):    
        # angle = np.random.uniform(train_args.quad1[0], train_args.quad1[1])
        # xpos = train_args.radius * np.cos(angle)
        # ypos = train_args.radius * np.sin(angle) 
        # goal_pos = np.array([xpos, ypos])
        goal_pos = np.random.uniform(low=train_args.quad1low, high=train_args.quad1high, size=(goal_size,))
        env.set_goal(goal_pos)
        traj = rollout(env, policy, max_path_length=train_args.horizon)
        trajs.append(traj)
        angle_curr = np.arctan2(traj['observations'][-1, -2], traj['observations'][-1, -1])
        angles.append(angle_curr)
        goal_poses.append(goal_pos)
    return trajs, angles, goal_poses


def plot_traj(trajs, tag, nplot=20):
    colors = sns.color_palette("hls", nplot)
    plot_idx = random.sample(range(len(trajs)), k=nplot) #random
    fig = plt.figure()  
    # plt.xlim(-2.5, 2.5)
    # plt.ylim(-2.5, 2.5)  
    end_dists = []    
    for colorid, idx in enumerate(plot_idx):
        traj = np.array(trajs[idx]['observations'])
        angle_curr = np.arctan2(traj[-1, -2], traj[-1, -1])
        plt.plot(traj[:, 0], traj[:, 1], color=colors[colorid]) #c=cm.viridis(angle_curr)
        plt.scatter(traj[-1,-2], traj[-1,-1], color=colors[colorid], marker='x') #goal 
        end_dists.append(np.linalg.norm(traj[-1,:2] - traj[-1,-2:]))
    end_dists = np.array(end_dists)
    plt.title(round(np.mean(end_dists),4))         
    plt.savefig(os.path.join(log_path, tag))


def plot_act(trajs, tag):
    #plot expert acts
    fig, ax = plt.subplots(ac_size, 1)
    for traj in trajs:
        ax[0].plot(traj['actions'][:, 0])
        ax[1].plot(traj['actions'][:, 1])
    plt.savefig(os.path.join(log_path, tag))


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


def train_bc(trajs):
    bc_policy = Policy(obs_size, ac_size, train_args.bc_hidden_layer_size, train_args.bc_hidden_depth) 
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
    return bc_policy


def eval_policy(policy, mode, n_eval):
    trajs_test = []
    goal_poses_test = []
    for tn in range(n_eval):
        env.reset()
        if mode == 'in_dist':
            # angle = np.random.uniform(train_args.quad1[0], train_args.quad1[1])
            # xpos = train_args.radius * np.cos(angle)
            # ypos = train_args.radius * np.sin(angle)            
            # goal_pos = [xpos, ypos]
            goal_pos = np.random.uniform(low=train_args.quad1low, high=train_args.quad1high, size=(goal_size,))
            init_pos = np.random.uniform(low=train_args.quad1low, high=train_args.quad1high, size=(goal_size,))
        else: #test ood quad2
            # angle = np.random.uniform(train_args.quad2[0], train_args.quad2[1])
            # xpos = train_args.radius * np.cos(angle)
            # ypos = train_args.radius * np.sin(angle)            
            # goal_pos = [xpos, ypos]
            goal_pos = np.random.uniform(low=train_args.quad2low, high=train_args.quad2high, size=(goal_size,))            
            init_pos = np.random.uniform(low=train_args.quad2low, high=train_args.quad2high, size=(goal_size,))
        if not train_args.random_init_pos:
            init_pos = None 
        o = env.set_goal(goal_pos)
        o = process_o(o)
        traj = {'observations': [],'actions': [], 'next_observations': [], 'dones': [], 'rewards': []}
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
            o = no
        traj['observations'] = np.array(traj['observations'])
        traj['actions'] = np.array(traj['actions'])
        traj['next_observations'] = np.array(traj['next_observations'])
        traj['dones'] = np.array(traj['dones'])
        trajs_test.append(traj)
        goal_poses_test.append(goal_pos)     
    return trajs_test


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
    return nn_policy


def eval_nn_bestworst(quad, nn_policy, trajs):
    # Check residuals on OOD points
    #TODO #eval - choose some delta that was in train?
    size_sample = 10
    colors = sns.color_palette("hls", size_sample)
    train_goals = np.concatenate([t['observations'][-1][-goal_size:][None].copy() for t in trajs])
    fig = plt.figure()      
    # plt.xlim(-2.5, 2.5)
    # plt.ylim(-2.5, 2.5)       
    end_dists = []
    all_diffs_extrap_e = []
    for g_idx in range(size_sample):
        if quad == 1:
            # angle = np.random.uniform(train_args.quad1[0], train_args.quad1[1])
            # xpos = train_args.radius * np.cos(angle)
            # ypos = train_args.radius * np.sin(angle)            
            # goal_pos = [xpos, ypos]      
            goal_pos = np.random.uniform(low=train_args.quad1low, high=train_args.quad1high, size=(goal_size,))      
            init_pos = np.random.uniform(low=train_args.quad1low, high=train_args.quad1high, size=(goal_size,))
        elif quad == 2:                                    
            # angle = np.random.uniform(train_args.quad2[0], train_args.quad2[1])
            # xpos = train_args.radius * np.cos(angle)
            # ypos = train_args.radius * np.sin(angle)            
            # goal_pos = [xpos, ypos]  
            goal_pos = np.random.uniform(low=train_args.quad2low, high=train_args.quad2high, size=(goal_size,))          
            init_pos = np.random.uniform(low=train_args.quad2low, high=train_args.quad2high, size=(goal_size,)) 
        elif quad == 3:
            # angle = np.random.uniform(train_args.quad3[0], train_args.quad3[1])
            # xpos = train_args.radius * np.cos(angle)
            # ypos = train_args.radius * np.sin(angle)            
            # goal_pos = [xpos, ypos]            
            goal_pos = np.random.uniform(low=train_args.quad3low, high=train_args.quad3high, size=(goal_size,))
            init_pos = np.random.uniform(low=train_args.quad3low, high=train_args.quad3high, size=(goal_size,))  
        elif quad == 4:
            # angle = np.random.uniform(train_args.quad4[0], train_args.quad4[1])
            # xpos = train_args.radius * np.cos(angle)
            # ypos = train_args.radius * np.sin(angle)            
            # goal_pos = [xpos, ypos] 
            goal_pos = np.random.uniform(low=train_args.quad4low, high=train_args.quad4high, size=(goal_size,))           
            init_pos = np.random.uniform(low=train_args.quad4low, high=train_args.quad4high, size=(goal_size,))
        if not train_args.random_init_pos:
            init_pos = None 
        end_pos = goal_pos.copy()
        traj_curr_goal = []
        dist_curr_goal = []        
        for k in range(len(trajs)):
            o = env.reset()
            o = env.set_goal(goal_pos)
            o = process_o(o)
            #Find closest point 
            closest_point_idx = k
            closest_point = train_goals[closest_point_idx].copy()
            closest_traj_obs = trajs[closest_point_idx]['observations'].copy()
            g_diff = end_pos - closest_point
            g_diff_t = torch.Tensor(g_diff[None]).to(device)
            traj = {'observations': [],'actions': [], 'next_observations': []}
            for i in range(train_args.horizon):
                t1s = torch.Tensor(closest_traj_obs[i][None]).to(device)
                deltas = torch.Tensor(o[None]).to(device) - t1s     
                nn_input = torch.cat([t1s, deltas], dim=-1)          
                ac_final = nn_policy(nn_input).cpu().detach().numpy()[0]
                no, r, d, _ = env.step(ac_final)
                no = process_o(no).copy()
                traj['observations'].append(o.copy())
                traj['actions'].append(ac_final.copy())
                traj['next_observations'].append(no.copy())
                o = no.copy()
            traj_curr_goal.append(traj)
            dist_curr_goal.append(np.linalg.norm(traj['observations'][-1][:2] - end_pos.copy()))            
        dist_curr_goal = np.array(dist_curr_goal)
        best_idx = np.argmin(dist_curr_goal)
        best_traj = traj_curr_goal[best_idx]        
        worst_idx = np.argmax(dist_curr_goal)
        worst_traj = traj_curr_goal[worst_idx]
        #plot
        plt.plot(np.array(best_traj['observations'])[:,0], np.array(best_traj['observations'])[:,1], color=colors[g_idx])
        plt.plot(np.array(worst_traj['observations'])[:,0], np.array(worst_traj['observations'])[:,1], linestyle=':', color=colors[g_idx])        
        plt.scatter([end_pos[0]],[end_pos[1]], color=colors[g_idx], marker='x', s=50)
        plt.scatter([trajs[best_idx]['observations'][-1][-2]], [trajs[best_idx]['observations'][-1][-1]], color=colors[g_idx], marker='^', s=50)
        plt.scatter([trajs[worst_idx]['observations'][-1][-2]], [trajs[worst_idx]['observations'][-1][-1]], color=colors[g_idx], marker='o', s=50)          
        end_dists.append(np.linalg.norm(best_traj['observations'][-1][:2] - end_pos.copy()))  
    end_dists = np.array(end_dists)
    all_diffs_extrap_e = np.array(all_diffs_extrap_e)
    print("Average end distance is %.5f"%(np.mean(end_dists)))
    plt.title(round(np.mean(end_dists),4))  
    plt.savefig(os.path.join(log_path, 'test_nn_worstbest'+str(quad)+'.png'))         


def train_bilinear(trajs):
    #all to all training 
    #TODO reshape batch size!
    bilinear_policy = BilinearPolicy(obs_size, ac_size, train_args.bilinear_hidden_layer_size, \
                                        train_args.bilinear_hidden_depth, train_args.fourier) 
    bilinear_policy.to(device)
    optimizer = optim.Adam(list(bilinear_policy.parameters()))
    losses = []
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
    return bilinear_policy      


def eval_bilinear_bestworst(quad, bilinear_policy, trajs):
    # Check residuals on OOD points
    #TODO #eval - choose some delta that was in train?
    size_sample = 10
    colors = sns.color_palette("hls", size_sample)
    train_goals = np.concatenate([t['observations'][-1][-goal_size:][None].copy() for t in trajs])
    fig = plt.figure()    
    # plt.xlim(-2.5, 2.5)
    # plt.ylim(-2.5, 2.5)          
    end_dists = []
    all_diffs_extrap_e = []
    for g_idx in range(size_sample):
        if quad == 1:
            # angle = np.random.uniform(train_args.quad1[0], train_args.quad1[1])
            # xpos = train_args.radius * np.cos(angle)
            # ypos = train_args.radius * np.sin(angle)            
            # goal_pos = [xpos, ypos]            
            goal_pos = np.random.uniform(low=train_args.quad1low, high=train_args.quad1high, size=(goal_size,))
            init_pos = np.random.uniform(low=train_args.quad1low, high=train_args.quad1high, size=(goal_size,))
        elif quad == 2:                                    
            # angle = np.random.uniform(train_args.quad2[0], train_args.quad2[1])
            # xpos = train_args.radius * np.cos(angle)
            # ypos = train_args.radius * np.sin(angle)            
            # goal_pos = [xpos, ypos] 
            goal_pos = np.random.uniform(low=train_args.quad2low, high=train_args.quad2high, size=(goal_size,))           
            init_pos = np.random.uniform(low=train_args.quad2low, high=train_args.quad2high, size=(goal_size,)) 
        elif quad == 3:
            # angle = np.random.uniform(train_args.quad3[0], train_args.quad3[1])
            # xpos = train_args.radius * np.cos(angle)
            # ypos = train_args.radius * np.sin(angle)            
            # goal_pos = [xpos, ypos]   
            goal_pos = np.random.uniform(low=train_args.quad3low, high=train_args.quad3high, size=(goal_size,))         
            init_pos = np.random.uniform(low=train_args.quad3low, high=train_args.quad3high, size=(goal_size,))  
        elif quad == 4:
            # angle = np.random.uniform(train_args.quad4[0], train_args.quad4[1])
            # xpos = train_args.radius * np.cos(angle)
            # ypos = train_args.radius * np.sin(angle)            
            # goal_pos = [xpos, ypos]  
            goal_pos = np.random.uniform(low=train_args.quad4low, high=train_args.quad4high, size=(goal_size,))          
            init_pos = np.random.uniform(low=train_args.quad4low, high=train_args.quad4high, size=(goal_size,))
        if not train_args.random_init_pos:
            init_pos = None 
        end_pos = goal_pos.copy()
        traj_curr_goal = []
        dist_curr_goal = []        
        for k in range(len(trajs)):
            o = env.reset()
            o = env.set_goal(goal_pos)
            o = process_o(o)
            #Find closest point 
            closest_point_idx = k
            closest_point = train_goals[closest_point_idx].copy()
            closest_traj_obs = trajs[closest_point_idx]['observations'].copy()
            g_diff = end_pos - closest_point
            g_diff_t = torch.Tensor(g_diff[None]).to(device)
            traj = {'observations': [],'actions': [], 'next_observations': []}
            for i in range(train_args.horizon):
                t1s = torch.Tensor(closest_traj_obs[i][None]).to(device)
                deltas = torch.Tensor(o[None]).to(device) - t1s               
                ac_final = bilinear_policy(t1s, deltas).cpu().detach().numpy()[0]
                no, r, d, _ = env.step(ac_final)
                no = process_o(no).copy()
                traj['observations'].append(o.copy())
                traj['actions'].append(ac_final.copy())
                traj['next_observations'].append(no.copy())
                o = no.copy()
            traj_curr_goal.append(traj)
            dist_curr_goal.append(np.linalg.norm(traj['observations'][-1][:2] - end_pos.copy()))            
        dist_curr_goal = np.array(dist_curr_goal)
        best_idx = np.argmin(dist_curr_goal)
        best_traj = traj_curr_goal[best_idx]        
        worst_idx = np.argmax(dist_curr_goal)
        worst_traj = traj_curr_goal[worst_idx]
        #plot
        plt.plot(np.array(best_traj['observations'])[:,0], np.array(best_traj['observations'])[:,1], color=colors[g_idx])
        plt.plot(np.array(worst_traj['observations'])[:,0], np.array(worst_traj['observations'])[:,1], linestyle=':', color=colors[g_idx])
        plt.scatter([end_pos[0]],[end_pos[1]], color=colors[g_idx], marker='x', s=50)
        plt.scatter([trajs[best_idx]['observations'][-1][-2]], [trajs[best_idx]['observations'][-1][-1]], color=colors[g_idx], marker='^', s=50)
        plt.scatter([trajs[worst_idx]['observations'][-1][-2]], [trajs[worst_idx]['observations'][-1][-1]], color=colors[g_idx], marker='o', s=50)            
        end_dists.append(np.linalg.norm(best_traj['observations'][-1][:2] - end_pos.copy()))
    end_dists = np.array(end_dists)
    all_diffs_extrap_e = np.array(all_diffs_extrap_e)
    print("Average end distance is %.5f"%(np.mean(end_dists)))
    plt.title(round(np.mean(end_dists),4))  
    plt.savefig(os.path.join(log_path, 'test_bilinear_worstbest'+str(quad)+'.png')) 



# Data collection
print('EXPERT')
#for each task, create 1000 expert demos
data = torch.load(train_args.expert_policy_filename)
expert_policy = data['evaluation/policy']
expert_policy.cuda()
trajs, angles, traj_goal_poses = collect_data(env, expert_policy, num_trajs=train_args.expert_n_traj, device=device)
plot_traj(trajs, 'expert.png', nplot=50)
plot_act(trajs, 'expert_acts.png')

#TODO plot losses at scale to check underfitting 
#BC
print('BC')  
#TRAIN BC  
bc_policy = train_bc(trajs)
#TEST BC - pi(s,g)=a
bc_traj_test = eval_policy(bc_policy, 'in_dist', train_args.bc_n_test_traj)
plot_traj(bc_traj_test, 'bc_in_dist.png', nplot=len(bc_traj_test))
bc_traj_test = eval_policy(bc_policy, 'ood', train_args.bc_n_test_traj)
plot_traj(bc_traj_test, 'bc_ood.png', nplot=len(bc_traj_test))

#NN - f(s,g,ds,dg)=a'
print('NN')
nn_policy = train_NN(trajs)
eval_nn_bestworst(1, nn_policy, trajs)
eval_nn_bestworst(2, nn_policy, trajs)
eval_nn_bestworst(3, nn_policy, trajs)

#BILINEAR - f(s,g)g(ds,dg)=a'
print('BILINEAR')
#train - all to all 
#TODO learn weighting?
#TODO timestep?
bilinear_policy = train_bilinear(trajs)
eval_bilinear_bestworst(1, bilinear_policy, trajs)
eval_bilinear_bestworst(2, bilinear_policy, trajs)
eval_bilinear_bestworst(3, bilinear_policy, trajs)
eval_bilinear_bestworst(4, bilinear_policy, trajs)

