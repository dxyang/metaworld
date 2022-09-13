"""for reach-v2 and push-v2, set the goal/obj range not by env."""

from __future__ import unicode_literals, print_function, division
import math
import torch.optim as optim
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
from io import open
import unicodedata
import string
import re
import random

import torch.nn as nn
from torch import optim

import pdb
import functools
import cv2

from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import test_cases_latest_nonoise
from render_utils import trajectory2vid
from networks import *


def run_eval_original(trajs, render=True):
    """Check original policy on OOD points"""
    end_dists = []
    returns = []
    for k in range(len(trajs)):
        #compare gripper pos and goal pos
        goal_pos = trajs[k]['obs'][-1][-3:]
        end_dists.append(np.linalg.norm(trajs[k]['obs'][-1][:3] - goal_pos))
        returns.append(np.sum(trajs[k]['reward']))
        # print('success', trajs[k]['done'][-1])
        if render:
            trajectory2vid(env, trajs[k], 'bc_ood_'+env_name+'_'+str(k)+'_'+str(goal_pos), env_name)
    end_dists = np.array(end_dists)
    print("Average end distance is %.5f"%(np.mean(end_dists)))
    print("Average return is %.5f"%(np.mean(returns)))
    print("Average length is %.5f"%(np.mean([len(t['obs']) for t in trajs])))
    print("Success rate is %.5f"%(np.mean([t['done'][-1] for t in trajs])))
    print("Mean final reward is %.5f"%(np.mean([t['reward'][-1] for t in trajs])))


def collect_data(env, policy, num_trajs=1000, mode='in_dist', device=None):
    """Collect expert rollouts from the environment, store it in some datasets"""
    trajs = []
    for tn in range(num_trajs):
        env.reset()
        #use env params / input to decide range
        low = env.goal_space.low #(-0.1, 0.8, 0.05)
        high = env.goal_space.high #(0.1, 0.9, 0.3)
        split_per = 0.5 #0.8
        # split_per = np.array([split_per, 1., 1.])
        if mode == 'in_dist':
            high = low + split_per*(high - low)
        elif mode == 'ood':
            low = low + split_per*(high - low)
        # print('collect data', mode, high, low)
        end_pos = np.random.uniform(low, high, size=(3,))
        env.goal = end_pos
        o = env.reset_model()
        traj = {'obs': [],'action': [], 'next_obs': [], 'done': [], 'reward': []}
        for _ in range(env.max_path_length):
            if hasattr(policy,'get_action'):
                ac = policy.get_action(o)
            else:
                t1s = torch.Tensor(o[None]).to(device)
                ac = policy(t1s).cpu().detach().numpy()[0]
            no, r, _, info = env.step(ac)
            traj['obs'].append(o.copy())
            traj['action'].append(ac.copy())
            traj['next_obs'].append(no.copy())
            traj['done'].append(info['success'])
            # traj['reward'].append(r)
            traj['reward'].append(info['in_place_reward'])
            o = no
            if info['success']: #break when done
                break
        traj['obs'] = np.array(traj['obs'])
        traj['action'] = np.array(traj['action'])
        traj['next_obs'] = np.array(traj['next_obs'])
        traj['done'] = np.array(traj['done'])
        trajs.append(traj)
    return trajs


#TODO vae default None
def get_latent(x, vae=None, vae_mode=False):
    if vae_mode:
        data_t = torch.Tensor(x[None]).to(device)
        _, _, mu, var = vae.forward(data_t)
        return mu.detach().cpu().numpy()[0]
    else:
        return x


#ENV
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#TODO extend to more tasks, goal pos should always be last 3 pos.
env_name = 'reach-v2'
reach_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["reach-v2-goal-observable"]
# env_name = 'push-v2'
# reach_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["push-v2-goal-observable"]

env = reach_goal_observable_cls(seed=0)
env.random_init = False
obs_size = env.observation_space.shape[0]
ac_size = env.action_space.shape[0]
goal_size = env.goal_space.shape[0]


#BC
expert_policy = functools.reduce(lambda a,b : a if a[0] == env_name else b, test_cases_latest_nonoise)[1]
trajs = collect_data(env, expert_policy, num_trajs=1000)
run_eval_original(trajs, render=False)
# pdb.set_trace()
#visualize some train trajectories (save as videos)
# for demo_id in random.choices(range(len(trajs)), k=3):
#     print('generating demo video', demo_id)
#     trajectory2vid(env, trajs[demo_id], 'expert_vids_'+env_name+'_'+str(demo_id), env_name)
print('saved videos')
fig = plt.figure()
plt.scatter([t['obs'][0,-3] for t in trajs],[t['obs'][0,-2] for t in trajs])
fig.savefig(os.path.join('figs',env_name,'expert_targets.png'))
trajs_ood = collect_data(env, expert_policy, mode='ood', num_trajs=1000)
fig = plt.figure()
plt.scatter([t['obs'][0,-3] for t in trajs_ood],[t['obs'][0,-2] for t in trajs_ood])
fig.savefig(os.path.join('figs',env_name,'expert_targets_ood.png'))
# pdb.set_trace()

# BC training
hidden_layer_size = 1000
hidden_depth = 3
policy = Policy(obs_size, ac_size, hidden_layer_size, hidden_depth) # 10 dimensional latent
policy.to(device)
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
        #each trajectory different length
        t1_idx_pertraj = np.array([np.random.randint(len(trajs[idx]['obs'])) for idx in t1_idx])
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
#plot losses
fig = plt.figure()
plt.plot(losses)
if not os.path.exists('figs'):
    os.mkdir('figs')
fig.savefig(os.path.join('figs',env_name,'bc_losses.png'))
#visualize some BC trajectories (save as videos)
bc_trajs = collect_data(env, policy, num_trajs=5, device=device)
# for demo_id in random.choices(range(len(bc_trajs)), k=1):
#     print('generating bc video', demo_id)
#     goal_pos = str(bc_trajs[demo_id]['obs'][-1][-3:])
#     trajectory2vid(env, bc_trajs[demo_id], 'bc_vids_'+env_name+'_'+str(demo_id)+'_'+goal_pos, env_name)
print('Finished Training')


#VAE
vae_mode = False
if vae_mode:
    trajs_vae = collect_data(env, expert_policy, device=device)
    #TODO for high dim state space what latent_dim/in_dim makes sense?
    latent_dim = 3
    in_dim = 3
    vae = VanillaVAE(in_dim=in_dim, latent_dim=latent_dim, hidden_dims=[64,64])
    vae.to(device)
    data = []
    for traj in trajs_vae:
        data.append(traj['obs'])
    #TODO what are taking out of state? only goal pos? shouldn't be full state?
    data = np.concatenate(data, axis=0)[:,-3:]
    # VAE Training
    print('Training VAE')
    num_epochs = 50
    batch_size = 64
    EPS = 1e-9
    optimizer = optim.Adam(vae.parameters())
    num_data = len(data)
    losses = []
    recon_losses = []
    kld_losses = []
    idxs = np.array(range(len(data)))
    num_batches = len(idxs) // batch_size
    # Train the model with regular SGD
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        np.random.shuffle(idxs)
        running_loss = 0.0
        running_recon_loss = 0.0
        running_kl_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()
            curr_idx1 = idxs[batch_size*i: batch_size*i + batch_size]
            x = torch.from_numpy(data[curr_idx1]).to(device).float()
            recon, input_val, mu, var = vae.forward(x)
            losses_all = vae.loss_function(recon, input_val, mu, var, **{'M_N': 10.*float(batch_size)/float(num_data)})
            loss = losses_all['loss']
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.cpu().detach().numpy()
            running_recon_loss += losses_all['Reconstruction_Loss'].cpu().detach().numpy()
            running_kl_loss += losses_all['KLD'].cpu().detach().numpy()
            if i % 1 == 0 and i > 0:    # print every 2000 mini-batches
                # print('[%d, %5d] full loss: %.6f, Reconstruction_Loss: %.6f,  KLD: %.6f,' %
                #     (epoch + 1, i + 1, running_loss / 100., running_recon_loss/100., running_kl_loss/100.))
                losses.append(running_loss/100.)
                recon_losses.append(running_recon_loss/100.)
                kld_losses.append(running_kl_loss/100.)
                running_loss = 0.0
                running_recon_loss = 0.0
                running_kl_loss = 0.0
    #plot vae losses and reconstruction
    fig = plt.figure()
    plt.ylim(0, 0.01)
    plt.plot(losses)
    fig.savefig(os.path.join('figs',env_name,'vae_losses.png'))
    fig = plt.figure()
    plt.plot(recon_losses, color='g')
    fig.savefig(os.path.join('figs',env_name,'vae_recon_losses.png'))
    fig = plt.figure()
    plt.plot(kld_losses, color='r')
    fig.savefig(os.path.join('figs',env_name,'vae_kld_losses.png'))
    # Testing reconstruction
    num_plot = 100
    new_data = copy.deepcopy(data)
    np.random.shuffle(new_data)
    x = torch.from_numpy(new_data[:num_plot]).to(device).float()
    recon, input_val, mu, var = vae.forward(x)
    recon_np = recon.cpu().detach().numpy()
    fig = plt.figure()
    #TODO view 3d recon
    plt.scatter(new_data[:num_plot,0], new_data[:num_plot,1])
    plt.scatter(recon_np[:num_plot,0], recon_np[:num_plot,1], marker='x')
    fig.savefig('figs/vae_reconstruction.png')
    #TODO visualize mu, if latent_dim==2 can use notebook.
    print('Finished Training VAE')


#RESIDUAL
mode = 'concat'
state_mode = True
#TODO dimensions (all but concat)
if mode == 'nonlinear_hypernet':
    # Training an adaptation policy based on shifts directly in parameter space
    adapt_policy = TransformPolicy(2, 2, 32, 2)
    full_param_shapes = np.array([p.shape for p in list(adapt_policy.parameters())])
    param_shapes = np.array([p.flatten().shape[0] for p in list(adapt_policy.parameters())])
    size_params = np.sum(param_shapes)
    param_breakdowns = np.cumsum(param_shapes)
    res_out_size = size_params
elif mode =='linear_hypernet':
    res_out_size = 6
elif mode =='concat':
    res_out_size = ac_size
elif mode =='add':
    res_out_size = 2
elif mode == 'combine':
    res_out_size = 2
    action_size = 2
    comb_policy = Policy(action_size+res_out_size, res_out_size, 32, 1) #combine action with residual
    comb_policy.to(device)
if vae_mode:
    res_in_size = ac_size + latent_dim
else:
    res_in_size = ac_size + goal_size #action, delta
if state_mode:
    res_in_size += obs_size #state input too
hidden_layer_size = 32
hidden_depth = 1
if state_mode:
    hidden_layer_size = 64
    hidden_depth = 2
residual_policy = Policy(res_in_size, res_out_size, hidden_layer_size, hidden_depth)
residual_policy.to(device)
num_epochs = 500
batch_size = 50
EPS = 1e-9
criterion = nn.MSELoss()
opt_params = list(residual_policy.parameters())
if mode == 'combine':
    opt_params += list(comb_policy.parameters())
optimizer = optim.Adam(opt_params)
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
        t1_idx_pertraj = np.array([np.random.randint(min(len(trajs[idx1]['obs']),len(trajs[idx2]['obs']))) for idx1,idx2 in zip(t1_idx,t2_idx)])
        t1_states = np.concatenate([trajs[c_idx]['obs'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
        t1_actions = np.concatenate([trajs[c_idx]['action'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
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
        #goal as input to latent
        g_delta = np.concatenate([get_latent(trajs[t2_idx_diff]['obs'][-1, -3:])[None] -
                                    get_latent(trajs[t1_idx_diff]['obs'][-1, -3:])[None]
                                     for (t1_idx_diff, t2_idx_diff) in zip(t1_idx, t2_idx)])
        g_delta_torch = torch.Tensor(g_delta).float().to(device)
        # Full action transform
        if mode == 'concat':
            res_in = torch.cat([a1_pred.detach(), g_delta_torch], dim=-1)
            if state_mode:
                res_in = torch.cat([res_in, t1_states], dim=-1)
            residual_pred = residual_policy(res_in)
            a2_pred_residual = residual_pred
        # Hypernet
        elif mode == 'linear_hypernet':
            residual_pred = residual_policy(g_delta_torch)
            w_reshaped = torch.reshape(residual_pred[:, :4], (residual_pred.shape[0], 2, 2))
            b_reshaped = torch.reshape(residual_pred[:, 4:6], (residual_pred.shape[0], 2))
            a2_preds = []
            for s_idx in range(len(a1_pred)):
                a2_pred_curr = torch.mm(w_reshaped[s_idx], a1_pred[s_idx][:, None]) + b_reshaped[s_idx][:, None]
                a2_preds.append(a2_pred_curr[:, 0][None])
            a2_pred_residual = torch.vstack(a2_preds)
        # Full NN Hypernet
        elif mode == 'nonlinear_hypernet':
            residual_pred = residual_policy(g_delta_torch)
            a2_preds = []
            for s_idx in range(len(a1_pred)):
                # Feed the residual vector through the adaptation network
                curr_params = residual_pred[s_idx]
                delta_params_W1 = curr_params[0: param_breakdowns[0]].reshape(full_param_shapes[0])
                delta_params_b1 = curr_params[param_breakdowns[0]:param_breakdowns[1]].reshape(full_param_shapes[1])
                delta_params_W2 = curr_params[param_breakdowns[1]:param_breakdowns[2]].reshape(full_param_shapes[2])
                delta_params_b2 = curr_params[param_breakdowns[2]:param_breakdowns[3]].reshape(full_param_shapes[3])
                delta_params_Wout = curr_params[param_breakdowns[3]:param_breakdowns[4]].reshape(full_param_shapes[4])
                delta_params_bout = curr_params[param_breakdowns[4]:param_breakdowns[5]].reshape(full_param_shapes[5])
                delta_params_list = [delta_params_W1, delta_params_b1,
                                delta_params_W2, delta_params_b2,
                                delta_params_Wout, delta_params_bout]
                a2_pred_curr = adapt_policy.forward_parameters(a1_pred[s_idx][None, :], delta_params_list)
                a2_preds.append(a2_pred_curr)
            a2_pred_residual = torch.vstack(a2_preds)
        elif mode == 'add':
            residual_pred = residual_policy(g_delta_torch)
            a2_pred_residual = a1_pred.detach() + residual_pred
        elif mode == 'combine':
            #residual
            res_in = torch.cat([a1_pred.detach(), g_delta_torch], dim=-1)
            if state_mode:
                t1_only_state = t1_states[:,:2] #obs [s,g], want only s
                res_in = torch.cat([res_in, t1_only_state], dim=-1)
            residual_pred = residual_policy(res_in)
            #combine
            comb_in = torch.cat([a1_pred, residual_pred], dim=-1)
            comb_pred = comb_policy(comb_in)
            a2_pred_residual = comb_pred
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
#plot losses
fig = plt.figure()
plt.plot(losses)
fig.savefig('figs/residual_losses.png')
print('Finished Training')


#Test RESIDUAL
def run_eval_residual(ood_goals, trajs, policy, residual_policy):
    """Check residuals on OOD points"""
    train_goals = np.concatenate([t['obs'][-1][-3:][None].copy() for t in trajs])
    train_goals_latent = np.concatenate([get_latent(t['obs'][-1][-3:])[None] for t in trajs])
    end_dists = []
    returns = []
    residual_ood_traj = []
    for k in range(len(ood_goals)):
        env.reset()
        end_pos = ood_goals[k]
        #Find closest point
        closest_point_idx = np.argmin(np.linalg.norm(train_goals_latent - get_latent(end_pos).copy(), axis=-1))
        closest_point = train_goals[closest_point_idx].copy()
        closest_traj_obs = trajs[closest_point_idx]['obs'].copy()
        env.goal = end_pos.copy()
        g_diff = get_latent(end_pos) - get_latent(closest_point)
        o = env.reset_model()
        print('closest_point', closest_point)
        traj = {'obs': [],'action': [], 'next_obs': [], 'done': [], 'reward': []}
        for i in range(len(closest_traj_obs)):
            t1s = torch.Tensor(closest_traj_obs[i][None]).to(device) #torch.Tensor(o[None]).to(device)
            ac = policy(t1s).detach()
            if mode == 'concat':
                g_diff_t = torch.Tensor(g_diff[None]).to(device)
                res_in = torch.cat([ac, g_diff_t], dim=-1)
                if state_mode:
                    res_in = torch.cat([res_in, t1s], dim=-1)
                ac_final = residual_policy(res_in).cpu().detach().numpy()[0]
            elif mode == 'linear_hypernet':
                # Hypernet
                g_diff_t = torch.Tensor(g_diff[None]).to(device)
                residual_pred = residual_policy(g_diff_t)
                w_reshaped = torch.reshape(residual_pred[:, :4], (residual_pred.shape[0], 2, 2))
                b_reshaped = torch.reshape(residual_pred[:, 4:6], (residual_pred.shape[0], 2))
                ac_final = torch.mm(w_reshaped[0], ac[0][:, None]) + b_reshaped[0][:, None]
                ac_final = ac_final.detach().cpu().numpy()[:, 0]
            elif mode == 'nonlinear_hypernet':
                g_diff_t = torch.Tensor(g_diff[None]).to(device)
                residual_pred = residual_policy(g_diff_t)
                # Feed the residual vector through the adaptation network
                curr_params = residual_pred[0]
                delta_params_W1 = curr_params[0: param_breakdowns[0]].reshape(full_param_shapes[0])
                delta_params_b1 = curr_params[param_breakdowns[0]:param_breakdowns[1]].reshape(full_param_shapes[1])
                delta_params_W2 = curr_params[param_breakdowns[1]:param_breakdowns[2]].reshape(full_param_shapes[2])
                delta_params_b2 = curr_params[param_breakdowns[2]:param_breakdowns[3]].reshape(full_param_shapes[3])
                delta_params_Wout = curr_params[param_breakdowns[3]:param_breakdowns[4]].reshape(full_param_shapes[4])
                delta_params_bout = curr_params[param_breakdowns[4]:param_breakdowns[5]].reshape(full_param_shapes[5])
                delta_params_list = [delta_params_W1, delta_params_b1,
                                delta_params_W2, delta_params_b2,
                                delta_params_Wout, delta_params_bout]
                ac_final = adapt_policy.forward_parameters(ac, delta_params_list)[0].detach().numpy()
            elif mode == 'add':
                g_diff_t = torch.Tensor(g_diff[None]).to(device)
                residual_ac = residual_policy(g_diff_t).cpu().detach().numpy()[0]
                ac_final = ac.cpu().numpy()[0] + residual_ac
            elif mode == 'combine':
                #residual
                g_diff_t = torch.Tensor(g_diff[None]).to(device)
                res_in = torch.cat([ac, g_diff_t], dim=-1)
                if state_mode:
                    t1_only_state = t1s[:,:2] #obs [s,g]
                    res_in = torch.cat([res_in, t1_only_state], dim=-1)
                residual_pred = residual_policy(res_in)
                #combine
                comb_in = torch.cat([ac, residual_pred], dim=-1)
                comb_pred = comb_policy(comb_in)
                ac_final = comb_pred.cpu().detach().numpy()[0]
            no, r, _, info = env.step(ac_final)
            traj['obs'].append(o.copy())
            traj['action'].append(ac_final.copy())
            traj['next_obs'].append(no.copy())
            traj['done'].append(info['success'])
            # traj['reward'].append(r)
            traj['reward'].append(info['in_place_reward'])
            o = no.copy()
        #compare gripper pos and goal pos
        end_dists.append(np.linalg.norm(traj['obs'][-1][:3] - end_pos.copy()))
        returns.append(np.sum(traj['reward']))
        # print('success', traj['done'][-1])
        #create video from traj
        # trajectory2vid(env, traj, 'residual_ood_vids_'+env_name+'_'+str(k)+'_'+str(end_pos), env_name)
        residual_ood_traj.append(traj)
    end_dists = np.array(end_dists)
    print("Average end distance is %.5f"%(np.mean(end_dists)))
    print("Average return is %.5f"%(np.mean(returns)))
    print("Average length is %.5f"%(np.mean([len(t['obs']) for t in residual_ood_traj])))
    print("Success rate is %.5f"%(np.mean([t['done'][-1] for t in residual_ood_traj])))
    print("Mean final reward is %.5f"%(np.mean([t['reward'][-1] for t in residual_ood_traj])))
    return residual_ood_traj

def plot_3d_viz(trajs, tag, nplot=20):
    colors = sns.color_palette("hls", nplot)
    plot_idx = random.sample(range(len(trajs)), k=nplot) #random
    final_rews = np.array([traj['reward'][-1] for traj in trajs])
    fig = plt.figure()  
    ax = plt.axes(projection='3d')      
    end_dists = []    
    for colorid, idx in enumerate(plot_idx):
        t = np.array(trajs[idx]['obs'])
        ax.plot3D(t[:,0], t[:,1], t[:,2], color=colors[colorid], linestyle=':') #end effector traj
        ax.scatter3D(t[-1,-3], t[-1,-2], t[-1,-1], color=colors[colorid], marker='x') #gt goal     
        end_dists.append(np.linalg.norm(t[-1][:3] - t[-1][-3:])) #dist between end effector and gt goal
    end_dists = np.array(end_dists)
    plt.title(round(np.mean(end_dists),4))         
    fig.savefig(tag)


# Sample some OOD goals
n_test=20
#bc on ood (defines ood goals and uses bc policy)
ood_traj = collect_data(env, policy, num_trajs=n_test, mode='ood', device=device)
print('eval BC ood')
run_eval_original(ood_traj, render=False)
# pdb.set_trace()
plot_3d_viz(ood_traj, os.path.join('figs',env_name,'bc_ood_3d.png'))
plot_3d_viz(trajs, os.path.join('figs',env_name,'expert_3d.png'))
bc_traj = collect_data(env, policy, num_trajs=n_test, mode='in_dist', device=device)
plot_3d_viz(bc_traj, os.path.join('figs',env_name,'bc_in_dist_3d.png'))

#residual on ood
ood_goals = np.concatenate([t['obs'][-1][-3:][None].copy() for t in ood_traj])
print('eval RESIDUAL ood')
residual_ood_traj = run_eval_residual(ood_goals, trajs, policy, residual_policy)
plot_3d_viz(residual_ood_traj, os.path.join('figs',env_name,'res_ood_3d.png'))
pdb.set_trace()

#TODO some latent space analysis?
