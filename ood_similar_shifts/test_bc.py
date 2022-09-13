#break bc.
import argparse
import numpy as np
import pdb
import functools
import os
import random
import matplotlib.pyplot as plt
from gym.spaces import Box
import copy
import seaborn as sns

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import test_cases_latest_nonoise
from render_utils import trajectory2vid
from networks import *


parser = argparse.ArgumentParser(description='task and ood config')
parser.add_argument('--task-name', type=str, default='reach-v2')
parser.add_argument('--split-per', type=float, default=0.5, help='percent of Box in dist')
parser.add_argument('--goal-ood', default=False, action='store_true')
parser.add_argument('--obj-ood', default=False, action='store_true')
# parser.add_argument('--random-hand-init', default=False, action='store_true')
parser.add_argument('--ood-axis', type=int, default=0) #x axis
parser.add_argument('--render', default=False, action='store_true', help='if True will render expert and bc')
parser.add_argument('--vae', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true') #smaller number or epochs and trajs
parser.add_argument('--expert-break-on-succ', default=False, action='store_true') #if use flag expert demos will stop on success and not max_length. default is to run until max_steps
parser.add_argument('--res-timestep', default=False, action='store_true')
parser.add_argument('--res-full-state', default=False, action='store_true')
parser.add_argument('--res-grasp-state', default=False, action='store_true')
parser.add_argument('--reduced-state-space', default=False, action='store_true')
parser.add_argument('--eval-res-best', default=False, action='store_true')
parser.add_argument('--train-bc', default=False, action='store_true')



#TODO make residual works without vae!
#TODO vae on states? -- distinguish between finding closest [what to transform] and residual input [what does transformation depend on]

def run_eval(trajs, render=False, tag=None, env_name=None, gen_type="", obj_poses=None, goal_poses=None):
    """get trajectory stats"""
    returns = []
    end_dists = [] #tcp
    closest_dists = [] #closest end effector has been to goal
    for k in range(len(trajs)):
        returns.append(np.sum(trajs[k]['reward']))
        goal_pos = trajs[k]['obs'][-1][-3:]
        #TODO not a good indicator for every task.
        end_dists.append(np.linalg.norm(trajs[k]['tcp'] - goal_pos))
        # pdb.set_trace()
        closest_dists.append(np.min(np.linalg.norm(trajs[k]['obs'][:,:3] - goal_pos, axis=-1)))
        # print('success', trajs[k]['done'][-1])
        if render:
            trajectory2vid(env, trajs[k], tag, env_name, gen_type, obj_pos=obj_poses[k], goal_pos=goal_poses[k])
    print("Average end distance is %.5f"%(np.mean(end_dists))) #tcp to target
    print("Average closest distance is %.5f"%(np.mean(closest_dists))) #closest end effector to target
    print("Average return is %.5f"%(np.mean(returns)))
    print("Average length is %.5f"%(np.mean([len(t['obs']) for t in trajs])))
    print("Success rate is %.5f"%(np.mean([t['done'][-1] for t in trajs])))
    print("Mean final reward is %.5f"%(np.mean([t['reward'][-1] for t in trajs])))

def get_latent(x, vae=None):
    if args.vae:
        assert vae is not None
        data_t = torch.Tensor(x[None]).to(device)
        _, _, mu, var = vae.forward(data_t)
        return mu.detach().cpu().numpy()[0]
    else:
        return x

def run_eval_residual(trajs, env, policy, residual_policy, train_obj=None,
                        train_goals=None, ood_obj=None, ood_goals=None, vae=None, device=None):
    """Check residuals on OOD points"""
    returns = []
    end_dists = []
    closest_dists = [] #closest end effector has been to goal
    residual_ood_traj = []
    for k in range(len(ood_goals)):
        env.reset()
        end_pos = ood_goals[k] if args.goal_ood else ood_obj[k]
        #Find closest goal/obj
        if args.goal_ood:
            train_goals_latent = np.concatenate([get_latent(t, vae)[None] for t in train_goals])
            closest_point_idx = np.argmin(np.linalg.norm(train_goals_latent - get_latent(end_pos.copy(), vae), axis=-1))
            closest_point = train_goals[closest_point_idx].copy()
        elif args.obj_ood:
            train_obj_latent = np.concatenate([get_latent(t, vae)[None] for t in train_obj])
            closest_point_idx = np.argmin(np.linalg.norm(train_obj_latent - get_latent(end_pos.copy(), vae), axis=-1))
            closest_point = train_obj[closest_point_idx].copy()
        closest_traj_obs = trajs[closest_point_idx]['obs'].copy()
        closest_traj_acts = trajs[closest_point_idx]['action'].copy()
        g_diff = get_latent(end_pos, vae) - get_latent(closest_point, vae)
        o, _, _ = env.reset_model_ood(obj_ood=args.obj_ood, goal_ood=args.goal_ood, obj_pos=ood_obj[k], goal_pos=ood_goals[k])
        if args.reduced_state_space:
            o = np.concatenate([o[:3], o[-3:]]).copy()
        traj = {'obs': [],'action': [], 'next_obs': [], 'done': [], 'reward': []}
        for i in range(len(closest_traj_obs)):
            t1s = torch.Tensor(closest_traj_obs[i][None]).to(device) #torch.Tensor(o[None]).to(device)
            if args.train_bc:
                ac = policy(t1s).detach()
            else:
                ac = torch.Tensor(closest_traj_acts[i][None]).to(device)
            g_diff_t = torch.Tensor(g_diff[None]).to(device)
            res_in = torch.cat([ac, g_diff_t], dim=-1)
            # pdb.set_trace()
            if args.res_timestep:
                res_in = torch.cat([res_in, torch.Tensor([i]).to(device)[None]], dim=-1)
            if args.res_grasp_state:
                res_in = torch.cat([res_in, t1s[:,3][None]], dim=-1)
            elif args.res_full_state:
                res_in = torch.cat([res_in, t1s], dim=-1)
            ac_final = residual_policy(res_in).cpu().detach().numpy()[0]
            no, r, _, info = env.step(ac_final)
            if args.reduced_state_space:
                no = np.concatenate([no[:3], no[-3:]]).copy()            
            traj['obs'].append(o.copy())
            traj['action'].append(ac_final.copy())
            traj['next_obs'].append(no.copy())
            traj['done'].append(info['success'])
            traj['reward'].append(info['in_place_reward'])
            o = no.copy()
            # if info['success']: #break when done
            #     break
        end_dists.append(np.linalg.norm(traj['obs'][-1][:3] - end_pos.copy()))
        closest_dists.append(np.min(np.linalg.norm(trajs[k]['obs'][:,:3] - end_pos, axis=-1)))
        returns.append(np.sum(traj['reward']))
        residual_ood_traj.append(traj)
    end_dists = np.array(end_dists)
    print("Average end distance is %.5f"%(np.mean(end_dists))) #tcp to target
    print("Average closest distance is %.5f"%(np.mean(closest_dists))) #closest end effector to target
    print("Average return is %.5f"%(np.mean(returns)))
    print("Average length is %.5f"%(np.mean([len(t['obs']) for t in residual_ood_traj])))
    print("Success rate is %.5f"%(np.mean([t['done'][-1] for t in residual_ood_traj])))
    print("Mean final reward is %.5f"%(np.mean([t['reward'][-1] for t in residual_ood_traj])))
    return residual_ood_traj


def run_eval_residual_best(trajs, env, policy, residual_policy, train_obj=None,
                        train_goals=None, ood_obj=None, ood_goals=None, vae=None, device=None):
    """Check residuals, transform all train, see which is best"""
    best_end_dists = []
    best_closest_dists = [] #closest end effector has been to goal
    best_returns = []    
    best_residual_trajs = []
    for k in range(len(ood_goals)): #new target       
        end_dists = []
        closest_dists = [] #closest end effector has been to goal
        returns = []   
        residual_trajs = []     
        end_pos = ood_goals[k] if args.goal_ood else ood_obj[k]
        for j, transform_traj in enumerate(trajs): #what to transform
            transform_traj_obs = transform_traj['obs'].copy()
            # pdb.set_trace()
            transform_traj_acts = transform_traj['action'].copy()
            # assert transform_traj[-1][-3:].copy() == train_goals[j]
            # print(transform_traj[-1][-3:], train_goals[j])
            g_diff = get_latent(end_pos, vae) - get_latent(train_goals[j], vae)
            env.reset()
            o, _, _ = env.reset_model_ood(obj_ood=args.obj_ood, goal_ood=args.goal_ood, obj_pos=ood_obj[k], goal_pos=ood_goals[k])
            if args.reduced_state_space:
                o = np.concatenate([o[:3], o[-3:]]).copy()
            traj = {'obs': [],'action': [], 'next_obs': [], 'done': [], 'reward': []}
            for i in range(len(transform_traj_obs)):
                t1s = torch.Tensor(transform_traj_obs[i][None]).to(device) #torch.Tensor(o[None]).to(device)
                if args.train_bc:
                    ac = policy(t1s).detach()
                else:
                    ac = torch.Tensor(transform_traj_acts[i][None]).to(device)
                g_diff_t = torch.Tensor(g_diff[None]).to(device)
                res_in = torch.cat([ac, g_diff_t], dim=-1)
                if args.res_timestep:
                    res_in = torch.cat([res_in, torch.Tensor([i]).to(device)[None]], dim=-1)
                if args.res_grasp_state:
                    res_in = torch.cat([res_in, t1s[:,3][None]], dim=-1)
                elif args.res_full_state:
                    res_in = torch.cat([res_in, t1s], dim=-1)
                ac_final = residual_policy(res_in).cpu().detach().numpy()[0]
                no, r, _, info = env.step(ac_final)
                if args.reduced_state_space:
                    no = np.concatenate([no[:3], no[-3:]]).copy()            
                traj['obs'].append(o.copy())
                traj['action'].append(ac_final.copy())
                traj['next_obs'].append(no.copy())
                traj['done'].append(info['success'])
                traj['reward'].append(info['in_place_reward'])
                o = no.copy()
                # if info['success']: #break when done
                #     break
            end_dists.append(np.linalg.norm(traj['obs'][-1][:3] - end_pos.copy()))
            closest_dists.append(np.min(np.linalg.norm(trajs[k]['obs'][:,:3] - end_pos, axis=-1)))
            returns.append(np.sum(traj['reward']))
            residual_trajs.append(traj) 
        best_idx = np.argmin(end_dists)
        best_end_dists.append(end_dists[best_idx])    
        best_closest_dists.append(closest_dists[best_idx])       
        best_returns.append(returns[best_idx])       
        best_residual_trajs.append(residual_trajs[best_idx]) 
    best_end_dists = np.array(best_end_dists)
    print("Average end distance is %.5f"%(np.mean(best_end_dists))) #tcp to target
    print("Average closest distance is %.5f"%(np.mean(best_closest_dists))) #closest end effector to target
    print("Average return is %.5f"%(np.mean(best_returns)))
    print("Average length is %.5f"%(np.mean([len(t['obs']) for t in best_residual_trajs])))
    print("Success rate is %.5f"%(np.mean([t['done'][-1] for t in best_residual_trajs])))
    print("Mean final reward is %.5f"%(np.mean([t['reward'][-1] for t in best_residual_trajs])))
    return best_residual_trajs


def collect_data(env, policy, num_trajs=1000, mode='in_dist', device=None, break_on_succ=True):
    """Collect expert rollouts from the environment, store it in some datasets"""
    trajs = []
    obj_poses = []
    goal_poses = []
    for tn in range(num_trajs):
        env.reset()
        if mode=='default':
            #sanity check expert demos [not train ones]
            o = env.reset_model()
            obj_pos, goal_pos = None, None
        elif mode=='vae': #collect data for vae training, use all
            env.random_init = True
            o = env.reset_model()
            env.random_init = False
            obj_pos, goal_pos = None, None
            if hasattr(env, 'obj_init_pos'):
                obj_pos = env.obj_init_pos
            if hasattr(env, 'goal'):
                goal_pos = env.goal
        else: #in_dist/ood collect data for bc or res training/evaluation
            # pdb.set_trace()
            o, obj_pos, goal_pos = env.reset_model_ood(mode, args.ood_axis, args.split_per, args.obj_ood, args.goal_ood)     
        traj = {'obs': [],'action': [], 'next_obs': [], 'done': [], 'reward': [], 'tcp': None}
        horizon = env.max_path_length
        # horizon = 100 #TODO
        for _ in range(horizon):
            if hasattr(policy,'get_action'):
                ac = policy.get_action(o)
                if args.reduced_state_space:
                    o = np.concatenate([o[:3], o[-3:]]).copy()                
            else:
                if args.reduced_state_space:
                    o = np.concatenate([o[:3], o[-3:]]).copy()                
                t1s = torch.Tensor(o[None]).to(device)
                ac = policy(t1s).cpu().detach().numpy()[0]
            no, r, _, info = env.step(ac)                             
            traj['obs'].append(o.copy())
            traj['action'].append(ac.copy())
            traj['next_obs'].append(no.copy()) #not used
            traj['done'].append(info['success'])
            traj['reward'].append(info['in_place_reward'])
            o = no
            if break_on_succ and info['success']: #break when done
                break
            # env.render()
        traj['obs'] = np.array(traj['obs'])
        traj['action'] = np.array(traj['action'])
        traj['next_obs'] = np.array(traj['next_obs'])
        traj['done'] = np.array(traj['done'])
        traj['tcp'] = env.tcp_center #gripper at end of trajectory TODO not good indicator for all tasks
        trajs.append(traj)
        obj_poses.append(obj_pos)
        goal_poses.append(goal_pos)
    return trajs, obj_poses, goal_poses

def plot_3d_viz(trajs, tag, nplot=20, threeD=True):
    colors = sns.color_palette("hls", nplot)
    plot_idx = random.sample(range(len(trajs)), k=nplot) #random
    final_rews = np.array([traj['reward'][-1] for traj in trajs])
    # plot_idx = np.argpartition(final_rews, nplot)[:nplot] #worst final rew
    # plot_idx = np.argpartition(final_rews, nplot)[-nplot:] #best final rew
    fig = plt.figure()  
    if threeD:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0.4, 1.0)      
    end_dists = []    
    for colorid, idx in enumerate(plot_idx):
        t = np.array(trajs[idx]['obs'])
        if threeD:
            ax.plot3D(t[:,0], t[:,1], t[:,2], color=colors[colorid], linestyle=':') #end effector traj
            ax.scatter3D(t[-1,-3], t[-1,-2], t[-1,-1], color=colors[colorid], marker='x') #gt goal
        else:
            ax.plot(t[:,0], t[:,1], color=colors[colorid], linestyle=':') #end effector traj
            ax.scatter(t[-1,-3], t[-1,-2], color=colors[colorid], marker='x') #gt goal     
        end_dists.append(np.linalg.norm(t[-1][:3] - t[-1][-3:])) #dist between end effector and gt goal
    end_dists = np.array(end_dists)
    plt.title(round(np.mean(end_dists),4))         
    fig.savefig(tag)


args = parser.parse_args()
if args.goal_ood:
    gen_type = 'goal'
elif args.obj_ood:
    gen_type = 'obj'

assert not (args.res_full_state and args.res_grasp_state)

#ENV
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)
#choose 10 tasks -
#   for each one what part is relevant to randomize (obj/target/hand init pos)
#   describe the task 'subtrajectories' e.g. push = grip + push
# ML10 tasks
#what is defined in init 'env.goal' and can seemingly be modified |
# paper (Table 2) says can modify [note that could be referring to v1 and not v2] |
# what reset_model actually modifies when random_init=True
m10_tasks = ['basketball-v2', #obj,goal,hand | obj,goal | obj,goal
         'button-press-v2', #obj,goal,hand | obj | obj
        #  'dial-turn-v2', #obj,goal,hand | obj | obj,goal
         'drawer-close-v2', #obj,hand | obj | obj
         'peg-insert-side-v2', #obj,goal,hand | obj,goal | obj,goal
         'pick-place-v2', #obj,goal,hand | obj,goal | obj,goal
         'push-v2', #obj,goal,hand | obj,goal | obj,goal
         'reach-v2', #obj,goal,hand | goal | obj,goal
        #  'sweep-into-v2', #obj,goal,hand | obj | obj
        #  'window-open-v2', #obj,hand | obj | obj
         ]

####################
"""Expert default all tasks"""
run_default = False
if run_default:
    for env_name in m10_tasks:
        print(env_name)
        reach_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name+"-goal-observable"]
        env = reach_goal_observable_cls(seed=0)
        print('max_path_length', env.max_path_length)
        print('TARGET_RADIUS', env.TARGET_RADIUS)
        env.random_init = False
        env._freeze_rand_vec = False
        expert_policy = functools.reduce(lambda a,b : a if a[0] == env_name else b, test_cases_latest_nonoise)[1]
        trajs, obj_poses, goal_poses = collect_data(env, expert_policy, num_trajs=1, mode='default')
        run_eval(trajs, True, 'default_expert', env_name, obj_poses=obj_poses, goal_poses=goal_poses)
    quit()
###################

#ENV
env_name = args.task_name
print(env_name, args)
reach_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name+"-goal-observable"]
env = reach_goal_observable_cls(seed=0)
# print('max_path_length', env.max_path_length)
env.random_init = False
env._freeze_rand_vec = False
vae_env = reach_goal_observable_cls(seed=0)
vae_env.random_init = True
vae_env._freeze_rand_vec = False
obs_size = env.observation_space.shape[0]
if args.reduced_state_space:
    obs_size = 6
ac_size = env.action_space.shape[0]
goal_size = env.goal_space.shape[0]
obj_size = len(env.obj_init_pos)

n_expert_traj = 100#0
n_vae_traj = 1000
n_eval_traj = 100
num_epochs_bc = 5000
num_epochs_vae = 50
num_epochs_res = 500
batch_size_bc = 50
batch_size_vae = 64
batch_size_res = 50
nplot = 20

#DEBUG
if args.debug:
    n_expert_traj = 10
    n_vae_traj = 2
    n_eval_traj = 2
    num_epochs_bc = 2
    num_epochs_vae = 2
    num_epochs_res = 2
    batch_size_bc = 1
    batch_size_vae = 1
    batch_size_res = 2
    nplot = 1


if not os.path.exists(os.path.join('figs',env_name)):
    os.mkdir(os.path.join('figs',env_name))

#EXPERT
print('EXPERT')
#for each task, create 1000 expert demos
expert_policy = functools.reduce(lambda a,b : a if a[0] == env_name else b, test_cases_latest_nonoise)[1]
trajs, traj_obj_poses, traj_goal_poses = collect_data(env, expert_policy, num_trajs=n_expert_traj, break_on_succ=args.expert_break_on_succ) #1000
# pdb.set_trace()
run_eval(trajs, render=False)
#render some expert demos
if args.render:
    for demo_id in random.sample(range(len(trajs)), k=5):
        sampled_pos = str(traj_goal_poses[demo_id]) if args.goal_ood else str(traj_obj_poses[demo_id])
        trajectory2vid(env, trajs[demo_id], 'expert_'+env_name+'_'+str(demo_id)+'_'+sampled_pos, env_name, gen_type, traj_obj_poses[demo_id], traj_goal_poses[demo_id])
#TODO check that traj consistent between collect_data and trajectory2vid [print out states, rendering has 1st frame issue]
# pdb.set_trace()
plot_3d_viz(trajs, os.path.join('figs',env_name,'expert3dviz.png'), nplot)


#BC
if not args.train_bc:
    policy = None
    _, bc_traj_obj_poses, bc_traj_goal_poses = collect_data(env, expert_policy, num_trajs=n_eval_traj, device=device) #100
    _, bc_ood_traj_obj_poses, bc_ood_traj_goal_poses = collect_data(env, expert_policy, num_trajs=n_eval_traj, mode='ood', device=device) #100
else:
    #train bc on train set
    hidden_layer_size = 1000
    hidden_depth = 3
    policy = Policy(obs_size, ac_size, hidden_layer_size, hidden_depth) # 10 dimensional latent
    policy.to(device)
    # Train standard goal conditioned policy
    EPS = 1e-9
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(policy.parameters()))
    losses = []
    idxs = np.array(range(len(trajs)))
    num_batches = len(idxs) // batch_size_bc
    losses = []
    # Train the model with regular SGD
    for epoch in range(num_epochs_bc):  # loop over the dataset multiple times
        np.random.shuffle(idxs)
        running_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()
            t1_idx = np.random.randint(len(trajs), size=(batch_size_bc,)) # Indices of first trajectory
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
    fig.savefig(os.path.join('figs',env_name,'bc_losses.png'))
    # print('Finished Training')

    #eval
    print('eval BC in dist')
    bc_trajs, bc_traj_obj_poses, bc_traj_goal_poses = collect_data(env, policy, num_trajs=n_eval_traj, device=device) #100
    #statistcs on train
    run_eval(bc_trajs, render=False)
    #render some demos with bc policy on train set
    if args.render:
        for demo_id in random.sample(range(len(bc_trajs)), k=5):
            sampled_pos = str(bc_traj_goal_poses[demo_id]) if args.goal_ood else str(bc_traj_obj_poses[demo_id])
            trajectory2vid(env, bc_trajs[demo_id], 'bc_test_'+str(demo_id)+'_'+sampled_pos, env_name, gen_type, bc_traj_obj_poses[demo_id], bc_traj_goal_poses[demo_id])
    plot_3d_viz(bc_trajs, os.path.join('figs',env_name,'bc3dviz.png'), nplot)

    #OOD eval
    print('eval BC ood')
    bc_ood_traj, bc_ood_traj_obj_poses, bc_ood_traj_goal_poses = collect_data(env, policy, num_trajs=n_eval_traj, mode='ood', device=device) #100
    #statistics on ood test
    run_eval(bc_ood_traj, render=False)
    #render some demos with bc policy on ood test set
    if args.render:
        for demo_id in random.sample(range(len(bc_ood_traj)), k=5):
            sampled_pos = str(bc_ood_traj_goal_poses[demo_id]) if args.goal_ood else str(bc_ood_traj_obj_poses[demo_id])
            trajectory2vid(env, bc_ood_traj[demo_id], 'bc_ood_'+str(demo_id)+'_'+sampled_pos, env_name, gen_type, bc_ood_traj_obj_poses[demo_id], bc_ood_traj_goal_poses[demo_id])
    # pdb.set_trace()
    plot_3d_viz(bc_ood_traj, os.path.join('figs',env_name,'bcood3dviz.png'), nplot)

#TODO should we use obs information or underlying obj/goal information for embedding?
#VAE
if not args.vae:
    vae = None
else:
    trajs_vae, obj_vae_poses, goal_vae_poses = collect_data(vae_env, expert_policy, num_trajs=n_vae_traj, mode='vae', device=device)
    #TODO for high dim state space what latent_dim/in_dim makes sense?
    latent_dim = 3
    in_dim = 3
    vae = VanillaVAE(in_dim=in_dim, latent_dim=latent_dim, hidden_dims=[64,64])
    vae.to(device)
    #TODO what are taking out of state? only goal pos? shouldn't be full state?
    if args.obj_ood:
        data = np.array(obj_vae_poses)
    elif args.goal_ood:
        data = np.array(goal_vae_poses)
    # pdb.set_trace()
    # VAE Training
    print('Training VAE')
    EPS = 1e-9
    optimizer = optim.Adam(vae.parameters())
    num_data = len(data)
    losses = []
    recon_losses = []
    kld_losses = []
    idxs = np.array(range(len(data)))
    num_batches = len(idxs) // batch_size_vae
    # Train the model with regular SGD
    for epoch in range(num_epochs_vae):  # loop over the dataset multiple times
        np.random.shuffle(idxs)
        running_loss = 0.0
        running_recon_loss = 0.0
        running_kl_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()
            curr_idx1 = idxs[batch_size_vae*i: batch_size_vae*i + batch_size_vae]
            x = torch.from_numpy(data[curr_idx1]).to(device).float()
            recon, input_val, mu, var = vae.forward(x)
            losses_all = vae.loss_function(recon, input_val, mu, var, **{'M_N': 10.*float(batch_size_vae)/float(num_data)})
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
    fig.savefig(os.path.join('figs',env_name,'vae_reconstruction.png'))
    #TODO visualize mu, if latent_dim==2 can use notebook.
    print('Finished Training VAE')

#RESIDUAL
print('Training RESIDUAL')
res_out_size = ac_size
if args.vae:
    res_in_size = ac_size + latent_dim
#TODO assumes only obj or goal is changing
elif args.goal_ood:
    res_in_size = ac_size + goal_size #action, delta
elif args.obj_ood:
    res_in_size = ac_size + obj_size #action, delta
if args.res_timestep:
    res_in_size += 1
if args.res_grasp_state:
    res_in_size += 1
elif args.res_full_state:
    res_in_size += obs_size #TODO can use part of state
hidden_layer_size = 32
hidden_depth = 1
residual_policy = Policy(res_in_size, res_out_size, hidden_layer_size, hidden_depth)
residual_policy.to(device)
EPS = 1e-9
criterion = nn.MSELoss()
opt_params = list(residual_policy.parameters())
optimizer = optim.Adam(opt_params)
losses = []
idxs = np.array(range(len(trajs)))
num_batches = len(idxs) // batch_size_res
losses = []
# Train the model with regular SGD
# pdb.set_trace()
for epoch in range(num_epochs_res):  # loop over the dataset multiple times
    np.random.shuffle(idxs)
    running_loss = 0.0
    for i in range(num_batches):
        optimizer.zero_grad()
        t1_idx = np.random.randint(len(trajs), size=(batch_size_res,)) # Indices of first trajectory
        t2_idx = np.random.randint(len(trajs), size=(batch_size_res,)) # Indices of second trajectory
        t1_idx_pertraj = np.array([np.random.randint(min(len(trajs[idx1]['obs']),len(trajs[idx2]['obs']))) for idx1,idx2 in zip(t1_idx,t2_idx)])
        t1_states = np.concatenate([trajs[c_idx]['obs'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
        t1_actions = np.concatenate([trajs[c_idx]['action'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
        t1_states = torch.Tensor(t1_states).float().to(device)
        t1_actions = torch.Tensor(t1_actions).float().to(device)
        if args.train_bc:
            a1_pred = policy(t1_states.to(device)) #first action prediction
        else:
            a1_pred = t1_actions
        t2_states = np.concatenate([trajs[c_idx]['obs'][t_idx][None] for (c_idx, t_idx) in zip(t2_idx, t1_idx_pertraj)])
        t2_actions = np.concatenate([trajs[c_idx]['action'][t_idx][None] for (c_idx, t_idx) in zip(t2_idx, t1_idx_pertraj)])
        t2_states = torch.Tensor(t2_states).float().to(device)
        t2_actions = torch.Tensor(t2_actions).float().to(device)
        if args.train_bc:
            a2_pred = policy(t2_states.to(device)).detach() #first action prediction
        else:
            a2_pred = t2_actions
        #goal as input to latent
        if args.goal_ood:
            g_delta = np.concatenate([get_latent(traj_goal_poses[t2_idx_diff],vae)[None] -
                                        get_latent(traj_goal_poses[t1_idx_diff],vae)[None]
                                        for (t1_idx_diff, t2_idx_diff) in zip(t1_idx, t2_idx)])
        elif args.obj_ood:
            g_delta = np.concatenate([get_latent(traj_obj_poses[t2_idx_diff],vae)[None] -
                                        get_latent(traj_obj_poses[t1_idx_diff],vae)[None]
                                        for (t1_idx_diff, t2_idx_diff) in zip(t1_idx, t2_idx)])
        g_delta_torch = torch.Tensor(g_delta).float().to(device)
        # Full action transform
        res_in = torch.cat([a1_pred.detach(), g_delta_torch], dim=-1)
        # pdb.set_trace()
        if args.res_timestep:
            res_in = torch.cat([res_in, torch.Tensor(t1_idx_pertraj).to(device).reshape(-1,1)], dim=-1)
        if args.res_grasp_state:
            res_in = torch.cat([res_in, t1_states[:,3].reshape(-1,1)], dim=-1)
        elif args.res_full_state:
            res_in = torch.cat([res_in, t1_states], dim=-1)
        residual_pred = residual_policy(res_in)
        a2_pred_residual = residual_pred
        # L2 regression on actions
        loss = torch.mean(torch.linalg.norm(a2_pred_residual - a2_pred, dim=-1))
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 1000000 == 0:    # print every 2000 mini-batches
            # print('[%d, %5d] loss: %.8f' %
            #       (epoch + 1, i + 1, running_loss / 10.))
            losses.append(running_loss/10.)
            running_loss = 0.0
        losses.append(loss.item())
#plot losses
fig = plt.figure()
plt.plot(losses)
fig.savefig(os.path.join('figs',env_name,'residual_losses.png'))
print('Finished Training RESIDUAL')


#RESIDUAL in dist eval
print('eval RES in dist')
if args.eval_res_best:
    res_traj = run_eval_residual_best(trajs, env, policy, residual_policy, train_obj=traj_obj_poses, train_goals=traj_goal_poses, ood_obj=bc_traj_obj_poses, ood_goals=bc_traj_goal_poses, vae=vae, device=device)
else:
    res_traj = run_eval_residual(trajs, env, policy, residual_policy, train_obj=traj_obj_poses, train_goals=traj_goal_poses, ood_obj=bc_traj_obj_poses, ood_goals=bc_traj_goal_poses, vae=vae, device=device)
if args.render:
    for demo_id in random.sample(range(len(res_traj)), k=5):
        sampled_pos = str(bc_traj_goal_poses[demo_id]) if args.goal_ood else str(bc_traj_obj_poses[demo_id])
        trajectory2vid(env, res_traj[demo_id], 'res_'+str(demo_id)+'_'+sampled_pos, env_name, gen_type, bc_traj_obj_poses[demo_id], bc_traj_goal_poses[demo_id])
# pdb.set_trace()
plot_3d_viz(res_traj, os.path.join('figs',env_name,'res3dviz.png'), nplot)

#RESIDUAL OOD eval
print('eval RES ood')
if args.eval_res_best:
    res_ood_traj = run_eval_residual_best(trajs, env, policy, residual_policy, train_obj=traj_obj_poses, train_goals=traj_goal_poses, ood_obj=bc_ood_traj_obj_poses, ood_goals=bc_ood_traj_goal_poses, vae=vae, device=device)
    plot_3d_viz(res_ood_traj, os.path.join('figs',env_name,'best_resood3dviz.png'), nplot)
res_ood_traj_reg = run_eval_residual(trajs, env, policy, residual_policy, train_obj=traj_obj_poses, train_goals=traj_goal_poses, ood_obj=bc_ood_traj_obj_poses, ood_goals=bc_ood_traj_goal_poses, vae=vae, device=device)
if args.render:
    for demo_id in random.sample(range(len(res_ood_traj)), k=5):
        sampled_pos = str(bc_ood_traj_goal_poses[demo_id]) if args.goal_ood else str(bc_ood_traj_obj_poses[demo_id])
        trajectory2vid(env, res_ood_traj[demo_id], 'res_ood_'+str(demo_id)+'_'+sampled_pos, env_name, gen_type, bc_ood_traj_obj_poses[demo_id], bc_ood_traj_goal_poses[demo_id])
plot_3d_viz(res_ood_traj_reg, os.path.join('figs',env_name,'reg_resood3dviz.png'), nplot)
# pdb.set_trace()