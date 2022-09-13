import pdb
import os
import cv2
import numpy as np


# metaworld/scripts/scripted_policy_movies.ipynb
# README -- Accessing Single Goal Environments

# def writer_for(tag, fps, res, env_name, gen_type):
#     movie_dir = "movies_res1"
#     if not os.path.exists(movie_dir):
#         os.mkdir(movie_dir)
#     env_path = os.path.join(movie_dir,env_name, gen_type)
#     if not os.path.exists(env_path):
#         os.makedirs(env_path)
#     return cv2.VideoWriter(
#         f'{env_path}/{tag}.avi',
#         cv2.VideoWriter_fourcc('M','J','P','G'),
#         fps,
#         res
#     )

# def trajectory2vid(env, traj, tag, env_name, gen_type="", obj_pos=None, goal_pos=None):
#     # print('rendering ', tag)
#     resolution = (1920, 1080)
#     camera = 'corner' # one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
#     flip=False # if True, flips output image 180 degrees
#     if 'peg-insert-side-v2'==env_name:
#         #TODO
#         camera = 'behindGripper'
#         flip = True

#     #TODO need to use same scheme as in test_bc, reset does something?
#     env.reset()
#     env.random_init = False
#     #None -- default values.
#     if goal_pos is not None:
#         env.goal = goal_pos #traj['obs'][-1][-3:] -- obs doesn't have inner values updated
#     if obj_pos is not None:
#         env.init_config['obj_init_pos'] = obj_pos #traj['obs'][0][4:7]
#         env.obj_init_pos = env.init_config['obj_init_pos']
#     o = env.reset_model()
#     # env.sim.forward()
#     # o = env._get_obs()

#     writer = writer_for(tag, env.metadata['video.frames_per_second'], resolution, env_name, gen_type)
#     for st in range(len(traj['obs'])):
#         #TODO right place to generate an image
#         # img = env.sim.render(*resolution, mode='offscreen', camera_name=camera)[:,:,::-1]
#         # pdb.set_trace()
#         no, r, _, info = env.step(traj['action'][st])
#         img = env.sim.render(*resolution, mode='offscreen', camera_name=camera)[:,:,::-1]
#         if flip: img = cv2.rotate(img, cv2.ROTATE_180)
#         writer.write(img)
#         if traj['done'][st]:
#             break

#     writer.release()

#     #TODO return trajectory



def writer_for(log_dir, tag, fps, res):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    env_path = os.path.join(log_dir, tag)
    return cv2.VideoWriter(
        f'{env_path}/{tag}.avi',
        cv2.VideoWriter_fourcc('M','J','P','G'),
        fps,
        res
    )

def trajectory2vid(env, traj, tag, env_name):
    goal_size = 3
    resolution = (1920, 1080)
    camera = 'corner' # one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
    flip=False # if True, flips output image 180 degrees
    log_dir = os.path.join()
    writer = writer_for(tag, env.metadata['video.frames_per_second'], resolution, env_name, gen_type)    
    env.reset()
    if env_name in ['reach_v2', 'push_v2']: #metaworld
        goal_pos = traj[0]['obs'][-1,-goal_size:]
        o = env.reset_model(goal_pos=goal_pos)
        for st in range(len(traj['obs'])):
            no, r, _, info = env.step(traj['action'][st])
            img = env.sim.render(*resolution, mode='offscreen', camera_name=camera)[:,:,::-1]
            if flip: img = cv2.rotate(img, cv2.ROTATE_180)
            writer.write(img)
            if traj['done'][st]:
                break        
    else: #franka reach / dubins
        goal_pos = traj[0]['observations'][-1,-goal_size:]
        o = env.set_goal(goal_pos)
        for st in range(len(traj['observations'])):
            no, r, _, info = env.step(traj['actions'][st])
            img = env.sim.render(*resolution, mode='offscreen', camera_name=camera)[:,:,::-1]
            if flip: img = cv2.rotate(img, cv2.ROTATE_180)
            writer.write(img)
            if traj['dones'][st]:
                break
    writer.release()