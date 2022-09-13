import pdb
import os
import functools
import cv2
import numpy as np

from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
                            # these are ordered dicts where the key : value
                            # is env_name : env_constructor
from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import ALL_ENVS, test_cases_latest_nonoise



reach_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["reach-v2-goal-observable"]
env = reach_goal_observable_cls(seed=0)
env.reset()  # Reset environment
a = env.action_space.sample()  # Sample an action
obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
assert not (obs[-3:] == np.zeros(3)).all()   # The env's are goal observable, meaning the goal is not zero'd out

pdb.set_trace()


def trajectory_generator(env, policy, act_noise_pct, res=(640, 480), camera='corner'):
    action_space_ptp = env.action_space.high - env.action_space.low

    env.reset()
    env.reset_model()
    o = env.reset()
    # pdb.set_trace()

    for _ in range(env.max_path_length):
        a = policy.get_action(o)
        a = np.random.normal(a, act_noise_pct * action_space_ptp)

        o, r, done, info = env.step(a)
        # Camera is one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
        yield r, done, info, env.sim.render(*res, mode='offscreen', camera_name=camera)[:,:,::-1]

def writer_for(tag, fps, res):
    if not os.path.exists('movies'):
        os.mkdir('movies')
    return cv2.VideoWriter(
        f'movies/{tag}.avi',
        cv2.VideoWriter_fourcc('M','J','P','G'),
        fps,
        res
    )

resolution = (1920, 1080)
camera = 'corner' # one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
flip=False # if True, flips output image 180 degrees

config = [
    # env, action noise pct, cycles, quit on success
    ('reach-v2', np.zeros(4), 1, True),
]

for env, noise, cycles, quit_on_success in config:
    tag = env + '-noise-' + np.array2string(noise, precision=2, separator=',', suppress_small=True)

    policy = functools.reduce(lambda a,b : a if a[0] == env else b, test_cases_latest_nonoise)[1]
    env = ALL_ENVS[env]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    writer = writer_for(tag, env.metadata['video.frames_per_second'], resolution)
    for _ in range(cycles):
        for r, done, info, img in trajectory_generator(env, policy, noise, resolution, camera):
            if flip: img = cv2.rotate(img, cv2.ROTATE_180)
            writer.write(img)
            if quit_on_success and info['success']:
                break

    writer.release()

