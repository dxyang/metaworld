from goal_env.robotics.fetch_env import FetchEnv
from goal_env.robotics.fetch.slide import FetchSlideEnv
from goal_env.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from goal_env.robotics.fetch.push import FetchPushEnv
from goal_env.robotics.fetch.reach import FetchReachEnv

from gym.envs.registration import registry, register, make, spec

for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''

    for mode in ['Near', 'Far', 'Left', 'Right', 'Train', 'Test']:

        # Keep this for backward compatability
        if mode == 'Train':
            goal_sampling_type = 'near'
        elif mode == 'Test':
            goal_sampling_type = 'far'
        else:
            goal_sampling_type = mode.lower()

        kwargs = {
            'reward_type': reward_type,
            'goal_sampling_type': goal_sampling_type,
        }

        # Fetch
        register(
            id='FetchSlide{}{}-v1'.format(mode, suffix),
            entry_point='goal_env.robotics:FetchSlideEnv',
            kwargs={**kwargs, 'far_goal_threshold': 0.3},
            max_episode_steps=50,
        )

        register(
            id='FetchPickAndPlace{}{}-v1'.format(mode, suffix),
            entry_point='goal_env.robotics:FetchPickAndPlaceEnv',
            kwargs={**kwargs, 'far_goal_threshold': 0.1},
            max_episode_steps=50,
        )

        register(
            id='FetchReach{}{}-v1'.format(mode, suffix),
            entry_point='goal_env.robotics:FetchReachEnv',
            kwargs={**kwargs, 'far_goal_threshold': 0.1},
            max_episode_steps=50,
        )

        register(
            id='FetchPush{}{}-v1'.format(mode, suffix),
            entry_point='goal_env.robotics:FetchPushEnv',
            kwargs={**kwargs, 'far_goal_threshold': 0.25},
            max_episode_steps=50,
        )
