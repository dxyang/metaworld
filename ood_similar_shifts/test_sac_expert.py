import gym
import numpy as np
import pdb

# ###################
# # stablebaselines
# #https://stable-baselines3.readthedocs.io/en/master/modules/sac.html

# from stable_baselines3 import SAC

# env = gym.make("Pendulum-v1")

# model = SAC("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000, log_interval=4)
# model.save("sac_pendulum")

# del model # remove to demonstrate saving and loading

# model = SAC.load("sac_pendulum")

# # obs = env.reset()
# # while True:
# #     action, _states = model.predict(obs, deterministic=True)
# #     obs, reward, done, info = env.step(action)
# #     env.render()
# #     if done:
# #       obs = env.reset()
# #####################



######################
# rlkit  
# https://github.com/rail-berkeley/rlkit/blob/master/examples/sac.py

# from gym.envs.mujoco import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerReachEnvV2


def experiment(variant):

    env_name = 'reach-v2'
    # reach_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name+"-goal-observable"]
    # env = reach_goal_observable_cls(seed=0)

    seed = 0
    st0 = np.random.get_state()
    np.random.seed(seed)
    env1 = SawyerReachEnvV2()
    env1._freeze_rand_vec = False
    env1._set_task_called = True
    env1.reset()
    env1._freeze_rand_vec = True 
    #  
    env2 = SawyerReachEnvV2()
    env2._freeze_rand_vec = False
    env2._set_task_called = True
    env2.reset()
    env2._freeze_rand_vec = True     
    np.random.set_state(st0) 

    expl_env = NormalizedBoxEnv(env1)
    eval_env = NormalizedBoxEnv(env2)
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()
    return algorithm




if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger('reachv2-SAC', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    algorithm = experiment(variant)
    pdb.set_trace()
######################

