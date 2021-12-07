from gym.envs.mujoco import HalfCheetahEnv, HopperEnv, Walker2dEnv, AntEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, RAILGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import torch
import gpytorch
import argparse
import numpy as np
import random
import mj_envs

import gym
from gp_models import MultitaskGPModel


def experiment(variant, expl_env, eval_env, policy_type='tanh_gaussian'):
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    M = variant['layer_size']

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    if policy_type == 'tanh_gaussian':
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M] * 2,
        )
    elif policy_type == 'rail_gaussian':
        policy = RAILGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M] * 4,
            std_architecture="shared",
            gp_m=variant['trainer_kwargs']['gp_model'],
            gp_l=variant['trainer_kwargs']['gp_likelihood'],
        )
    else:
        raise SystemExit('Error: Invalid policy type.')

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

    return policy


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='Name of the environment to be run on', type=str)
    parser.add_argument('--num_epochs', help='Number of training epochs', type=int, default=3000)
    parser.add_argument('--rb_size', help='Size of the replay buffer', type=int, default=int(1E6))
    parser.add_argument('--label', help='Label of the experiment', type=str, default='null')
    parser.add_argument('--use_gpu', action='store_true', help='Enable GPU')
    parser.add_argument('--use_fixed_alpha', action='store_true', help='Disable automatic entropy tuning')
    parser.add_argument('--alpha', help='Fixed alpha value', type=float, default=1.0)
    # GP Arguments
    parser.add_argument('--use_gp', action='store_true', help='Switch to Gaussian Process')
    parser.add_argument('--model_file', help='Model for Gaussian Process', type=str)
    parser.add_argument('--data_file', help='Training data for Gaussian Process', type=str)
    parser.add_argument('--gp_rank', help='Rank of the task covar module', type=int, default=1)
    parser.add_argument('--kernel_type', help='Kernel for the GP', type=str, default='matern12')
    # Training Arguments
    parser.add_argument('--policy_type', help='Type of policy used', type=str, default='tanh_gaussian')
    parser.add_argument('--batch_size', help='Size of the training batch', type=int, default=256)
    parser.add_argument('--eval_per_epoch', help='Evaluation steps per epoch', type=int, default=5000)
    parser.add_argument('--pretrain_policy', help='Number of epochs to pre-train the policy', type=int, default=0)
    parser.add_argument('--snapshot_mode', help='Log snapshot mode', type=str, default='none')
    args = parser.parse_args()

    # Seeding
    random_seed = random.SystemRandom().randint(int(1e6), int(1e9))
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    name = args.name

    if name == 'half_cheetah':
        expl_env = NormalizedBoxEnv(HalfCheetahEnv())
        eval_env = NormalizedBoxEnv(HalfCheetahEnv())
    elif name == 'hopper':
        expl_env = NormalizedBoxEnv(HopperEnv())
        eval_env = NormalizedBoxEnv(HopperEnv())
    elif name == 'walker':
        expl_env = NormalizedBoxEnv(Walker2dEnv())
        eval_env = NormalizedBoxEnv(Walker2dEnv())
    elif name == 'ant':
        expl_env = NormalizedBoxEnv(AntEnv())
        eval_env = NormalizedBoxEnv(AntEnv())
    else:
        expl_env = gym.make(name)
        eval_env = gym.make(name)

    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        #
        environment_name=name,
        seed=random_seed,
        replay_buffer_size=args.rb_size,
        #
        algorithm_kwargs=dict(
            num_epochs=args.num_epochs,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=args.batch_size,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=not args.use_fixed_alpha,
            fixed_alpha=args.alpha,
            use_gp=args.use_gp,
            gp_model=None,
            gp_likelihood=None,
            pretrain_policy=args.pretrain_policy,
        ),
    )

    ptu.set_gpu_mode(args.use_gpu)  # optionally set the GPU (default=False)
    setup_logger(args.label, variant=variant, snapshot_mode=args.snapshot_mode)

    action_dim = expl_env.action_space.low.size
    obs_dim = expl_env.observation_space.low.size

    if args.use_gp:
        data = np.load(args.data_file, allow_pickle=True)

        if type(data[0]['observations'][0]) is dict:
            for traj in data:
                traj['observations'] = [t['state_observation'] for t in traj['observations']]

        train_x = torch.from_numpy(
            np.array([j for i in [traj['observations'] for traj in data] for j in i])).float().to(
            ptu.device)
        train_y = torch.from_numpy(np.array([j for i in [traj['actions'] for traj in data] for j in i])).float().to(
            ptu.device)

        # Initialize likelihood and model
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=action_dim).to(ptu.device)

        model = MultitaskGPModel(train_x, train_y, likelihood, num_tasks=action_dim, rank=args.gp_rank,
                                 ard_num_dims=obs_dim, kernel_type=args.kernel_type).to(ptu.device)

        model.load_state_dict(torch.load('nppac/{}'.format(args.model_file)))
        model.eval()
        likelihood.eval()

        variant['trainer_kwargs']['gp_model'] = model
        variant['trainer_kwargs']['gp_likelihood'] = likelihood

    # Run experiments
    returned_policy = experiment(variant, expl_env=expl_env, eval_env=eval_env, policy_type=args.policy_type)
