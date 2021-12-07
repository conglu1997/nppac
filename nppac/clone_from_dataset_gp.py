from gym.envs.mujoco import HalfCheetahEnv, HopperEnv, Walker2dEnv, AntEnv

from rlkit.envs.wrappers import NormalizedBoxEnv
import rlkit.torch.pytorch_util as ptu
import torch

import argparse
import numpy as np
import random

import gpytorch
import datetime
import gym
import mj_envs
import os

from scipy.stats import norm
from gp_models import MultitaskGPModel, VariMultitaskGPModel, BatchIndependentMultitaskGPModel


def rollout(
        env,
        model,
        likelihood,
        max_path_length=np.inf,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next element will be a list of dictionaries, with the index into
    the list being the index into the time
     - env_infos
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        while path_length < max_path_length:
            o_torch = torch.from_numpy(np.array([o])).float().to(ptu.device)
            observed_pred = likelihood(model(o_torch))
            a = observed_pred.mean.data.cpu().numpy()

            if len(a) == 1:
                a = a[0]

            next_o, r, d, env_info = env.step(a)

            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            env_infos.append(env_info)
            path_length += 1
            if d:
                break
            o = next_o

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        env_infos=env_infos,
    )


def collect_new_paths(
        env,
        model,
        likelihood,
        max_path_length,
        num_steps,
        discard_incomplete_paths,
):
    paths = []
    num_steps_collected = 0
    while num_steps_collected < num_steps:
        max_path_length_this_loop = min(  # Do not go over num_steps
            max_path_length,
            num_steps - num_steps_collected,
        )

        path = rollout(
            env,
            model,
            likelihood,
            max_path_length=max_path_length_this_loop,
        )

        path_len = len(path['actions'])
        if (
                path_len != max_path_length
                and not path['terminals'][-1]
                and discard_incomplete_paths
        ):
            break
        num_steps_collected += path_len
        paths.append(path)
    return paths


def batch_assess(model, likelihood, X, Y):
    lik, sq_diff = [], []

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X))
        m = observed_pred.mean
        v = observed_pred.variance
    avg_v = torch.mean(v)

    Y = Y.cpu().data.numpy()
    m = m.cpu().data.numpy()
    v = v.cpu().data.numpy()

    l = np.sum(norm.logpdf(Y, loc=m, scale=v ** 0.5), 1)
    sq = ((m - Y) ** 2)

    lik.append(l)
    sq_diff.append(sq)

    lik = np.concatenate(lik, 0)
    sq_diff = np.array(np.concatenate(sq_diff, 0), dtype=float)
    return np.average(lik), np.average(sq_diff) ** 0.5, avg_v


if __name__ == "__main__":
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='half_cheetah')
    parser.add_argument('--data_set',
                        help='Dataset used to generate state-action pairs',
                        type=str)
    parser.add_argument('--training_epochs',
                        help='Number of training epochs',
                        type=int,
                        default=1000)
    parser.add_argument('--evaluation_interval',
                        help='Evaluation interval',
                        type=int,
                        default=20)
    parser.add_argument('--n_test_episodes',
                        help='Number of test episodes',
                        type=int,
                        default=10)
    parser.add_argument('--save_policies', action='store_true', help='Save policy files')
    parser.add_argument('--use_gpu', action='store_true', help='Enable GPU')
    parser.add_argument('--use_ard', action='store_true', help='Use ARD')
    parser.add_argument('--kernel_type', help='Kernel for the GP', type=str, default='matern12')
    parser.add_argument('--gp_rank',
                        help='Rank of task covar module',
                        type=int,
                        default=1)
    parser.add_argument('--save_dir',
                        help='Directory to save in',
                        type=str,
                        default='gp_model_hand_any')
    parser.add_argument('--gp_type',
                        help='Type of GP',
                        type=str,
                        default='multitask')
    parser.add_argument('--keep_num',
                        help='What #traj to keep, by default has no effect.',
                        type=int,
                        default=1000)

    args = parser.parse_args()
    name = args.name
    epochs = args.training_epochs
    saving_policies = args.save_policies

    if not os.path.exists(f'nppac/{args.save_dir}'):
        os.makedirs(f'nppac/{args.save_dir}')

    ptu.set_gpu_mode(args.use_gpu)  # optionally set the GPU (default=False)

    # Seeding
    random_seed = random.SystemRandom().randint(1e6, 1e9)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if name == 'half_cheetah':
        env = NormalizedBoxEnv(HalfCheetahEnv())
    elif name == 'hopper':
        env = NormalizedBoxEnv(HopperEnv())
    elif name == 'walker':
        env = NormalizedBoxEnv(Walker2dEnv())
    elif name == 'ant':
        env = NormalizedBoxEnv(AntEnv())
    elif name in ['door', 'hammer', 'pen', 'relocate']:
        env = gym.make(f'{name}-binary-v0')
    else:
        raise SystemExit('Error: Invalid name.')

    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    data = np.load(args.data_set, allow_pickle=True)

    # Ablation to randomly filter the dataset, not active by default.
    if args.keep_num < len(data):
        print(f'Keeping {args.keep_num} trajectories.')
        data = np.random.choice(data, args.keep_num, replace=False)

        with open(f'nppac/{args.save_dir}/data.npy', 'wb') as f:
            np.save(f, data)

    if type(data[0]['observations'][0]) is dict:
        # Convert to just the states
        for traj in data:
            traj['observations'] = [t['state_observation'] for t in traj['observations']]

    train_x = torch.from_numpy(np.array([j for i in [traj['observations'] for traj in data] for j in i])).float().to(
        ptu.device)
    train_y = torch.from_numpy(np.array([j for i in [traj['actions'] for traj in data] for j in i])).float().to(
        ptu.device)

    print('Data Loaded!')

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=action_dim).to(ptu.device)

    ard_num_dims = obs_dim if args.use_ard else None

    # Currently using action dim as rank, ignoring the argument
    if args.gp_type == 'multitask':
        model = MultitaskGPModel(train_x, train_y, likelihood, num_tasks=action_dim, rank=args.gp_rank,
                                 ard_num_dims=ard_num_dims, kernel_type=args.kernel_type).to(ptu.device)
    elif args.gp_type == 'batch_ind':
        model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood, num_tasks=action_dim,
                                                 ard_num_dims=ard_num_dims, kernel_type=args.kernel_type).to(ptu.device)
    elif args.gp_type == 'varimulti':
        num_inducing_points = 200
        inducing_points = torch.rand(action_dim, num_inducing_points, obs_dim)
        model = VariMultitaskGPModel(inducing_points, num_tasks=action_dim, kernel_type=args.kernel_type).to(ptu.device)
    else:
        raise SystemExit('Invalid GP type.')

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    if args.gp_type == 'varimulti':
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y))
    else:
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for epoch in range(0, epochs + 1):
        model.train()
        likelihood.train()

        if args.gp_type == 'varimulti':
            # Training on epoch 0 but whatever
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
            trainl, trains, avg_v = batch_assess(model, likelihood, train_x[:600], train_y[:600])
        else:
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            if epoch > 0:
                loss.backward()
                optimizer.step()

            trainl, trains, avg_v = batch_assess(model, likelihood, train_x, train_y)

        print('Iter %d/%d - Loss: %.3f, Train mean log likelihood: %.3f, Train RMSE: %.3f' % (
            epoch, epochs, loss.item(), trainl, trains
        ))

        if epoch % args.evaluation_interval == 0:
            model.eval()
            likelihood.eval()

            max_path_length = 1000

            start = datetime.datetime.now()

            ps = collect_new_paths(
                env,
                model,
                likelihood,
                max_path_length,
                max_path_length * args.n_test_episodes,
                discard_incomplete_paths=True,
            )

            finish = datetime.datetime.now()
            print("Profiling took: ", finish - start)

            eval_rew = np.mean([np.sum(p['rewards']) for p in ps])
            eval_std = np.std([np.sum(p['rewards']) for p in ps])
            print(f'Epoch {epoch}, Offline Return: {eval_rew}, Std: {eval_std}')

            if saving_policies:
                torch.save(model.state_dict(),
                           f'nppac/{args.save_dir}/gp_{name}_{args.gp_type}_{epoch}.pt')
