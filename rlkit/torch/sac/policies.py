import numpy as np
import torch
from torch import nn as nn

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.core import eval_np
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.networks import Mlp
import gpytorch

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

from torch.distributions import Normal
import rlkit.torch.pytorch_util as ptu


class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
            use_gp=False,
            gp_model=None,
            gp_likelihood=None,
    ):
        """
        :param prior: BC Prior
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std
        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None

        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            entropy = tanh_normal.normal.entropy()
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )

                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value,
                )

                if use_gp:
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        observed_pred = gp_likelihood(gp_model(obs))
                        mean = observed_pred.mean.detach()
                        std = observed_pred.variance.detach()

                    nm = Normal(mean, std)
                    log_prob = log_prob - nm.log_prob(action)

                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()
        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )


class RAILGaussianPolicy(Mlp, ExplorationPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            std_architecture="shared",
            gp_m=None,
            gp_l=None,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            output_activation=torch.tanh,
            **kwargs
        )
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            if self.std_architecture == "shared":
                last_hidden_size = obs_dim
                if len(hidden_sizes) > 0:
                    last_hidden_size = hidden_sizes[-1]

                self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

        self.gp_m = gp_m
        self.gp_l = gp_l

        if self.gp_m is not None:
            self.gp_m.eval()
            self.gp_l.eval()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("gp_m", None)
        state.pop("gp_l", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.gp_m = None
        self.gp_l = None

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic, gp_model=self.gp_m, gp_likelihood=self.gp_l)[0]

    def forward(self,
                obs,
                reparameterize=True,
                deterministic=False,
                return_log_prob=False,
                use_gp=False,
                gp_model=None,
                gp_likelihood=None,
                ):
        if gp_model is not None:
            gp_model.eval()
            gp_likelihood.eval()
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))

        preactivation = self.last_fc(h)
        mean = self.output_activation(preactivation)
        if self.std is None:
            if self.std_architecture == "shared":
                log_std = self.last_fc_log_std(h)
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)

            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            log_std = None
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)
        log_prob = None
        entropy = None
        mean_action_log_prob = None

        if deterministic:
            action = mean
        else:
            normal = Normal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action = normal.rsample()
                else:
                    action = normal.sample()
                log_prob = normal.log_prob(action)

                if use_gp:
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        observed_pred = gp_likelihood(gp_model(obs))
                    std = observed_pred.variance.detach()
                    mean = observed_pred.mean.detach()

                    nm = Normal(mean, std)
                    log_prob = log_prob - nm.log_prob(action)

                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = normal.rsample()
                else:
                    action = normal.sample()

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob,
        )


class MakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)
