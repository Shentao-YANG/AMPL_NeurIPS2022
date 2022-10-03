import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_models.parallelized_ensemble import ParallelizedEnsemble
from utils.utils import print_banner
from copy import deepcopy


class QDiscriminatorProbabilisticEnsemble(ParallelizedEnsemble):

    def __init__(
            self,
            env_name,
            ensemble_size,        # Number of members in ensemble
            obs_dim,              # Observation dim of environment
            action_dim,           # Action dim of environment
            hidden_sizes,         # Hidden sizes for each model
            device,
            actor,
            critic,
            spectral_norm=False,  # Apply spectral norm to every hidden layer
            obs_range=10.,
            separate_mean_var=True,
            **kwargs
    ):
        super().__init__(
            env_name=env_name,
            ensemble_size=ensemble_size,
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=(obs_dim + 1) if separate_mean_var else 2 * (obs_dim + 1),      # We predict (reward, next_state - state)
            device=device,
            hidden_activation=nn.SiLU(),
            spectral_norm=spectral_norm,
            separate_mean_var=separate_mean_var,
            obs_range=obs_range,
            **kwargs
        )

        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.output_size = obs_dim + 1
        self.device = device
        self.separate_mean_var = separate_mean_var
        self.hidden_sizes = hidden_sizes

        self.max_logstd = nn.Parameter(
            torch.ones((1, obs_dim + 1), device=device) * (1. / 4.), requires_grad=True)
        self.min_logstd = nn.Parameter(
            -torch.ones((1, obs_dim + 1), device=device) * 5., requires_grad=True)

        self.actor = actor
        self.critic = critic

        print_banner(f"Initialize QDiscriminatorProbabilisticEnsemble with ensemble_size={ensemble_size}, hidden_sizes={hidden_sizes}, spectral_norm={spectral_norm}, separate_mean_var={self.separate_mean_var}")

    def forward(self, inputs, deterministic=False, return_dist=False):
        output = super().forward(inputs)
        if not self.separate_mean_var:
            mean, logstd = torch.chunk(output, 2, dim=-1)
        else:       # separate_mean_var
            mean, logstd = output

        # Variance clamping to prevent poor numerical predictions
        logstd = self.max_logstd - F.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + F.softplus(logstd - self.min_logstd)

        if deterministic:
            return mean, logstd if return_dist else mean

        std = torch.exp(logstd)
        eps = torch.randn(std.shape, device=self.device)
        samples = mean + std * eps

        if return_dist:
            return samples, mean, logstd
        else:
            return samples

    def get_loss(self, x, y, weight, split_by_model=False, return_l2_error=False, first_training=False):
        if first_training:
            # first time training use MLE
            return self._get_loss_mle(x=x, y=y, weight=weight, split_by_model=split_by_model, return_l2_error=return_l2_error)
        else:
            # after first time, use q-discriminated loss
            return self._get_loss_q_discriminate(x=x, y=y, weight=weight, split_by_model=split_by_model, return_l2_error=return_l2_error)

    def _get_loss_q_discriminate(self, x, y, weight, split_by_model=False, return_l2_error=False):
        # Note: we assume y here already accounts for the delta of the next state
        # weight: (ensemble_size, self.batch_size)
        # use critic.q1, sample one action at each s'

        mean, logstd = self.forward(x, deterministic=True, return_dist=True)    # (self.ensemble_size, self.batch_size, obs_dim + 1)
        if len(y.shape) < 3:
            y = y.unsqueeze(0).repeat(self.ensemble_size, 1, 1)

        mean_rns = mean[..., self.rns_torch]
        y_rns = y[..., self.rns_torch]          # (self.ensemble_size, self.batch_size, -1)

        # get normalized delta_s
        delta_s_sample = mean[..., 1:] + torch.exp(logstd[..., 1:]) * torch.randn(logstd[..., 1:].shape, device=self.device)
        # get unnormalized delta_s
        delta_s_sample = delta_s_sample * self.delta_obs_std.data[..., 1:] + self.delta_obs_mu.data[..., 1:]
        # get next_s = s + delta_s
        next_s_sample = x[..., :self.obs_dim] + delta_s_sample      # (self.ensemble_size, self.batch_size, obs_dim)
        next_a_sample = self.actor(next_s_sample).detach()          # (self.ensemble_size, self.batch_size, action_dim)

        # get unnormalized true_next_s_delta
        true_next_s = y[..., 1:] * self.delta_obs_std.data[..., 1:] + self.delta_obs_mu.data[..., 1:]
        # get true_next_s
        true_next_s = x[..., :self.obs_dim] + true_next_s           # (self.ensemble_size, self.batch_size, obs_dim)
        next_a_true_s = self.actor(true_next_s).detach()            # (self.ensemble_size, self.batch_size, action_dim)

        Q_fake = self.critic.q1(next_s_sample, next_a_sample)       # (self.ensemble_size, self.batch_size, 1)
        Q_true = self.critic.q1(true_next_s, next_a_true_s)         # (self.ensemble_size, self.batch_size, 1)

        # Maximize log-probability of reward + Q-discriminated next state
        sq_l2_error = (mean_rns - y_rns) ** 2                       # (self.ensemble_size, self.batch_size, -1)
        if return_l2_error:
            l2_error = (sq_l2_error.mean(dim=-1) * weight).mean(dim=-1)

        inv_var = torch.exp(-2 * logstd[..., :1])                   # (self.ensemble_size, self.batch_size, 1), only for reward
        loss = (sq_l2_error[..., :1] * inv_var + 2 * logstd[..., :1]).mean(dim=-1) + (Q_true - Q_fake).abs().mean(dim=-1)       # (self.ensemble_size, self.batch_size)
        loss = (loss * weight).mean(dim=-1)                         # (self.ensemble_size)

        if split_by_model:
            losses = [loss[i] for i in range(self.ensemble_size)]
            if return_l2_error:
                l2_errors = [l2_error[i] for i in range(self.ensemble_size)]
                return losses, l2_errors
            else:
                return losses
        else:   # return the loss sum over all the ensemble models
            clipping_bound_loss = 0.01 * (self.max_logstd * 2).sum() - 0.01 * (self.min_logstd * 2).sum()
            if return_l2_error:
                return loss.sum() + clipping_bound_loss, l2_error.sum()
            else:
                return loss.sum() + clipping_bound_loss

    def _get_loss_mle(self, x, y, weight, split_by_model=False, return_l2_error=False):
        # Note: we assume y here already accounts for the delta of the next state
        # weight: (ensemble_size, self.batch_size)

        mean, logstd = self.forward(x, deterministic=True, return_dist=True)
        if len(y.shape) < 3:
            y = y.unsqueeze(0).repeat(self.ensemble_size, 1, 1)

        mean_rns = mean[..., self.rns_torch]
        y_rns = y[..., self.rns_torch]          # (self.ensemble_size, self.batch_size, -1)

        # Maximize log-probability of transitions
        inv_var = torch.exp(-2 * logstd)
        sq_l2_error = (mean_rns - y_rns) ** 2   # (self.ensemble_size, self.batch_size, -1)
        if return_l2_error:
            l2_error = (sq_l2_error.mean(dim=-1) * weight).mean(dim=-1)

        loss = ((sq_l2_error * inv_var + 2 * logstd).mean(dim=-1) * weight).mean(dim=-1)

        if split_by_model:
            losses = [loss[i] for i in range(self.ensemble_size)]
            if return_l2_error:
                l2_errors = [l2_error[i] for i in range(self.ensemble_size)]
                return losses, l2_errors
            else:
                return losses
        else:   # return the loss sum over all the ensemble models
            clipping_bound_loss = 0.01 * (self.max_logstd * 2).sum() - 0.01 * (self.min_logstd * 2).sum()
            if return_l2_error:
                return loss.sum() + clipping_bound_loss, l2_error.sum()
            else:
                return loss.sum() + clipping_bound_loss

    @property
    def num_hidden_layers(self):
        return len(self.hidden_sizes)

    def get_idv_model_state(self, idx):
        params = [{"W": deepcopy(self.fcs[i].W.data[idx]), "b": deepcopy(self.fcs[i].b.data[idx])} for i in range(self.num_hidden_layers)]
        params.append({"W": deepcopy(self.last_fc.W.data[idx]), "b": deepcopy(self.last_fc.b.data[idx])})
        if self.separate_mean_var:
            params.append({"W": deepcopy(self.last_fc_std.W.data[idx]), 'b': deepcopy(self.last_fc_std.b.data[idx])})

        return params

    def load_model_state_from_dict(self, state_dict):
        num_hidden_layers = self.num_hidden_layers
        for model in range(self.ensemble_size):
            model_params = state_dict[model]
            for i in range(num_hidden_layers):
                self.fcs[i].W.data[model].copy_(model_params[i]["W"])
                self.fcs[i].b.data[model].copy_(model_params[i]['b'])
            self.last_fc.W.data[model].copy_(model_params[num_hidden_layers]["W"])
            self.last_fc.b.data[model].copy_(model_params[num_hidden_layers]['b'])
            if self.separate_mean_var:
                self.last_fc_std.W.data[model].copy_(model_params[num_hidden_layers+1]["W"])
                self.last_fc_std.b.data[model].copy_(model_params[num_hidden_layers+1]["b"])
