import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from utils.utils import print_banner
from dynamic_models.env_termimation_function import set_termination


def identity(x):
    return x


class ParallelizedLayer(nn.Module):

    def __init__(
        self,
        ensemble_size,
        input_dim,
        output_dim,
        device,
        w_std_value=1.0,
        b_init_value=0.0,
    ):
        super().__init__()

        # approximation to truncated normal of 2 stds
        w_init = torch.randn((ensemble_size, input_dim, output_dim), device=device)
        w_init = torch.fmod(w_init, 2) * w_std_value
        self.W = nn.Parameter(w_init, requires_grad=True)

        # constant initialization
        b_init = torch.zeros((ensemble_size, 1, output_dim), device=device).float()
        b_init += b_init_value
        self.b = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        # output dim is: (ensemble_size, batch_size, output_dim)
        return x @ self.W + self.b


class ParallelizedEnsemble(nn.Module):

    def __init__(
            self,
            env_name,
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            device,
            hidden_activation=F.relu,
            output_activation=identity,
            b_init_value=0.0,
            spectral_norm=False,
            separate_mean_var=False,
            obs_range=10.
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.elites = [i for i in range(self.ensemble_size)]

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.device = device
        self.separate_mean_var = separate_mean_var
        self.env_name = env_name
        self.original_termination = set_termination(self.env_name)
        self.obs_range = torch.tensor(obs_range, device=self.device).float()
        self.reward_range = torch.tensor(50., device=self.device)

        print_banner(f"Initialize ParallelizedEnsemble with env_name={self.env_name}, ensemble_size={ensemble_size}, hidden_sizes={hidden_sizes}, spectral_norm={spectral_norm}, separate_mean_var={self.separate_mean_var}, obs_range={obs_range}")

        # data normalization
        self.input_mu = nn.Parameter(
            torch.zeros(input_size, device=device), requires_grad=False).float()
        self.input_std = nn.Parameter(
            torch.ones(input_size, device=device), requires_grad=False).float()

        obs_dim = self.output_size - 1 if self.separate_mean_var else self.output_size // 2 - 1
        self.rns_np = np.array([True] + [True] * int(obs_dim))
        self.rns_torch = torch.from_numpy(self.rns_np).to(self.device)

        self.delta_obs_mu = nn.Parameter(
            torch.zeros(self.rns_np.sum(), device=device), requires_grad=False).float()
        self.delta_obs_std = nn.Parameter(
            torch.ones(self.rns_np.sum(), device=device), requires_grad=False).float()

        self.fcs = []

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = ParallelizedLayer(
                ensemble_size, in_size, next_size,
                device=device,
                w_std_value=1./(2*np.sqrt(in_size)),
                b_init_value=b_init_value,
            )
            if spectral_norm:
                fc = torch.nn.utils.parametrizations.spectral_norm(fc, name='W')
            self.__setattr__('fc%d' % i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = ParallelizedLayer(
            ensemble_size, in_size, output_size,
            device=device,
            w_std_value=1. / (2 * np.sqrt(in_size)),
            b_init_value=b_init_value,
        )
        if self.separate_mean_var:
            self.last_fc_std = ParallelizedLayer(
                ensemble_size, in_size, output_size,        # the var of r and ns
                device=device,
                w_std_value=1. / (2 * np.sqrt(in_size)),
                b_init_value=b_init_value,
            )

    def forward(self, inputs):
        dim = len(inputs.shape)

        # inputs normalization
        h = (inputs - self.input_mu) / self.input_std

        # repeat h to make amenable to parallelization
        if dim < 3:
            h = h.unsqueeze(0)
            if dim == 1:
                h = h.unsqueeze(0)
            h = h.repeat(self.ensemble_size, 1, 1)

        # standard feedforward network
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)

        if not self.separate_mean_var:
            preactivation = self.last_fc(h)
            output = self.output_activation(preactivation)
        else:       # separate_mean_var
            preactivation, preactivation_std = self.last_fc(h), self.last_fc_std(h)
            output = self.output_activation(preactivation), self.output_activation(preactivation_std)

        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            if not self.separate_mean_var:
                output = output.squeeze(1)
            else:       # separate_mean_var
                output = output[0].squeeze(1), output[1].squeeze(1)

        # output is (ensemble_size, batch_size, output_size) or tuple (mean, logstd) if separate_mean_var
        return output

    def sample(self, inputs):
        preds = self.forward(inputs)
        batch_size = preds.shape[1]
        model_idxes = np.random.choice(self.elites, size=batch_size)
        batch_idxes = np.arange(0, batch_size)
        samples = preds[model_idxes, batch_idxes]

        # return unnormalized delta
        samples[..., self.rns_torch] = samples[..., self.rns_torch] * self.delta_obs_std + self.delta_obs_mu
        return samples

    def fit_input_stats(self, data, y=None, mask=None):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std != std] = 0
        std[std < 1e-12] = 1e-12
        if y is not None:
            delta_mean = np.mean(y[:, self.rns_np], axis=0, keepdims=True)
            delta_std = np.std(y[:, self.rns_np], axis=0, keepdims=True)
            delta_std[delta_std != delta_std] = 0
            delta_std[delta_std < 1e-12] = 1e-12

            self.delta_obs_mu.data = torch.from_numpy(delta_mean).float().to(self.device)
            self.delta_obs_std.data = torch.from_numpy(delta_std).float().to(self.device)

        if mask is not None:
            mean *= mask
            std *= mask

        self.input_mu.data = torch.from_numpy(mean).float().to(self.device)
        self.input_std.data = torch.from_numpy(std).float().to(self.device)

    def fit_input_stats_torch(self, data, y=None, mask=None):
        mean = torch.mean(data, dim=0, keepdim=True)
        std = torch.std(data, dim=0, keepdim=True)
        std[std != std] = 0
        std[std < 1e-12] = 1e-12
        if y is not None:
            delta_mean = torch.mean(y[:, self.rns_np], dim=0, keepdim=True)
            delta_std = torch.std(y[:, self.rns_np], dim=0, keepdim=True)
            delta_std[delta_std != delta_std] = 0
            delta_std[delta_std < 1e-12] = 1e-12

            self.delta_obs_mu.data = delta_mean
            self.delta_obs_std.data = delta_std

        if mask is not None:
            mean *= mask
            std *= mask

        self.input_mu.data = mean
        self.input_std.data = std

    def termination(self, state, action, next_state, reward):
        # original termination condition
        term_original = self.original_termination(state, action, next_state)
        # next_state should be within the state space
        term_next_state = torch.any(next_state.abs() > self.obs_range, dim=-1, keepdim=True)
        # reward should be within reasonable range
        term_reward = reward.abs() > self.reward_range
        # augmented termination condition equals term_next_state OR term_reward
        term_augmented = torch.logical_or(term_next_state, term_reward)
        # change the rewards for the augmented termination to the lowest bound
        reward[term_augmented] = -self.reward_range
        # return the augmented termination as the OR of the original and the augmented
        return torch.logical_or(term_original, term_augmented)

    def set_reward_range(self, new_reward_range):
        self.reward_range = torch.tensor(new_reward_range, device=self.device).float()
        print_banner(f"New reward range is {self.reward_range.cpu().numpy()} !!!")
