import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import print_banner
import abc
from torch.distributions import Normal

NEGATIVE_SLOPE = 1. / 100.
print_banner(f"Negative_slope = {NEGATIVE_SLOPE}")
ATANH_MAX = 1. - 1e-7
ATANH_MIN = -1. + 1e-7
LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0


class Noise(object, metaclass=abc.ABCMeta):
    def __init__(self, device):
        self.device = device

    @abc.abstractmethod
    def sample_noise(self, shape, dtype=None, requires_grad=False):
        pass


class NormalNoise(Noise):
    def __init__(self, device, mean=0., std=1.):
        super().__init__(device=device)

        self.mean = mean
        self.std = std
        print_banner(f"Use Normal Noise with mean={self.mean} and std={self.std}.")

    def sample_noise(self, shape, dtype=None, requires_grad=False):
        return torch.randn(size=shape, dtype=dtype, device=self.device, requires_grad=requires_grad) * self.std + self.mean


class UniformNoise(Noise):
    def __init__(self, device, lower=0., upper=1.):
        super().__init__(device=device)

        self.lower = lower
        self.upper = upper
        print_banner(f"Use Uniform Noise in [{self.lower}, {self.upper}).")

    def sample_noise(self, shape, dtype=None, requires_grad=False):
        return torch.rand(size=shape, dtype=dtype, device=self.device, requires_grad=requires_grad) * (self.upper - self.lower) + self.lower


class ImplicitPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, noise, noise_method, noise_dim, device):
        # noise : {NormalNoise, UniformNoise}
        # noise_method : {"concat", "add", "multiply"}
        # noise_dim : dimension of noise for "concat" method
        super(ImplicitPolicy, self).__init__()

        self.hidden_size = (400, 300)

        if noise_dim < 1:
            noise_dim = min(10, state_dim // 2)
        noise_dim = int(noise_dim)

        print_banner(f"In implicit policy, use noise_method={noise_method} and noise_dim={noise_dim}")

        if noise_method == "concat":
            self.l1 = nn.Linear(state_dim + noise_dim, self.hidden_size[0])
        else:
            self.l1 = nn.Linear(state_dim, self.hidden_size[0])

        self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l3 = nn.Linear(self.hidden_size[1], action_dim)

        self.max_action = max_action
        self.noise = noise
        self.noise_method = noise_method
        self.noise_dim = noise_dim
        self.device = device

    def forward(self, state, return_raw_action=False):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        # state.shape = (batch_size, state_dim)
        if self.noise_method == "concat":
            epsilon = self.noise.sample_noise(shape=(np.prod(state.shape[:-1]) * self.noise_dim, 1)).reshape(state.shape[:-1] + (-1,)).clamp(-3, 3)
            state = torch.cat([state, epsilon], -1)      # dim = (state.shape[0], state_dim + noise_dim)
        if self.noise_method == "add":
            epsilon = self.noise.sample_noise(shape=state.shape)
            state = state + epsilon                     # dim = (state.shape[0], state_dim)
        if self.noise_method == "multiply":
            epsilon = self.noise.sample_noise(shape=state.shape)
            state = state * epsilon                     # dim = (state.shape[0], state_dim)

        a = F.leaky_relu(self.l1(state), negative_slope=NEGATIVE_SLOPE)
        a = F.leaky_relu(self.l2(a), negative_slope=NEGATIVE_SLOPE)
        raw_actions = self.l3(a)
        if return_raw_action:
            return self.max_action * torch.tanh(raw_actions), raw_actions
        else:
            return self.max_action * torch.tanh(raw_actions)

    def sample_multiple_actions(self, state, num_action=10, std=-1., return_raw_action=False):
        # num_action : number of actions to sample from policy for each state

        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        batch_size = state.shape[0]
        # e.g., num_action = 3, [s1;s2] -> [s1;s1;s1;s2;s2;s2]
        if std <= 0:
            state = state.unsqueeze(1).repeat(1, num_action, 1).view(-1, state.size(-1)).to(self.device)
        else:   # std > 0
            if num_action == 1:
                noises = torch.normal(torch.zeros_like(state), torch.ones_like(state))  # B * state_dim
                state = (state + (std * noises).clamp(-0.05, 0.05)).to(self.device)  # B x state_dim
            else:   # num_action > 1: sample (num_action-1) noisy states, concatenate with the original state
                state_noise = state.unsqueeze(1).repeat(1, num_action-1, 1)   # B * (num_action-1) * state_dim
                noises = torch.normal(torch.zeros_like(state_noise), torch.ones_like(state_noise))  # B * num_q_samples * state_dim
                state_noise = state_noise + (std * noises).clamp(-0.05, 0.05)  # N x num_action x state_dim
                state = torch.cat((state_noise, state.unsqueeze(1)), dim=1).view((batch_size * num_action), -1).to(self.device)  # (B * num_action) x state_dim
        # return [a11;a12;a13;a21;a22;a23] for [s1;s1;s1;s2;s2;s2]
        if return_raw_action:
            actions, raw_actions = self.forward(state, return_raw_action=return_raw_action)
            return state, actions, raw_actions
        else:
            return state, self.forward(state)

    def pre_scaling_action(self, actions):
        # action = self.max_action * torch.tanh(pre_tanh_action)
        # atanh(action / self.max_action) = atanh( tanh(pre_tanh_action) ) = pre_tanh_action
        return torch.atanh(torch.clamp(actions / self.max_action, min=ATANH_MIN, max=ATANH_MAX))


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, device):
        super(GaussianPolicy, self).__init__()
        self.hidden_size = (400, 300)
        print_banner(f"Use Gaussian Policy !")

        self.l1 = nn.Linear(state_dim, self.hidden_size[0])

        self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l_mu = nn.Linear(self.hidden_size[1], action_dim)
        self.l_log_std = nn.Linear(self.hidden_size[1], action_dim)

        self.epsilon = 1e-6
        self.max_action = max_action
        self.device = device

    def forward(self, state, return_raw_action=False):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        # state.shape = (batch_size, state_dim)
        h = F.leaky_relu(self.l1(state), negative_slope=NEGATIVE_SLOPE)
        h = F.leaky_relu(self.l2(h), negative_slope=NEGATIVE_SLOPE)
        mu = self.l_mu(h)
        log_std = self.l_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        # sample action
        normal_noise = Normal(torch.zeros_like(mu), torch.ones_like(std))
        action = mu + std * normal_noise.sample()
        tanh_action = torch.tanh(action)

        if return_raw_action:
            return self.max_action * tanh_action, action
        else:
            return self.max_action * tanh_action

    def log_prob(self, state, actions):
        h = F.leaky_relu(self.l1(state), negative_slope=NEGATIVE_SLOPE)
        h = F.leaky_relu(self.l2(h), negative_slope=NEGATIVE_SLOPE)
        mu = self.l_mu(h)
        log_std = self.l_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        # compute log_prob
        # log_prob is negative entropy
        tanh_action = torch.tanh(actions)
        log_prob = Normal(mu, std).log_prob(actions) - torch.log(1. - tanh_action * tanh_action + self.epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return log_prob

    def sample_multiple_actions(self, state, num_action=10, std=-1., return_raw_action=False):
        # num_action : number of actions to sample from policy for each state

        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        batch_size = state.shape[0]
        # e.g., num_action = 3, [s1;s2] -> [s1;s1;s1;s2;s2;s2]
        if std <= 0:
            state = state.unsqueeze(1).repeat(1, num_action, 1).view(-1, state.size(-1)).to(self.device)
        else:   # std > 0
            if num_action == 1:
                noises = torch.normal(torch.zeros_like(state), torch.ones_like(state))  # B * state_dim
                state = (state + (std * noises).clamp(-0.05, 0.05)).to(self.device)  # B x state_dim
            else:   # num_action > 1
                state_noise = state.unsqueeze(1).repeat(1, num_action-1, 1)   # B * num_action * state_dim
                noises = torch.normal(torch.zeros_like(state_noise), torch.ones_like(state_noise))  # B * num_q_samples * state_dim
                state_noise = state_noise + (std * noises).clamp(-0.05, 0.05)  # N x num_action x state_dim
                state = torch.cat((state_noise, state.unsqueeze(1)), dim=1).view((batch_size * num_action), -1).to(self.device)  # (B * num_action) x state_dim
        # return [a11;a12;a13;a21;a22;a23] for [s1;s1;s1;s2;s2;s2]
        if return_raw_action:
            actions, raw_actions = self.forward(state, return_raw_action=return_raw_action)
            return state, actions, raw_actions
        else:
            return state, self.forward(state)

    def pre_scaling_action(self, actions):
        # action = self.max_action * torch.tanh(pre_tanh_action)
        # atanh(action / self.max_action) = atanh( tanh(pre_tanh_action) ) = pre_tanh_action
        return torch.atanh(torch.clamp(actions / self.max_action, min=ATANH_MIN, max=ATANH_MAX))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.hidden_size = (400, 300)

        self.l1 = nn.Linear(state_dim + action_dim, self.hidden_size[0])
        self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l3 = nn.Linear(self.hidden_size[1], 1)

        self.l4 = nn.Linear(state_dim + action_dim, self.hidden_size[0])
        self.l5 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l6 = nn.Linear(self.hidden_size[1], 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], -1)
        q1 = F.leaky_relu(self.l1(state_action), negative_slope=NEGATIVE_SLOPE)
        q1 = F.leaky_relu(self.l2(q1), negative_slope=NEGATIVE_SLOPE)
        q1 = self.l3(q1)

        q2 = F.leaky_relu(self.l4(state_action), negative_slope=NEGATIVE_SLOPE)
        q2 = F.leaky_relu(self.l5(q2), negative_slope=NEGATIVE_SLOPE)
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.leaky_relu(self.l1(torch.cat([state, action], -1)), negative_slope=NEGATIVE_SLOPE)
        q1 = F.leaky_relu(self.l2(q1), negative_slope=NEGATIVE_SLOPE)
        q1 = self.l3(q1)
        return q1

    def q_min(self, state, action, no_grad=False):
        if no_grad:
            with torch.no_grad():
                q1, q2 = self.forward(state, action)
        else:
            q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

    def weighted_min(self, state, action, lmbda=0.75, no_grad=False):
        # lmbda * Q_min + (1-lmbda) * Q_max
        if no_grad:
            with torch.no_grad():
                q1, q2 = self.forward(state, action)
        else:
            q1, q2 = self.forward(state, action)
        return lmbda * torch.min(q1, q2) + (1. - lmbda) * torch.max(q1, q2)


class DiscriminatorWithSigmoid(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DiscriminatorWithSigmoid, self).__init__()
        self.hidden_size = (400, 300)
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, self.hidden_size[0]),
            nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE),
            nn.Linear(self.hidden_size[1], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        validity = self.model(x)
        return validity


class DiscriminatorWithoutSigmoid(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DiscriminatorWithoutSigmoid, self).__init__()
        self.hidden_size = (400, 300)
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, self.hidden_size[0]),
            nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE),
            nn.Linear(self.hidden_size[1], 1)
        )

    def forward(self, x):
        validity = self.model(x)
        return validity


class DensityRatio(nn.Module):
    def __init__(self, state_dim, action_dim, device, output_clipping, smoothing_power_alpha):
        super(DensityRatio, self).__init__()

        self.hidden_size = (400, 300)

        self.l1 = nn.Linear(state_dim + action_dim, self.hidden_size[0])
        self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l3 = nn.Linear(self.hidden_size[1], 1)

        self.l3.weight.data.uniform_(-0.003, 0.003)
        self.l3.bias.data.fill_(0.)

        self.device = device

        self.max_w = torch.tensor([[500.]], device=self.device)
        self.min_w = torch.tensor([[1e-8]], device=self.device)
        self.min_w_no_clipping = torch.tensor([[1e-8]], device=self.device)

        self.output_clipping = output_clipping == "True"
        self.smoothing_power_alpha = smoothing_power_alpha

        print(f"Initialize DensityRatio network, output_clipping={self.output_clipping}, max_w={self.max_w.item()}, min_w={self.min_w.item()}, min_w_no_clipping={self.min_w_no_clipping.item()}, smoothing_power_alpha={self.smoothing_power_alpha}")

    def forward(self, state, action):
        state_action = torch.cat([state, action], -1)
        q1 = F.leaky_relu(self.l1(state_action), negative_slope=NEGATIVE_SLOPE)
        q1 = F.leaky_relu(self.l2(q1), negative_slope=NEGATIVE_SLOPE)
        q1 = self.l3(q1)

        if self.output_clipping:
            # use softplus to clip to (min_w, max_w)
            q1 = self.max_w - F.softplus(self.max_w - q1)
            q1 = self.min_w + F.softplus(q1 - self.min_w)
            return q1
        else:
            return (F.softplus(q1 - self.min_w_no_clipping) + self.min_w_no_clipping).pow(self.smoothing_power_alpha)
