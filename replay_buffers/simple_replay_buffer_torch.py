import copy
from collections import OrderedDict
import numpy as np
from utils.utils import print_banner
import torch


class SimpleReplayBufferTorch(object):

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        device,
        store_log_probs=False,
        store_na_w=False,
    ):
        self.store_na_w = store_na_w
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self.device = device
        self._observations = torch.zeros((max_replay_buffer_size, observation_dim), device=self.device)
        self._next_obs = torch.zeros((max_replay_buffer_size, observation_dim), device=self.device)
        self._actions = torch.zeros((max_replay_buffer_size, action_dim), device=self.device)
        self._rewards = torch.zeros((max_replay_buffer_size, 1), device=self.device)
        self._terminals = torch.zeros((max_replay_buffer_size, 1), device=self.device, dtype=torch.uint8)
        if store_log_probs:
            self._logprobs = torch.zeros((max_replay_buffer_size, 1), device=self.device)
        if self.store_na_w:
            self._next_actions = torch.zeros((max_replay_buffer_size, action_dim), device=self.device)
            self._w = torch.ones((max_replay_buffer_size, 1), device=self.device)

        self._top = 0
        self._size = 0
        self._store_log_probs = store_log_probs

        self.total_entries = 0

        print_banner(f"Initialize SimpleReplayBufferTorch, size={self._observations.shape[0]}, store_na_w={self.store_na_w}")

    def obs_preproc(self, obs):
        return obs

    def obs_postproc(self, obs):
        return obs

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, next_action=None):
        self._observations[self._top] = self.obs_preproc(observation)
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = self.obs_preproc(next_observation)

        if self.store_na_w:
            self._next_actions[self._top] = next_action

        self._advance()

    def add_sample_with_logprob(self, observation, action, reward, terminal,
                                next_observation, logprob, next_action=None):
        if self._store_log_probs is False:
            raise ValueError('The replay buffer does not support storing log-probs !!!')
        self._logprobs[self._top] = logprob
        self.add_sample(observation, action, reward, terminal, next_observation, next_action)

    def add_path(self, path):
        for k, v in path.items():
            if isinstance(v, np.ndarray):
                path[k] = torch.from_numpy(v).float().to(self.device)

        if self.store_na_w:
            for i, (obs, action, reward, next_obs, terminal, next_action) in enumerate(zip(
                path["observations"],
                path["actions"],
                path["rewards"],
                path["next_observations"],
                path["terminals"],
                path["next_actions"]
            )):
                self.add_sample(
                    observation=obs,
                    action=action,
                    reward=reward,
                    next_observation=next_obs,
                    terminal=terminal,
                    next_action=next_action
                )
        else:
            for i, (obs, action, reward, next_obs, terminal) in enumerate(zip(
                path["observations"],
                path["actions"],
                path["rewards"],
                path["next_observations"],
                path["terminals"]
            )):
                self.add_sample(
                    observation=obs,
                    action=action,
                    reward=reward,
                    next_observation=next_obs,
                    terminal=terminal
                )
        self.terminate_episode()

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    def get_transitions(self):
        return torch.cat([
            self._observations[:self._size],
            self._actions[:self._size],
            self._rewards[:self._size],
            self._next_obs[:self._size]
        ], dim=1)

    def get_logprobs(self):
        return self._logprobs[:self._size].copy()

    def relabel_rewards(self, rewards):
        self._rewards[:len(rewards)] = rewards

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1
        self.total_entries += 1

    def random_batch(self, batch_size, min_pct=0, max_pct=1, include_logprobs=False, return_indices=False, device=None):
        indices = torch.randint(
            low=int(min_pct * self._size),
            high=int(max_pct * self._size),
            size=(batch_size,),
            device=self.device
        )
        batch = dict(
            observations=self.obs_postproc(self._observations[indices]),
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self.obs_postproc(self._next_obs[indices]),
        )
        if self._store_log_probs and include_logprobs:
            batch['logprobs'] = self._logprobs[indices]

        if self.store_na_w:
            batch['next_actions'] = self._next_actions[indices]
            batch['weights'] = self._w[indices]

        if return_indices:
            return batch, indices
        else:
            return batch

    def get_snapshot(self):
        return dict(
            observations=self._observations[:self._size],
            actions=self._actions[:self._size],
            rewards=self._rewards[:self._size],
            terminals=self._terminals[:self._size],
            next_observations=self._next_obs[:self._size]
        )

    def load_snapshot(self, snapshot):
        for t in range(snapshot['observations'].shape[0]):
            self.add_sample(
                observation=snapshot['observations'][t],
                action=snapshot['actions'][t],
                reward=snapshot['rewards'][t],
                next_observation=snapshot['next_observations'][t],
                terminal=snapshot['terminals'][t]
            )

    def top(self):
        return self._top

    def num_steps_can_sample(self):
        return self._size

    def max_replay_buffer_size(self):
        return self._max_replay_buffer_size

    def obs_dim(self):
        return self._observation_dim

    def action_dim(self):
        return self._action_dim

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])

    @property
    def reward_stat(self):
        return {
            "mean": self._rewards.mean().cpu().numpy().mean(),
            "std": self._rewards.std().cpu().numpy().mean(),
            "max": self._rewards.max().cpu().numpy().mean(),
            "min": self._rewards.min().cpu().numpy().mean()
        }

    @property
    def terminals_pos_weight(self):
        num_pos = self._terminals.sum().cpu().numpy().mean()
        num_neg = self._size - num_pos
        pos_weight = num_neg / num_pos if num_pos > 0.5 else 1e4
        print_banner(f"num_pos {num_pos}, num_neg {num_neg}, pos_weight {pos_weight:.3f}")
        return pos_weight

    @property
    def obs_range(self):
        return np.maximum(self._observations.abs().max(0)[0].cpu().numpy(), self._next_obs.abs().max(0)[0].cpu().numpy()) * 2.

    def make_rewards_positive(self):
        self._rewards = (self._rewards - self._rewards.min() + 1e-3) / (self._rewards.max() - self._rewards.min())

    def copy_weights(self, weights):
        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights).float().to(self.device)
        assert weights.shape[0] == self._observations.shape[0], f"Number of weights: {weights.shape[0]}; Number of states: {self._observations.shape[0]}"
        self._w = copy.deepcopy(weights)
        print_banner("Loaded weights !!!")
