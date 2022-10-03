import gym
from gym.spaces import Discrete

from replay_buffers.simple_replay_buffer_torch import SimpleReplayBufferTorch
from utils.utils import get_dim
import torch
import numpy as np


class EnvReplayBufferTorch(SimpleReplayBufferTorch):

    def __init__(
            self,
            max_replay_buffer_size,
            env,
            device,
            store_log_probs=False,
            store_na_w=False
    ):
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if isinstance(self._ob_space, gym.spaces.Box):
            self._ob_shape = self._ob_space.shape
        else:
            self._ob_shape = None

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            device=device,
            store_log_probs=store_log_probs,
            store_na_w=store_na_w
        )
        self._init_states = torch.from_numpy(np.vstack([env.reset() for _ in range(int(1e5))])).float().to(self.device)

    def obs_preproc(self, obs):
        if len(obs.shape) > len(self._ob_space.shape):
            obs = torch.reshape(obs, (obs.shape[0], self._observation_dim))
        else:
            obs = torch.reshape(obs, (self._observation_dim,))
        return obs

    def obs_postproc(self, obs):
        if self._ob_shape is None:
            return obs
        if len(obs.shape) > 1:
            obs = torch.reshape(obs, (obs.shape[0], *self._ob_shape))
        else:
            obs = torch.reshape(obs, self._ob_shape)
        return obs

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, next_action=None, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = torch.zeros(self._action_dim, device=self.device, dtype=torch.uint8)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            next_action=next_action
        )

    def all_start(self, batch_size=2048, device=None):
        if batch_size == -1:
            return self._init_states
        else:
            ind = torch.randint(low=0, high=self._init_states.shape[0], size=(batch_size,), device=self.device)
            return self._init_states[ind]

    def get_snapshot(self):
        return super().get_snapshot()
