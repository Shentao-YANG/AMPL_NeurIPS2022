import gym
from gym.spaces import Discrete

from replay_buffers.simple_replay_buffer import SimpleReplayBuffer
from utils.utils import get_dim
import numpy as np
import torch


class EnvReplayBuffer(SimpleReplayBuffer):

    def __init__(
            self,
            max_replay_buffer_size,
            env,
            store_log_probs=False
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
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
            store_log_probs=store_log_probs
        )
        self._init_states = np.vstack([env.reset() for _ in range(int(1e5))])

    def obs_preproc(self, obs):
        if len(obs.shape) > len(self._ob_space.shape):
            obs = np.reshape(obs, (obs.shape[0], self._observation_dim))
        else:
            obs = np.reshape(obs, (self._observation_dim,))
        return obs

    def obs_postproc(self, obs):
        if self._ob_shape is None:
            return obs
        if len(obs.shape) > 1:
            obs = np.reshape(obs, (obs.shape[0], *self._ob_shape))
        else:
            obs = np.reshape(obs, self._ob_shape)
        return obs

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal
        )

    def all_start(self, batch_size=2048, device=None):
        if batch_size == -1:
            starts = self._init_states
        else:
            ind = np.random.randint(self._init_states.shape[0], size=batch_size)
            starts = self._init_states[ind]
        if device is not None:
            starts = torch.FloatTensor(starts).to(device)
        return starts

    def get_snapshot(self):
        return super().get_snapshot()
