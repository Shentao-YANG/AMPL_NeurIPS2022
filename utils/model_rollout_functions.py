import torch
from utils.utils import get_numpy, print_banner
import numpy as np


def _create_full_tensors(start_states, max_path_length, obs_dim, action_dim, device):
    num_rollouts = start_states.shape[0]
    observations = torch.zeros((num_rollouts, max_path_length+1, obs_dim), device=device)
    if isinstance(start_states, np.ndarray):
        observations[:, 0] = torch.from_numpy(start_states).float().to(device)
    else:
        observations[:, 0] = start_states
    actions = torch.zeros((num_rollouts, max_path_length, action_dim), device=device)
    rewards = torch.zeros((num_rollouts, max_path_length, 1), device=device)
    terminals = torch.zeros((num_rollouts, max_path_length, 1), device=device, dtype=torch.uint8)
    return observations, actions, rewards, terminals


def _get_prediction(dynamics_model, states, actions):
    state_actions = torch.cat([states, actions], dim=-1)
    transitions = dynamics_model.sample(state_actions)
    if (transitions != transitions).any():
        print_banner('WARNING: NaN TRANSITIONS IN DYNAMICS MODEL ROLLOUT')
        transitions[transitions != transitions] = 0

    rewards = transitions[:, :1]
    delta_obs = transitions[:, 1:]
    next_obs = states + delta_obs
    dones = dynamics_model.termination(states, actions, next_obs, rewards)

    return rewards, dones, next_obs


def _create_paths(observations, actions, rewards, terminals, max_path_length, replay_buffer_device):
    if replay_buffer_device == "numpy":
        observations = get_numpy(observations)
        actions = get_numpy(actions)
        rewards = get_numpy(rewards)
        terminals = get_numpy(terminals)

    paths = []
    for i in range(len(observations)):
        rollout_len = 1
        while rollout_len < max_path_length and terminals[i, rollout_len-1, 0] < 0.5:  # just check 0 or 1
            rollout_len += 1
        paths.append(dict(
            observations=observations[i, :rollout_len],
            actions=actions[i, :rollout_len],
            rewards=rewards[i, :rollout_len],
            next_observations=observations[i, 1:rollout_len + 1],
            terminals=terminals[i, :rollout_len],
        ))
    return paths


def _get_policy_actions(states, action_kwargs):
    policy = action_kwargs['policy']
    actions, *_ = policy.forward(states)
    return actions


def _model_rollout(
        dynamics_model,                             # torch dynamics model: (s, a) --> (r, d, s')
        start_states,                               # numpy array of states: (num_rollouts, obs_dim)
        get_action,                                 # method for getting action
        device,
        replay_buffer_device,
        action_kwargs=None,                         # kwargs for get_action (ex. policy or actions)
        max_path_length=1000,                       # maximum rollout length (if not terminated)
        create_full_tensors=_create_full_tensors,
        get_prediction=_get_prediction,
        create_paths=_create_paths,
):
    if action_kwargs is None:
        action_kwargs = dict()
    if max_path_length is None:
        raise ValueError('Must specify max_path_length in rollout function')

    obs_dim = dynamics_model.obs_dim
    action_dim = dynamics_model.action_dim

    s, a, r, d = create_full_tensors(start_states, max_path_length, obs_dim, action_dim, device=device)
    for t in range(max_path_length):
        a[:, t] = get_action(s[:, t], action_kwargs)
        r[:, t], d[:, t], s[:, t+1] = get_prediction(
            dynamics_model,
            s[:, t], a[:, t],
        )

    paths = create_paths(s, a, r, d, max_path_length, replay_buffer_device)

    return paths


def policy(dynamics_model, policy, start_states, device, max_path_length=1000, replay_buffer_device="numpy", **kwargs):
    return _model_rollout(
        dynamics_model,
        start_states,
        _get_policy_actions,
        device=device,
        action_kwargs=dict(policy=policy),
        max_path_length=max_path_length,
        replay_buffer_device=replay_buffer_device,
        **kwargs,
    )
