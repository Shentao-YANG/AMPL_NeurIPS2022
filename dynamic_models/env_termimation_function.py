import torch


def term_halfcheetah(obs, act, next_obs):
    return torch.zeros((next_obs.shape[0], 1), dtype=bool, device=next_obs.device)


def term_walker2d(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done =  (height > 0.8) \
                * (height < 2.0) \
                * (angle > -1.0) \
                * (angle < 1.0)
    done = ~not_done
    done = done[:, None]
    return done


def term_hopper(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done =  torch.isfinite(next_obs).all(axis=-1) \
                * (torch.abs(next_obs[:, 1:]) < 100).all(axis=-1) \
                * (height > .7) \
                * (torch.abs(angle) < .2)
    done = ~not_done
    done = done[:, None]
    return done


def term_maze2d(obs, act, next_obs):
    return torch.zeros((next_obs.shape[0], 1), dtype=bool, device=next_obs.device)


def term_door(obs, act, next_obs):
    return torch.zeros((next_obs.shape[0], 1), dtype=bool, device=next_obs.device)


def term_pen(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
    return next_obs[:, 26:27] < 0.075


def set_termination(env_name):
    if 'halfcheetah' in env_name:
        ret = term_halfcheetah
    elif 'walker2d' in env_name:
        ret = term_walker2d
    elif 'hopper' in env_name:
        ret = term_hopper
    elif "maze2d" in env_name:
        ret = term_maze2d
    elif "pen" in env_name:
        ret = term_pen
    elif "door" in env_name:
        ret = term_door
    else:
        raise NotImplementedError(f"Do not know the termination function of {env_name}!")

    return ret






