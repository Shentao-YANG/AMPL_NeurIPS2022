from gym.spaces import Box, Discrete, Tuple
import torch
import numpy as np
import random, os


def always_n(train_steps, n):
    return n


def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))


def mode(env, mode_type):
    try:
        getattr(env, mode_type)()
    except AttributeError:
        pass


def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def _elem_or_tuple_to_variable(elem_or_tuple, device):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e, device=device) for e in elem_or_tuple
        )
    return torch.from_numpy(elem_or_tuple).float().to(device)


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch, device):
    return {
        k: _elem_or_tuple_to_variable(x, device=device)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }


def print_banner(s, separator="-", num_star=60):
    print(separator * num_star, flush=True)
    print(s, flush=True)
    print(separator * num_star, flush=True)


def set_torch_random_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)                                # as reproducibility docs
    torch.manual_seed(seed)                             # as reproducibility docs
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False              # as reproducibility docs
        torch.backends.cudnn.deterministic = True           # as reproducibility docs
    print_banner(f"Set torch random seed: {seed}")
