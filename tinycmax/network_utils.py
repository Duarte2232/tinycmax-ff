from collections.abc import Sequence

from dotmap import DotMap
import torch


def recursive_clone(x):
    """
    Clone all tensors in a nested structure.
    """
    if isinstance(x, torch.Tensor):
        return x.clone()
    elif isinstance(x, Sequence):
        return [recursive_clone(xx) for xx in x]
    elif isinstance(x, DotMap):
        return DotMap({k: recursive_clone(v) for k, v in x.items()})
    elif x is None:
        return None
    else:
        raise TypeError(f"expected torch.Tensor or Sequence, got {type(x)}")
