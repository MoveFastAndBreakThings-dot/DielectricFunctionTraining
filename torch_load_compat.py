"""PyTorch 2.6+ sets torch.load(weights_only=True) by default; e3nn constants.pt needs False."""

import torch

_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat
