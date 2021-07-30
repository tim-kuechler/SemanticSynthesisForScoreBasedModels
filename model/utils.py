import torch
import sde_lib


def restore_checkpoint(optimizer, model, ema, ckpt_dir, device='cuda'):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    if optimizer is not None: optimizer.load_state_dict(loaded_state['optimizer'])
    if model is not None: model.load_state_dict(loaded_state['models'], strict=False)
    if ema is not None: ema.load_state_dict(loaded_state['ema'])
    epoch = loaded_state['epoch']
    return epoch


def save_checkpoint(optimizer, model, ema, epoch, ckpt_dir):
    saved_state = {
        'optimizer': optimizer.state_dict(),
        'models': model.state_dict(),
        'ema': ema.state_dict(),
        'epoch': epoch
    }
    torch.save(saved_state, ckpt_dir)


def get_score_fn(sde, model, train=False):
    """Wraps `score_fn` so that the models output corresponds to a real time-dependent score function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A score models.
      train: `True` for training and `False` for evaluation.

    Returns:
      A score function.
    """
    if isinstance(sde, sde_lib.VPSDE):
        def score_fn(x, t):
            # Scale neural network output by standard deviation and flip sign
            # For VP-trained models, t=0 corresponds to the lowest noise level
            # The maximum value of time embedding is assumed to 999 for
            # continuously-trained models.
            labels = t * 999
            if train:
                model.train()
                score = model(x, labels)
            else:
                model.eval()
                score = model(x, labels)
            std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            score = -score / std[:, None, None, None]
            return score

    elif isinstance(sde, sde_lib.VESDE):
        def score_fn(x, t):
            labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            if train:
                model.train()
                score = model(x, labels)
            else:
                model.eval()
                score = model(x, labels)
            return score

    return score_fn

def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))
