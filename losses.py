import torch
import model.utils as utils
import numpy as np


def get_sde_loss_fn(sde, train, reduce_mean=True, eps=1e-5):
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      eps: A `float` number. The smallest time step to sample from.

    Returns:
      A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        """Compute the loss function.

        Args:
          model: A score models.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = utils.get_score_fn(sde, model, train=train)
        t = torch.rand(batch.shape[0], device=batch.device) * (1 - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        score = score_fn(perturbed_data, t)

        losses = torch.square(score * std[:, None, None, None] + z)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)

        loss = torch.mean(losses)
        return loss

    return loss_fn

def get_step_fn(config, score_model, optimizer, loss_fn, ema, scaler=None):
    """
    Gets the step function for training

    :param config: The config
    :param optimizer: The optimizer to use
    :param loss_fn: The loss function of the model
    :param ema: The EMA of the model
    :param scaler: The torch.cuda.amp.GradScaler if config.optim.mixed_prec is True
    :return: The training step function
    """
    def optimzer_optimization_fn(step):
        #Optimizer improvements
        if config.optim.warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = config.optim.lr * np.minimum(step / config.optim.warmup, 1.0)
        if config.optim.grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(score_model.parameters(), max_norm=config.optim.grad_clip)

    def step_fn(x, step):
        optimizer.zero_grad()
        loss = loss_fn(score_model, x)
        loss.backward()
        optimzer_optimization_fn(step)
        optimizer.step()
        ema.update(score_model.parameters())

        return loss.detach().item()

    def step_fn_mixed_prec(x, step):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = loss_fn(score_model, x)
        scaler.scale(loss).backward()
        optimzer_optimization_fn(step)
        scaler.step(optimizer)
        scaler.update()
        ema.update(score_model.parameters())

        return loss.detach().item()

    if not config.optim.mixed_prec:
        return step_fn
    else:
        assert scaler is not None
        return step_fn_mixed_prec
