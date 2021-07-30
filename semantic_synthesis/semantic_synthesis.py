import torch
import torch.nn.functional as F
import model.utils as mutils
from sampling import get_predictor, get_corrector


def get_semantic_segmentation_grad_fn(sde, sem_mask, sem_seg_model, max_scale):
    def scale_fn(t):
        if t > 0.7:
            return 0
        elif 0.7 >= t >= 0.45:
            return max_scale
        else:
            return (max_scale/0.45**2)*t**2

    def semantic_segmentation_grad_fn(x, t):
        with torch.enable_grad():
            x = x.clone().detach()
            x.requires_grad = True

            #Scale to [0, 1]
            max = torch.ones(x.shape[0], device='cuda:0')
            min = torch.ones(x.shape[0], device='cuda:0')
            for N in range(x.shape[0]):
                max[N] = torch.max(x[N, :, :, :])
                min[N] = torch.min(x[N, :, :, :])
            x = x - min[:, None, None, None] * torch.ones_like(x, device='cuda:0')
            x = torch.div(x, (max - min)[:, None, None, None])

            _, std = sde.marginal_prob(x, t)
            pred = sem_seg_model(x, std)

            #Logits to probabilities
            prob = F.softmax(pred, dim=1)
            #Reduce probabilities to only the ones matching the original label at one pixel
            prob = torch.mul(prob, sem_mask)
            #Removing channel dimension by discarding all 0 elements from above operation
            prob, _ = torch.max(prob, dim=1)
            grad = torch.autograd.grad(prob, x, torch.ones_like(prob))
        return grad[0] * scale_fn(t)

    return semantic_segmentation_grad_fn

def get_semantic_synthesis_sampler(config, sde, model, sem_seg_model, sem_mask, n_steps=1,
                                   probability_flow=False, denoise=True, eps=1e-3, device='cuda:0', scale=None):
    """
    Gets the sampler for semantic synthesis

    :param sde: The 'sde_lib.SDE' of the model
    :param model: The score model
    :param sem_seg_model: The semantic segmentation model trained for the dataset
    :param sem_mask: The semantic map to generate a sample from (in format (1, C, H, W))
    :param n_steps: The corrector steps per corrector update
    :param probability_flow: If predictor should use probability flow
    :param denoise: If true denoises the sample before returning it
    :param eps: A Number for numerical stability
    :param device: PyTorch device

    :return: The sampling function
    """
    #Create gradient functions
    score_fn = mutils.get_score_fn(sde, model, train=False)
    semantic_segmentation_grad_fn = get_semantic_segmentation_grad_fn(sde, sem_mask, sem_seg_model,
                                                        config.sampling.sem_seg_scale if scale is None else scale)

    # Get predictor and corrector
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())

    def total_grad_fn(x, t):
        return score_fn(x, t) + semantic_segmentation_grad_fn(x, t)

    #Create predictor & corrector update functions
    def semantic_predictor_update_fn(x, t):
        predictor_inst = predictor(sde, total_grad_fn, probability_flow)
        return predictor_inst.update_fn(x, t)

    def semantic_corrector_update_fn(x, t):
        corrector_inst = corrector(sde, total_grad_fn, config.sampling.snr, n_steps)
        return corrector_inst.update_fn(x, t)

    #Define sampler process
    shape = (config.sampling.n_samples_per_seg_mask, config.data.n_channels,
             sem_mask.shape[2], sem_mask.shape[3])
    def semantic_synthesis_sampler():
        """
        The semantic synthesis sampler function

        :return: Samples
        """
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape).to(device)

            timesteps = torch.linspace(1, eps, sde.N, device=device)

            for i in range(sde.N):
                vec_t = torch.ones(config.sampling.n_samples_per_seg_mask, device=device) * timesteps[i]
                x, x_mean = semantic_corrector_update_fn(x, vec_t)
                x, x_mean = semantic_predictor_update_fn(x, vec_t)
            return x_mean if denoise else x

    return semantic_synthesis_sampler
