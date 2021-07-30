from configs.cityscapes256 import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = 'vesde'

    # sampling
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'reverse_diffusion'
    sampling.corrector = 'langevin'

    # semantic sampling
    sampling.sample_data_dir = '/export/data/tkuechle/datasets/cityscapes256'
    sampling.sem_seg_model_dir = '/export/home/tkuechle/SemanticSynthesisForScoreBasedModels/output/test/seg.pth'
    sampling.n_samples_per_seg_mask = 1

    # data
    data = config.data

    # model
    model = config.model
    model.scale_by_sigma = True
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
    model.n_res_blocks = 2
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'output_skip'
    model.progressive_input = 'input_skip'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.
    model.fourier_scale = 16
    model.conv_size = 3
    model.sampling_eps = 1e-5

    return config
