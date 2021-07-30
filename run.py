from pathlib import Path
import os
import losses
from model.ema import ExponentialMovingAverage
from model.ncsnpp import NCSNpp
import torch
import torch.optim as optim
from losses import get_sde_loss_fn
import numpy as np
import sde_lib
from model import utils
import logging
import datasets.datasets as data_loader
import time
import sampling
from torchvision.utils import make_grid, save_image
from semantic_synthesis.semantic_synthesis import get_semantic_synthesis_sampler
from semantic_synthesis.models.unet.unet import UNet
import datasets.cityscapes256.cityscapes256 as cityscapes256
import datasets.flickr.flickr as flickr


def train(config, workdir):
    '''
    Runs the training

    :param config: The configuration
    :param workdir: The working directory
    '''
    # Create sample directory
    sample_dir = os.path.join(workdir, 'samples')
    Path(sample_dir).mkdir(parents=True, exist_ok=True)

    # Initialize models and optimizer
    score_model = NCSNpp(config)
    score_model = score_model.to(config.device)
    score_model = torch.nn.DataParallel(score_model)
    optimizer = optim.Adam(score_model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                           eps=config.optim.eps, weight_decay=config.optim.weight_decay)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    epoch = 1
    logging.info('Model, EMA and optimizer initialized')

    # Create checkpoint directories
    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Check for latest checkpoint
    if os.path.isfile(os.path.join(checkpoint_dir, 'curr_cpt.pth')):
        utils.restore_checkpoint(optimizer, score_model, ema, os.path.join(checkpoint_dir, 'curr_cpt.pth'))
        logging.info('Checkpoint restored')

    # Get data iterators
    data_loader_train, data_loader_eval = data_loader.get_dataset(config)
    logging.info('Dataset initialized')

    # Get SDE
    sde = sde_lib.get_SDE(config)
    logging.info('SDE initialized')

    # Get SDE loss function
    loss_fn_train = get_sde_loss_fn(sde, True, reduce_mean=config.training.reduce_mean)
    loss_fn_eval = get_sde_loss_fn(sde, False, reduce_mean=config.training.reduce_mean)
    logging.info('Loss function loaded')

    #Get step function
    scaler = None if not config.optim.mixed_prec else torch.cuda.amp.GradScaler()
    step_fn = losses.get_step_fn(config, score_model, optimizer, loss_fn_train, ema, scaler)

    # Get sampling function
    if config.training.snapshot_sampling:
        sampling_shape = (config.sampling.batch_size, config.data.n_channels,
                          config.sampling.sampling_height, config.sampling.sampling_width)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, config.model.sampling_eps)

    logging.info(f'Starting training loop at epoch {epoch}')
    step = 0
    loss_per_log_period = 0
    # Training cycle for one epoch
    for i in range(epoch, config.training.epochs + 1):
        start_time = time.time()
        for img, _ in data_loader_train:
            img = img.to(config.device)

            # Training step
            loss = step_fn(img, step)
            step += 1

            # Report loss and save to file
            loss_per_log_period += loss
            if step % config.training.log_freq == 0:
                mean_loss = loss_per_log_period / config.training.log_freq
                with open(os.path.join(workdir, 'training_loss.txt'), 'a+') as training_loss_file:
                    training_loss_file.write(str(step) + '\t' + str(mean_loss) + '\n')
                logging.info(f'step: {step} (epoch: {epoch}), training_loss: {mean_loss}')
                loss_per_log_period = 0

            # Report the loss on an evaluation dataset and save to file
            if step % config.training.eval_freq == 0 and data_loader_eval is not None:
                total_loss = 0
                with torch.no_grad():
                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    for eval_img, _ in data_loader_eval:
                        eval_img = eval_img.to(config.device)
                        eval_loss = loss_fn_eval(score_model, eval_img)
                        total_loss += eval_loss.item()
                    ema.restore(score_model.parameters())
                total_loss = total_loss / len(data_loader_eval)
                with open(os.path.join(workdir, 'eval_loss.txt'), 'a+') as eval_file:
                    eval_file.write(str(step) + '\t' + str(total_loss) + '\n')
                logging.info(f'step: {step} (epoch: {epoch}), eval_loss: {total_loss}')

        # Save the checkpoint
        logging.info(f'Saving checkpoint of epoch {epoch}')
        utils.save_checkpoint(optimizer, score_model, ema, epoch,
                              os.path.join(checkpoint_dir, 'curr_cpt.pth'))
        if epoch % config.training.checkpoint_save_freq == 0:
            utils.save_checkpoint(optimizer, score_model, ema, epoch,
                                  os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth'))

        # Generate and save samples
        if config.training.snapshot_sampling and epoch % config.training.sampling_freq == 0:
            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            samples, n = sampling_fn(score_model)
            ema.restore(score_model.parameters())
            this_sample_dir = os.path.join(sample_dir, f'epoch_{epoch}')
            Path(this_sample_dir).mkdir(parents=True, exist_ok=True)
            nrow = int(np.sqrt(samples.shape[0]))
            image_grid = make_grid(samples, nrow, padding=2)
            save_image(image_grid, os.path.join(this_sample_dir, 'sample.png'))
            logging.info(f'Samples generated in {this_sample_dir}')

        time_for_epoch = time.time() - start_time
        logging.info(f'Finished epoch {epoch}/{config.training.epochs} ({step // epoch} steps in this epoch) in {time_for_epoch} seconds')
        epoch += 1


def image_synthesis(config, workdir, mode):
    # Get checkpoint dir
    checkpoint_dir = os.path.join(workdir, 'checkpoints')

    # Create directory to sample_folder
    sample_dir = os.path.join(workdir, 'sem_sample')
    Path(sample_dir).mkdir(parents=True, exist_ok=True)

    # Load score model from latest checkpoint
    score_model = NCSNpp(config)
    score_model = score_model.to(config.device)
    score_model = torch.nn.DataParallel(score_model)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    utils.restore_checkpoint(None, score_model, ema, os.path.join(checkpoint_dir, 'curr_cpt.pth'))
    ema.copy_to(score_model.parameters())
    logging.info('Score Model loaded')

    # Load semantic segmentation model from checkpoint
    sem_seg_model = UNet(config)
    sem_seg_model = sem_seg_model.to(config.device)
    sem_seg_model = torch.nn.DataParallel(sem_seg_model)
    utils.restore_checkpoint(None, sem_seg_model, None, config.sampling.sem_seg_model_dir)
    logging.info('Semantic Segmentation Model loaded')

    # Get SDE
    sde = sde_lib.get_SDE(config)
    logging.info('SDE initialized')

    if mode == 'cond':
        # Get data iterators
        data_loader_sample = data_loader.get_semantic_sample_data(config)
        logging.info('Sample data loaded')

        for i, (img, target, file_name) in enumerate(data_loader_sample):
            img, target = img.to(config.device), target.to(config.device, dtype=torch.float32)
            file_name = ''.join(file_name)

            # Get sampling function
            sampling_fn = get_semantic_synthesis_sampler(config, sde, score_model, sem_seg_model, target)
            samples = sampling_fn()

            nrow = int(np.sqrt(samples.shape[0]))
            image_grid = make_grid(samples, nrow, padding=2, normalize=True)
            save_image(image_grid, os.path.join(sample_dir, file_name + '_sample_0.000025_0.7_0.4_8labels_1000steps.png'))
            save_image(img, os.path.join(sample_dir, file_name + '_original.png'))

            # Save original mask as color image
            if config.data.dataset == 'cityscapes256':
                cityscapes256.save_colorful_images(target, sample_dir, file_name + '_mask.png')
            if config.data.dataset == 'ade20k':
                pass
                #ade20k.save_colorful_images(target, sample_dir, file_name + '_mask.png')
            if config.data.dataset == 'flickr':
                flickr.save_colorful_images(target, sample_dir, file_name + '_mask.png')

            logging.info(f'Generated sample {i + 1} of {len(data_loader_sample)}')
    elif mode == 'uncond':
        sampling_shape = (config.sampling.batch_size, config.data.n_channels,
                          config.sampling.sampling_height, config.sampling.sampling_width)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, config.model.sampling_eps)
        for i in range(0, 20):
            samples, n = sampling_fn(score_model)
            nrow = int(np.sqrt(samples.shape[0]))
            image_grid = make_grid(samples, nrow, padding=2, normalize=True)
            save_image(image_grid, os.path.join(sample_dir, f'{i}_uncond_sample.png'))

            logging.info(f'Generated sample {i + 1} of {20}')





