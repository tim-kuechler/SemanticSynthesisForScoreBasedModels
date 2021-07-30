from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.datasets import SVHN
from torchvision.datasets import CIFAR10
from .flickr.flickr import FLICKR
from .cityscapes256.cityscapes256 import CITYSCAPES256
from .ade20k.ade20k import ADE20K
import torch
import os


def get_dataset(config):
    batch_size = config.training.batch_size
    if batch_size % torch.cuda.device_count() != 0:
        raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                         f'the number of devices ({torch.cuda.device_count()})')

    if config.data.dataset == 'flickr':
        dataset_train = FLICKR(train=True)
        data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
        dataset_eval = FLICKR(train=False)
        data_loader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, num_workers=4)
    if config.data.dataset == 'cityscapes256':
        dataset_train = CITYSCAPES256(root='/export/data/tkuechle/datasets/cityscapes_full', split='train', mode='fine',
                                      crop=config.data.crop_to_square)
        data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
        dataset_eval = CITYSCAPES256(root='/export/data/tkuechle/datasets/cityscapes_full', split='test', mode='fine',
                                     crop=config.data.crop_to_square)
        data_loader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, num_workers=4)
    if config.data.dataset == 'ade20k':
        dataset_train = ADE20K('/export/data/tkuechle/datasets/ade20k', train=True, crop=True)
        data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
        dataset_eval = ADE20K('/export/data/tkuechle/datasets/ade20k', train=False, crop=True)
        data_loader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, num_workers=4)

    return data_loader_train, data_loader_eval


def get_semantic_sample_data(config):
    dataset_dir = config.sampling.sample_data_dir

    if config.data.dataset == 'flickr':
        dataset = FLICKR(train=False, sample=True, scale_and_crop=False, root=dataset_dir)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    if config.data.dataset == 'cityscapes256':
        dataset = CITYSCAPES256(root=dataset_dir, split='val', mode='fine', crop=False)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    if config.data.dataset == 'ade20k':
        dataset = ADE20K(dataset_dir, train=False, crop=True)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    return data_loader
