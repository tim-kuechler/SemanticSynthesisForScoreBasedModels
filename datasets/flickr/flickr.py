import torch.nn.functional
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
from random import randint
from collections import namedtuple
from torchvision.utils import make_grid, save_image


id2trainId = {}
Label = namedtuple('Label', [
                   'name',
                   'id',
                   'trainId'])

labels = [
    #       name                    id    trainId
    Label('mountain1'           ,  134  ,      1),
    Label('sea1'                ,  154  ,      2),
    Label('sea2'                ,  147  ,      2),
    Label('sea3'                ,  177  ,      2),
    Label('clouds'              ,  105  ,      3),
    Label('sky'                 ,  156  ,      4),
    Label('forest'              ,  168  ,      5),
    Label('grass1'              ,  123  ,      6),
    Label('grass3'              ,  141  ,      6)
]

labels_sample = [
    #       name                    id    trainId
    Label('mountain'          ,  60   ,      1),
    Label('sea'                ,  97   ,      2),
    Label('clouds'             ,  147  ,      3),
    Label('sky'                ,  238  ,      4),
    Label('forest'             ,  89   ,      5),
    Label('grass'              ,  210  ,      6)
]

class FLICKR(Dataset):
    """S-Flickr Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
    """
    def __init__(self, root=None, sample=False, train=True, scale_and_crop=True):
        self.root = root
        self.sample = sample
        self.n_labels = 8
        self.scale_and_crop = scale_and_crop

        if train:
            self.data_csv = './datasets/flickr/flickr_landscapes_train_split.txt'
        else:
            self.data_csv = './datasets/flickr/flickr_landscapes_val_split.txt'

        if root is None:
            self.images_dir = '/export/data/compvis/datasets/rfw/flickr/data/'
            self.targets_dir = '/export/data/compvis/datasets/rfw/segmentation/flickr_segmentation_v2/'
        else:
            self.images_dir = os.path.join(root, 'data')
            self.targets_dir = os.path.join(root, 'segmentation')
            self.data_csv = os.path.join(root, 'maps.txt')

        self.file_names = []
        self.images = []
        self.targets = []
        with open(self.data_csv, 'r') as f:
            image_paths = f.read().splitlines()
            if self.sample:
                for p in image_paths:
                    self.file_names.append(p)
                    self.targets.append(os.path.join(self.targets_dir, p))
            else:
                for p in image_paths:
                    self.file_names.append(p)
                    self.images.append(os.path.join(self.images_dir, p))
                    self.targets.append(os.path.join(self.targets_dir, p.replace('.jpg', '.png')))

        # change id to trainId
        id2trainId[str(0)] = 0  # add an void class
        for obj in labels_sample if self.sample else labels:
            trainId = obj.trainId
            id = obj.id
            id2trainId[str(id)] = trainId


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target)
        """
        #Load and resize
        if self.sample:
            file_name = self.file_names[index]
            target = Image.open(self.targets[index])
            target = F.to_tensor(target) * 255
            target = target.long()
            target = torch.squeeze(target, dim=0)

            # Change labels in target to train ids
            for h in range(target.shape[0]):
                for w in range(target.shape[1]):
                    id = target[h, w].item()
                    try:
                        trainId = id2trainId[str(id)]
                    except:
                        trainId = id2trainId[str(0)]
                    target[h, w] = trainId

            target = torch.nn.functional.one_hot(target, num_classes=self.n_labels).permute(2, 0, 1)

            file_name = ''.join(file_name)[:-4]
            return torch.ones((3, target.shape[1], target.shape[2])), target, file_name
        else:
            file_name = self.file_names[index]
            image = Image.open(self.images[index]).convert('RGB')
            target = Image.open(self.targets[index])
            if self.scale_and_crop:
                too_small = np.minimum(image.size[0], image.size[1]) < 512
                if too_small:
                    scale = (512 / np.minimum(image.size[0], image.size[1])) + 0.1
                    image = F.resize(image, [int(image.size[1] * scale), int(image.size[0] * scale)], interpolation=F.InterpolationMode.BICUBIC)
                    target = F.resize(target, [int(image.size[1] * scale), int(image.size[0] * scale)], interpolation=F.InterpolationMode.NEAREST)

                #Crop
                top = randint(0, image.size[1] - 512)
                left = randint(0, image.size[0] - 512)
                image = F.crop(image, top, left, 512, 512)
                target = F.crop(target, top, left, 512, 512)

            #To tensor
            image = F.to_tensor(image)
            target = F.to_tensor(target) * 255
            target = torch.squeeze(target, dim=0)

            # Random flip
            if torch.rand(1) < 0.5:
                image = F.hflip(image)
                target = F.hflip(target)

            # Change labels in target to train ids
            for h in range(target.shape[0]):
                for w in range(target.shape[1]):
                    id = target[h, w].item()
                    try:
                        trainId = id2trainId[str(id)]
                    except:
                        trainId = id2trainId[str(0)]
                    target[h, w] = trainId

            target = torch.nn.functional.one_hot(target.long(), num_classes=self.n_labels).permute(2, 0, 1)

            file_name = file_name.split('/')[1]
            file_name = ''.join(file_name)[:-4]
            return image, target, file_name

    def __len__(self):
        return len(self.images) if len(self.images) != 0 else len(self.targets)

FLICKR_PALETTE = np.asarray([
    [0, 0, 0],
    [128, 64, 64],
    [54, 62, 167],
    [170, 170, 170],
    [95, 219, 255],
    [47, 140, 92],
    [29, 195, 49],
    [164, 219, 216]], dtype=np.uint8)

# borrow functions and modify it from https://github.com/fyu/drn/blob/master/segment.py
def save_colorful_images(pred, output_dir, filename):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    pred = torch.argmax(pred, dim=1)
    pred_cp = pred
    pred = pred.cpu().numpy()
    imgs = []
    for i in range(pred_cp.shape[0]):
        im = Image.fromarray(FLICKR_PALETTE[pred[i].squeeze()])
        imgs.append(torch.unsqueeze(F.to_tensor(im), dim=0))

    image = torch.cat(imgs, dim=0)
    nrow = int(np.sqrt(image.shape[0]))
    image_grid = make_grid(image, nrow, padding=2)
    save_image(image_grid, os.path.join(output_dir, filename))
