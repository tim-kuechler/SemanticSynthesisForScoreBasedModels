from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np
from random import randint
import torch
import torch.nn.functional
from scipy.io import loadmat


colors = loadmat('/export/data/tkuechle/datasets/ade20k/color150.mat')['colors']
class ADE20K(Dataset):
    """Cityscapes256 Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``training`` or ``validation``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
    """
    def __init__(self, root, train=True, crop=True):
        self.root = root
        split = 'training' if train else 'validation'
        self.images_dir = os.path.join(self.root, 'images', split)
        self.targets_dir = os.path.join(self.root, 'annotations', split)
        self.file_names = []
        self.images = []
        self.targets = []
        self.n_labels = 151
        self.crop = crop

        for file_name in os.listdir(self.images_dir):
            target_name = file_name.replace('.jpg', '.png')

            self.file_names.append(file_name)
            self.images.append(os.path.join(self.images_dir, file_name))
            self.targets.append(os.path.join(self.targets_dir, target_name))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target)
        """
        #Load and resize
        file_name = self.file_names[index]
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        too_small = np.minimum(image.size[0], image.size[1]) < 256
        if too_small:
            scale = (256 / np.minimum(image.size[0], image.size[1])) + 0.1
            image = F.resize(image, [int(image.size[1] * scale), int(image.size[0] * scale)], interpolation=F.InterpolationMode.BICUBIC)
            target = F.resize(target, [int(image.size[1] * scale), int(image.size[0] * scale)], interpolation=F.InterpolationMode.NEAREST)

        #Crop
        if self.crop:
            top = randint(0, image.size[1] - 256)
            left = randint(0, image.size[0] - 256)
            image = F.crop(image, top, left, 256, 256)
            target = F.crop(target, top, left, 256, 256)

        #To tensor
        image = F.to_tensor(image)
        target = F.to_tensor(target) * 255
        target = target.long()
        target = torch.squeeze(target, dim=0)

        target = torch.nn.functional.one_hot(target, num_classes=self.n_labels).permute(2, 0, 1)

        file_name = ''.join(file_name)[:-4]
        return image, target, file_name

    def __len__(self):
        return len(self.images)

# Code beneath borrowed and adaped from https://github.com/CSAILVision/semantic-segmentation-pytorch
def save_colorful_images(pred, output_dir, filename):
    # print predictions in descending order
    pred = np.int32(pred.detach().cpu().numpy())

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    #im_vis = np.concatenate((img, pred_color), axis=1)

    Image.fromarray(pred_color).save(os.path.join(output_dir, filename))

def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb

def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret