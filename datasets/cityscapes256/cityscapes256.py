"""Adapted from torchvision.datasets.cityscapes and https://github.com/pochih/FCN-pytorch/blob/master/python/Cityscapes_utils.py"""
from torch.utils.data import Dataset
import os
from PIL import Image
from collections import namedtuple
from os.path import exists, split
import torchvision.transforms.functional as F
from random import randint
import torch.nn.functional
import numpy as np


id2trainId = {}
Label = namedtuple('Label', [
                   'name',
                   'id',
                   'trainId',
                   'category',
                   'categoryId',
                   'hasInstances',
                   'ignoreInEval',
                   'color'])

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        1 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        2 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        3 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        4 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        5 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        6 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        7 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        8 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        9 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,       10 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       11 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       12 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       13 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       15 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       17 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       18 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       19 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

class CITYSCAPES256(Dataset):
    """Cityscapes256 Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
    """
    def __init__(self, root, split="train", mode="fine", crop=True):
        self.root = root
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.split = split
        self.file_names = []
        self.images = []
        self.targets = []
        self.n_labels = 20
        self.crop = crop

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in sorted(os.listdir(img_dir)):
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], f'{self.mode}_labelIds.png')

                self.images.append(os.path.join(img_dir, file_name))
                self.file_names.append(file_name)
                self.targets.append(os.path.join(target_dir, target_name))

        # change id to class index
        id2trainId[str(0)] = 0  # add an void class
        for obj in labels:
            if obj.ignoreInEval:
                continue
            idx = obj.trainId
            id = obj.id
            id2trainId[str(id)] = idx

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
        image = F.resize(image, [256, 512], interpolation=F.InterpolationMode.BICUBIC)

        target = Image.open(self.targets[index])
        target = F.resize(target, [256, 512], interpolation=F.InterpolationMode.NEAREST)

        #Crop
        if self.crop:
            left = randint(0, 256)
            image = F.crop(image, 0, left, 256, 256)
            target = F.crop(target, 0, left, 256, 256)

        #To tensor
        image = F.to_tensor(image)
        target = F.to_tensor(target) * 255
        target = target.long()
        target = torch.squeeze(target, dim=0)

        #Change labels in target to train ids
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
        return image, target, file_name

    def __len__(self):
        return len(self.images)

# borrow functions and modify it from https://github.com/fyu/drn/blob/master/segment.py
CITYSCAPE_PALETTE = np.asarray([
    [0, 0, 0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32]], dtype=np.uint8)


def save_colorful_images(pred, output_dir, filename):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    pred = torch.argmax(pred, dim=1)
    pred = pred.cpu().numpy()
    im = Image.fromarray(CITYSCAPE_PALETTE[pred[0].squeeze()])
    fn = os.path.join(output_dir, filename)
    out_dir = split(fn)[0]
    if not exists(out_dir):
        os.makedirs(out_dir)
    im.save(fn)