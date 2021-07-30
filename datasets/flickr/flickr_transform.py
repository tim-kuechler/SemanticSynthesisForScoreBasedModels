"""
Script that transform the file structure of the flickr dataset such that other programs can handle it easier.

The flickr dataset has subfolders e.g. sunset/img123.png. Some other programs tested for the paper need a dataset
without subfolders.
"""
import os
from shutil import copyfile
from pathlib import Path


def transform():
    data_csv = 'flickr_landscapes_train_split.txt'
    data_root = '/export/data/compvis/datasets/rfw/flickr/data/'
    segmentation_root = '/export/data/compvis/datasets/rfw/segmentation/flickr_segmentation_v2/'

    with open(data_csv, 'r') as f:
        image_paths = f.read().splitlines()

    i = 0
    for img_name in image_paths:
        seg_path = os.path.join(segmentation_root, img_name.replace('.jpg', '.png'))
        img_path = os.path.join(data_root, img_name)

        img_dir = '/export/home/tkuechle/datasets/flickr/img'
        Path(img_dir).mkdir(parents=True, exist_ok=True)
        seg_dir = '/export/home/tkuechle/datasets/flickr/seg'
        Path(seg_dir).mkdir(parents=True, exist_ok=True)

        copyfile(img_path, os.path.join(img_dir, img_name.split('/')[1]))
        copyfile(seg_path, os.path.join(seg_dir, img_name.replace('.jpg', '.png').split('/')[1]))
        i += 1
        if i % 100 == 0:
            print('step: ', i)


if __name__ == '__main__':
    transform()