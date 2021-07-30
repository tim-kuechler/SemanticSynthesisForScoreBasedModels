import os
from PIL import Image
import torchvision.transforms.functional as F
from collections import defaultdict
import torch


def find_labels():
    data_csv = 'flickr_landscapes_train_split.txt'
    targets_dir = '/export/data/compvis/datasets/rfw/segmentation/flickr_segmentation_v2/'

    targets = []
    with open(data_csv, 'r') as f:
        image_paths = f.read().splitlines()
        for p in image_paths:
            targets.append(os.path.join(targets_dir, p.replace('.jpg', '.png')))

    labels = defaultdict(int)
    for i in range(len(targets)):
        target = Image.open(targets[i])
        target = F.to_tensor(target) * 255
        target = target.long()
        target = torch.squeeze(target, dim=0)
        target = torch.unique(target)
        for k in range(target.shape[0]):
            id = target[k].item()
            labels[str(id)] += 1

        if i % 10 == 0:
            print(f'Img {i}/{len(targets)}')

    print(labels)

if __name__ == '__main__':
    find_labels()
