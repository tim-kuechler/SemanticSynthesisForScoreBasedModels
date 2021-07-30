"""
Calculates the maximum pairwise distance of the used datasets. This value should then be used as
sigma max in the model.
"""
from cityscapes256.cityscapes256 import CITYSCAPES256
from flickr.flickr import FLICKR
from ade20k.ade20k import ADE20K
from torch.utils.data import DataLoader
import torch
import copy


def calc_max_pairwise_distance(dataset, max_steps=20000000):
    if dataset == 'cityscapes256':
        dataset = CITYSCAPES256(root='/export/data/tkuechle/datasets/cityscapes_full', split='train', mode='fine', crop=False)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    elif dataset == 'flickr':
        dataset = FLICKR(train=True)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    elif dataset == 'ade20k':
        dataset = ADE20K('/export/data/tkuechle/datasets/ade20k', train=True, crop=True)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    steps = 0
    max_dist = 0.
    imgs_checked = []
    for i, (img1, _) in enumerate(data_loader):
        img1 = torch.squeeze(img1)
        img1_cp = copy.deepcopy(img1)
        del img1
        img1 = img1_cp
        for img2 in imgs_checked:
            dist = torch.norm(img2 - img1).item()
            if dist > max_dist:
                max_dist = dist

            steps += 1
            if steps % 1000 == 0:
                print(f'Step {steps} of {max_steps}. Curr. max pairw. dist.: {max_dist}, img: {i}')

        if steps >= max_steps:
            print(f'Finished: Max pairw. dist.: {max_dist}, images checked: {i}')
            break
        imgs_checked.append(img1)


if __name__ == '__main__':
    calc_max_pairwise_distance('ade20k')