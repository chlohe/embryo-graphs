import torch
import numpy as np
from torch.utils.data import Dataset

from . import T4SegmentationDataset2D

class T4SegmentationDataset2DDepthAsClass(Dataset):

    def __init__(self, data_dir='/datasets/train/stacks/static', label_dir='/datasets/train/seg/static', plane_count=11,
                 plane_size=(400, 400), use_normalisation=True, augmentations=None, use_augmentations=False, 
                 num_folds=1, fold=0, invert_selection=False, verbose=False):
        self.dataset = T4SegmentationDataset2D(data_dir=data_dir,
                                               label_dir=label_dir,
                                               plane_count=plane_count, 
                                               plane_size=plane_size, 
                                               use_normalisation=use_normalisation,
                                               augmentations=augmentations,
                                               use_augmentations=use_augmentations,
                                               num_folds=num_folds,
                                               fold=fold,
                                               invert_selection=invert_selection,
                                               verbose=verbose)

    def __getitem__(self, index):
        # Get example from segmentation dataset
        item, label = self.dataset[index]
        label = {
            'boxes': label['boxes'],
            'labels': (label['depths'] * 11).type(torch.int64),
            'masks': label['masks'],
            'image_id': torch.tensor(0),
            'area': label['area'],
            'iscrowd': torch.zeros((len(label['boxes']),)).int()
        }
        if 11 not in item.shape:
	    raise Exception('Incorrect number of planes! Have you tried Super-Focusing?')
        return item, label

    def __len__(self):
        return len(self.dataset)

if __name__ == '__main__':
    d = T4SegmentationDataset2DDepthAsClass()
    print(len(d))
    print(d[1][1])
