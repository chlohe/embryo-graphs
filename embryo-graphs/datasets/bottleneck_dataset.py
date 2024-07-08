import os.path as osp
import glob
import torch

from torch.utils.data import Dataset

class BottleneckDataset(Dataset):
    def __init__(self, bottlenecks_dir='data/bottlenecks'):
        self.bottleneck_paths = glob.glob(osp.join(bottlenecks_dir, '*.bottleneck'))

    def __getitem__(self, idx):
        return torch.load(self.bottleneck_paths[idx])

    def __len__(self):
        return len(self.bottleneck_paths)