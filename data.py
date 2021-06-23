"""Dataset utilities
"""
import torch
import cv2
import re
import yaml
import glob
import os
import hydra
import numpy as np

from utils import *
from tqdm import tqdm
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.panoptic.scannet import ScannetPanoptic
from torch_points3d.metrics.panoptic_tracker import PanopticTracker
from torch_geometric.data import InMemoryDataset


class SingleSceneDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None):
        super(SingleSceneDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = self.processed_paths[0]

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



@hydra.main(config_path='./config/config.yaml')
def main(hparams):
   pass 


if __name__ == '__main__':
    main(j
