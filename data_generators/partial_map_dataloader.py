#!/usr/bin/env python3

"""@package docstring
Implementation of pytorch util.data.dataset interface for partial map dataset"""

#custom imports
import utils

# pytorch imports
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.utils

# standard library imports
import os
import numpy as np
import cv2
import functools
from collections import OrderedDict
import yaml
import json
import glob
import re
import functools
import typing

class PartialMapDataset(Dataset):
    """Partial map dataset implementing torch.utils.data.Dataset interface
    """

    def __init__(self, dataset_dir: typing.Type[str]):
        """
        parses the directory get group all the relevant files into a dataset
        :param dataset_dir: dataset directory
        """
        self.dataset_dir = dataset_dir

        info_files = glob.glob(os.path.join(dataset_dir, '**', 'info*.yaml'), recursive=True)

        regex = re.compile(r'.*info(\d+).yaml$')
        file_indices = [re.search(regex, info_file).group(1) for info_file in info_files]

        costmap_files = [
            os.path.join(
                '/'.join(info_files[i].split('/')[0:-1]),
                'costmap' + str(file_indices[i]) + '.png'
            )
            for i in range(len(file_indices))
        ]

        unfiltered_files = list(zip(info_files, costmap_files))

        files = list(filter(
            lambda file_group: \
                functools.reduce(lambda is_file, x: is_file and os.path.isfile(x), file_group, True),
            unfiltered_files
        ))

        if len(unfiltered_files) != len(files):
            print('[LOG] {} corresponding files missing'.format(len(unfiltered_files) - len(files)))

        # make proper dictionary for better indexing
        self.files = [{
            'info':     files[i][0],
            'costmap':  files[i][1]
        } for i in range(len(files))]


    def __len__(self):
        """
        :return: total maps in the dataset
        """
        return len(self.files)

    def __getitem__(self, item):
        info = None

        import time
        with open(self.files[item]['info'], 'r') as f:
            try:
                info = yaml.load(f, Loader=yaml.CLoader)
            except yaml.YAMLError as e:
                print(e)
                return None

        # print(json.dumps(info, indent=4))
        if info is not None:
            return cv2.imread(self.files[item]['costmap'])


if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description='partial map data generator test')

    parser.add_argument(
        '--shuffle', dest='is_shuffle', action='store_true',
        help='shuffle the data (default true)'
    )
    parser.add_argument(
        '--no-shuffle', dest='is_shuffle', action='store_false',
        help='do not shuffle the data (default true)'
    )
    parser.set_defaults(is_shuffle=True)

    parser.add_argument(
        '--batch-size', type=int, default=4,
        metavar='N', help='batch size'
    )

    parser.add_argument(
        'dataset_dir', type=str, default='.',
        metavar='S', help='dataset directory'

    )

    args = parser.parse_args()

    dataset = PartialMapDataset(args.dataset_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.is_shuffle, num_workers=4)

    for batch_idx, data in enumerate(dataloader):
        print(data.size())