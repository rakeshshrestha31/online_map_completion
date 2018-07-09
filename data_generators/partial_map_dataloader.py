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
        files = [{
            'info':     files[i][0],
            'costmap':  files[i][1]
        } for i in range(len(files))]

        for i in files: print(i)


if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description='partial map data generator test')
    parser.add_argument(
        'dataset_dir', type=str, default='.',
        metavar='S', help='dataset directory'

    )

    args = parser.parse_args()

    dataset = PartialMapDataset(args.dataset_dir)
