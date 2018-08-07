#!/usr/bin/env python3

"""@package docstring
Implementation of pytorch util.data.dataset interface for partial map dataset"""

#custom imports
import utils
from kth.FloorPlanGraph import FloorPlanGraph
import utils.constants
from data_generators.OneDataInfo import OneDataInfo
from utils.generator_utils import collate_without_batching_dict

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
# import yaml
import json
import glob
import re
import functools
import typing
import itertools



class PartialMapDataset(Dataset):
    """Partial map dataset implementing torch.utils.data.Dataset interface
    """

    def __init__(self, partial_dataset_dir: typing.Type[str], original_dataset_dir: typing.Type[str]):
        """
        parses the directory get group all the relevant files into a dataset
        :param partial_dataset_dir: partial map dataset directory
        :param original_dataset_dir: original dataset dir with xml annotation files
        """
        self.partial_dataset_dir = partial_dataset_dir
        self.original_dataset_dir = original_dataset_dir

        # each data corresponding to a json file, we can generate multiple samples from each data
        self.total_data_len = 0  # the number of all samples
        self.all_data_lens = []   # a list to store the number of samples in each data
        self.accumulate_lens = 0  # a list to store the accumulated number of samples till current data
        self.all_data_info = []  # store the data info for each data
        self.ground_truth_dict = dict()

        info_files = glob.glob(os.path.join(partial_dataset_dir, '**', 'info*.json'), recursive=True)

        self.all_data_info = [OneDataInfo(json_path) for json_path in info_files]

        # construct a dictionary [name:floorplan] for all floorplans ground_truth
        xml_files = glob.glob(os.path.join(original_dataset_dir, '**', '*.xml'), recursive=True)
        xml_files_split = [i.split('/') for i in xml_files]
        xml_names = [split[-1][0:-4] for split in xml_files_split]
        floorplans = [FloorPlanGraph(file_path=xml_file) for xml_file in xml_files]

        original_size = (utils.constants.ORIGINAL_WIDTH, utils.constants.ORIGINAL_HEIGHT)
        ground_truth_images = [(floorplan.to_image(utils.constants.ORIGINAL_RESOLUTION,
                                                   original_size)*255).astype(dtype=np.uint8)
            for floorplan in floorplans]
        self.ground_truth_dict = dict(zip(xml_names, ground_truth_images))

        # compute all lens and accumulate_lens for fast indexing
        self.all_data_lens = [info.__len__() for info in self.all_data_info]
        self.accumulate_lens = list(itertools.accumulate(self.all_data_lens))
        self.total_data_len = self.accumulate_lens[-1]

    def _index_of_sample(self, index) -> (int, int):
        """
        get the index of floorplan and the index of sample in this floorplan
        :param index: index in all samples
        :return: (int, int) -> (data_index, sample_index_in_floorplan)
        """
        if index >= self.total_data_len or index < 0:
            raise IndexError("Error index when get item.")

        data_index = 0
        while not index < self.accumulate_lens[data_index]:
            data_index += 1

        if data_index != 0:
            sample_index = index - self.accumulate_lens[data_index - 1]
        else:
            sample_index = index

        return data_index, sample_index

    def __len__(self):
        """
        :return: total maps in the dataset
        """
        return self.total_data_len


    def __getitem__(self, item):
        """

        :param item: item index
        :return: input_data (C x H x W) and ground_truth (H x W) where C-channel, W-width, H-height and frontiers
        """
        data_index, sample_index = self._index_of_sample(item)
        input_image, ground_truth_image, frontiers = \
            self.all_data_info[data_index][sample_index, self.ground_truth_dict]

        return input_image, ground_truth_image, frontiers


if __name__ == '__main__':
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
        'partial_dataset_dir', type=str, default='.',
        metavar='S', help='partial map dataset directory'

    )

    parser.add_argument(
        'original_dataset_dir', type=str, default='.',
        metavar='S', help='original kth dataset directory'

    )

    args = parser.parse_args()

    dataset = PartialMapDataset(args.partial_dataset_dir, args.original_dataset_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_without_batching_dict,
                            shuffle=args.is_shuffle, num_workers=1)

    for batch_idx, batch_data in enumerate(dataloader):
        input, target = batch_data
        print('input', input.size())
        print('target', target.size())

        import utils.vis_utils

        # shift the transparency channel to show translucent non-frontiers
        input = utils.vis_utils.get_transparancy_adjusted_input(input)

        # collated GT and input
        mixed_data = torch.FloatTensor(2 * args.batch_size, *(input[0].size()))
        mixed_data[0::2] = input
        mixed_data[1::2] = utils.vis_utils.get_padded_occupancy_grid(target)
        mixed_data[1::2, -1, :, :] = input[:, -1, :, :]

        grid = utils.vis_utils.make_grid(mixed_data, nrow=int(args.batch_size / 2), padding=0)

        ## overlapping GT and input
        # grid = utils.vis_utils.make_grid(
        #     input + torch.cat([target, torch.ones(target.size(0), 1, *(target.shape[2:]))], dim=1),
        #     nrow=int(args.batch_size / 2), padding=0
        # )
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        plt.imshow(ndarr)
        cv2.imwrite('/tmp/training_input.png', ndarr)
        plt.show()
