g#!/usr/bin/env python3

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

        bounding_box_files = [
            os.path.join(
                '/'.join(info_files[i].split('/')[0:-1]),
                'boundingBox' + str(file_indices[i]) + '.png'
            )
            for i in range(len(file_indices))
        ]

        ground_truth_files = [
            os.path.join(
                '/'.join(info_files[i].split('/')[0:-2]),
                'GT' + '.bmp'
            )
            for i in range(len(file_indices))
        ]

        unfiltered_files = [{
            'info_file': info_files[i],
            'costmap_file': costmap_files[i],
            'bounding_box_file': bounding_box_files[i],
            'ground_truth_file': ground_truth_files[i]
        } for i in range(len(file_indices))]

        self.files = list(filter(
            lambda file_group: \
                # check if each file in file group (info, costmap, bounding box, ground truth) exist
                functools.reduce(lambda is_file, x: is_file and os.path.isfile(file_group[x]), file_group, True),
            unfiltered_files
        ))

        if len(unfiltered_files) != len(self.files):
            print('[LOG] {} corresponding files missing'.format(len(unfiltered_files) - len(self.files)))


    def __len__(self):
        """
        :return: total maps in the dataset
        """
        return len(self.files)

    def __getitem__(self, item):
        """

        :param item: item index
        :return: input (B x C x W x H) and target (B X 1 X W X H) where B-batch size, C-channel, W-width, H-height
        """

        info = None
        with open(self.files[item]['info_file'], 'r') as f:
            try:
                info = yaml.load(f, Loader=yaml.CLoader)
            except yaml.YAMLError as e:
                print(e)

        if info is None:
            raise Exception('error loading info (yaml) file ')

        # print(json.dumps(info, indent=4))

        costmap_image = cv2.imread(self.files[item]['costmap_file'], cv2.IMREAD_COLOR)

        ground_truth_image = cv2.imread(self.files[item]['ground_truth_file'], cv2.IMREAD_GRAYSCALE)

        bounding_box_image = cv2.imread(self.files[item]['bounding_box_file'], cv2.IMREAD_GRAYSCALE)
        bounding_box_image = np.expand_dims(bounding_box_image, -1)

        # dims W x H x C
        input_image = np.concatenate((costmap_image, bounding_box_image), axis=-1)

        # dims C x W x H
        input_image = input_image.transpose(2, 0, 1)
        ground_truth_image = np.expand_dims(ground_truth_image, 0)

        # normalize
        input_image = input_image.astype(dtype=np.float32)
        input_image /= 255.0

        ground_truth_image = ground_truth_image.astype(dtype=np.float32)
        ground_truth_image /= 255.0

        return input_image, ground_truth_image


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
        '--batch-size', type=int, default=16,
        metavar='N', help='batch size'
    )

    parser.add_argument(
        'dataset_dir', type=str, default='.',
        metavar='S', help='dataset directory'

    )

    args = parser.parse_args()

    dataset = PartialMapDataset(args.dataset_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.is_shuffle, num_workers=4)

    for batch_idx, batch_data in enumerate(dataloader):
        input, target = batch_data
        print('input', input.size())
        print('target', target.size())

        import utils.vis_utils


        # shift the transparency channel to show translucent non-frontiers
        input[:, -1, :, :] += 0.1
        grid = utils.vis_utils.make_grid(input, nrow=int(np.sqrt(args.batch_size)), padding=0)

        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        plt.imshow(ndarr)
        cv2.imwrite('/tmp/training_input.png', ndarr)
        plt.show()
