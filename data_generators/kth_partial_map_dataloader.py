#!/usr/bin/env python3

"""@package docstring
Implementation of pytorch util.data.dataset interface for partial map dataset"""

#custom imports
import utils
from kth.FloorPlanGraph import FloorPlanGraph
import utils.constants

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

        info_files = glob.glob(os.path.join(partial_dataset_dir, '**', 'info*.json'), recursive=True)

        regex = re.compile(r'.*info(\d+).json')
        file_indices = [re.search(regex, info_file).group(1) for info_file in info_files]

        # <partial_dataset_dir>/[small,middle,large]/<floorplan_name>/<simulation_run_idx>/costmap<iter_idx>.png
        costmap_files = [
            os.path.join(
                '/'.join(info_files[i].split('/')[0:-1]),
                'costmap' + str(file_indices[i]) + '.png'
            )
            for i in range(len(file_indices))
        ]

        info_files_split = [i.split('/') for i in info_files]

        # <partial_dataset_dir>/[small,middle,large]/<floorplan_name>/<simulation_run_idx>/boundingBox<iter_idx>.png
        bounding_box_files = [
            os.path.join(
                '/'.join(info_files_split[i][0:-1]),
                'boundingBox' + str(file_indices[i]) + '.png'
            )
            for i in range(len(file_indices))
        ]

        # TODO: avoid redundant ground truth names for each simulation and iteration. Use indexing rather than filenames
        # TODO: maybe load all the ground truth maps for efficiency
        # <partial_dataset_dir>/[small,middle,large]/<floorplan_name>/GT.bmp
        ground_truth_files = [
            os.path.join(
                '/'.join(info_files_split[i][0:-2]),
                'GT' + '.bmp'
            )
            for i in range(len(file_indices))
        ]

        # construct a dictionary [name:floorplan] for all floorplans
        xml_files = glob.glob(os.path.join(original_dataset_dir, '**', '*.xml'), recursive=True)
        xml_files_split = [i.split('/') for i in xml_files]
        xml_names = [split[-1][0:-4] for split in xml_files_split]
        floorplans = [FloorPlanGraph(file_path=xml_file) for xml_file in xml_files]
        self.floorplan_dict = dict(zip(xml_names, floorplans))


        # xml file names
        xml_names = [info_files_split[i][-3]
            for i in range(len(file_indices))
        ]

        unfiltered_files = [{
            'info_file': info_files[i],
            'costmap_file': costmap_files[i],
            'bounding_box_file': bounding_box_files[i],
            'ground_truth_file': ground_truth_files[i],
            'xml_name': xml_names[i]
        } for i in range(len(file_indices))]

        self.dataset_meta_info = unfiltered_files

        # self.dataset_meta_info = list(filter(
        #     lambda file_group: \
        #         # check if each file in file group (info, costmap, bounding box, ground truth) exist
        #         functools.reduce(lambda is_file, x: is_file and os.path.isfile(file_group[x]), file_group, True),
        #     unfiltered_files
        # ))

        if len(unfiltered_files) != len(self.dataset_meta_info):
            print('[LOG] {} corresponding files missing'.format(len(unfiltered_files) - len(self.dataset_meta_info)))

        # parse floorplan into graph for later efficiency
        # self.dataset_meta_info = [
        #     dict(floorplan_graph=FloorPlanGraph(file_path=meta_info['xml_annotation_file']), #FloorPlanGraph(),
        #          **meta_info)
        #     for meta_info in self.dataset_meta_info
        # ]

    def get_bounding_box_image(self, bounding_boxes, image_size):
        """
        get bounding box image for generating masks around frontiers
        :param bounding_boxes: list of bounding box specs (x, y, width, height) as dict
        :param image_size: size of the image the bounding box is part of
        :return: numpy image of type float32 of size param image_size
        """
        output_image = np.zeros(image_size, dtype=np.uint8)
        for bounding_box in bounding_boxes:
            center = {
                'x': bounding_box['x'] + np.round(bounding_box['width'] / 2),
                'y': bounding_box['y'] + np.round(bounding_box['height'] / 2)
            }

            new_size = {
                'x': int(bounding_box['width'] * utils.constants.FRONTIER_MASK_RESIZE_FACTOR),
                'y': int(bounding_box['height'] * utils.constants.FRONTIER_MASK_RESIZE_FACTOR)
            }
            new_left_corner = {
                'x': int(center['x'] - np.round(new_size['x'] / 2)),
                'y': int(center['y'] - np.round(new_size['y'] / 2))
            }

            new_right_corner = {
                'x':   int(new_left_corner['x'] + np.round(new_size['x'])),
                'y':   int(new_left_corner['y'] + np.round(new_size['y']))
            }

            cv2.rectangle(
                output_image,
                (new_left_corner['x'], new_left_corner['y']),
                (new_right_corner['x'], new_right_corner['y']),
                255, -1
            )
        return output_image

    def __len__(self):
        """
        :return: total maps in the dataset
        """
        return len(self.dataset_meta_info)

    def get_best_resolution(self, original_resolution, original_size, target_size):
        """

        :param original_resolution:
        :param original_size: (y, x) size (matrix form)
        :param target_size:
        :return: coarsest resolution among the two x and y
        """
        resolutions = list([
            original_resolution * original_size[i] / target_size[i] for i in range(2)
        ])

        return max(resolutions)

    def pad_image(self, input_image, target_size, trucate_function):
        """

        :param input_image:
        :param target_size: (y, x) size (matrix form)
        :param trucate_function truncate function to use (ceil, floor, round etc)
        :return: padded image
        """
        input_size = input_image.shape
        output_image = np.zeros(
            (target_size[0], target_size[1], input_size[2] if len(input_size) == 3 else 1),
            dtype=input_image.dtype
        )
        start = [int(trucate_function((target_size[i] - input_size[i]) / 2.0)) for i in range(2)]

        output_image[start[0]:(start[0] + input_size[0]), start[1]:(start[1] + input_size[1])] = input_image

        return output_image

    def __getitem__(self, item) -> (torch.FloatTensor, torch.FloatTensor, dict, (int, int), float):
        """

        :param item: item index
        :return: input (B x C x H x W), target (B x 1 x H x W), where B-batch size, C-channel, W-width, H-height
                info about frontiers (dict),
                (u, v) co-ordinates of origin (translation from image frame to world co-ordinates)
                resolution (scale factor from image to world co-ordinates)
                (i.e. [x, y] = [(u - center_u) * resolution, (v - center_v) * resolution]
        """

        info = None
        with open(self.dataset_meta_info[item]['info_file'], 'r') as f:
            try:
                info = json.load(f)
            except Exception as e:
                print(e)

        if info is None:
            raise Exception('error loading info (yaml) file ')

        # print(json.dumps(info, indent=4))


        costmap_image = cv2.imread(self.dataset_meta_info[item]['costmap_file'], cv2.IMREAD_COLOR)
        original_costmap_size = costmap_image.shape

        best_resolution = self.get_best_resolution(
            utils.constants.ORIGINAL_RESOLUTION,
            original_costmap_size,
            (utils.constants.TARGET_HEIGHT, utils.constants.TARGET_WIDTH)
        )

        best_size = tuple(reversed([
            int(np.round(utils.constants.ORIGINAL_RESOLUTION / best_resolution * original_costmap_size[i]))
            for i in range(2)
        ]))

        costmap_image = cv2.resize(
            costmap_image,
            # (utils.constants.TARGET_WIDTH, utils.constants.TARGET_HEIGHT)
            best_size,
            utils.constants.RESIZE_INTERPOLATION
        )

        costmap_image = self.pad_image(
            costmap_image,
            (utils.constants.TARGET_HEIGHT, utils.constants.TARGET_WIDTH),
            np.ceil
        )

        # todo: instead of resizing, get the ground truth and bounding box images in the desired resolution

        # ground_truth_image = cv2.imread(self.dataset_meta_info[item]['ground_truth_file'], cv2.IMREAD_GRAYSCALE)

        ground_truth_image = self.floorplan_dict[self.dataset_meta_info[item]['xml_name']].to_image(
            best_resolution, # utils.constants.ORIGINAL_RESOLUTION * original_costmap_size[1] * 1.0 / utils.constants.TARGET_WIDTH,
            best_size
            # (original_costmap_size[1], original_costmap_size[0])
        )

        ground_truth_image = np.expand_dims(ground_truth_image, -1)

        ground_truth_image = self.pad_image(
            ground_truth_image,
            (utils.constants.TARGET_HEIGHT, utils.constants.TARGET_WIDTH),
            np.ceil
        )
        # ground_truth_image = cv2.resize(ground_truth_image, (utils.constants.WIDTH, utils.constants.HEIGHT))

        # gaussian blur (so that the correct prediction space is not a single pixel wide)
        ground_truth_image = cv2.GaussianBlur(ground_truth_image, (3, 3), 0)

        # bounding_box_image = cv2.imread(self.dataset_meta_info[item]['bounding_box_file'], cv2.IMREAD_GRAYSCALE)
        bounding_box_image = self.get_bounding_box_image(info['BoundingBoxes'], original_costmap_size[:2])

        ground_truth_image = np.expand_dims(ground_truth_image, -1)

        bounding_box_image = cv2.resize(
            bounding_box_image,
            # (utils.constants.TARGET_WIDTH, utils.constants.TARGET_HEIGHT)
            best_size,
            utils.constants.RESIZE_INTERPOLATION
        )

        bounding_box_image = np.expand_dims(bounding_box_image, -1)

        bounding_box_image = self.pad_image(
            bounding_box_image,
            (utils.constants.TARGET_HEIGHT, utils.constants.TARGET_WIDTH),
            np.ceil
        )
        # bounding_box_image = cv2.resize(bounding_box_image, (utils.constants.TARGET_WIDTH, utils.constants.TARGET_HEIGHT))



        # dims H x W x C
        input_image = np.concatenate((costmap_image, bounding_box_image), axis=-1)

        # dims C x H x W
        input_image = input_image.transpose(2, 0, 1)
        ground_truth_image = ground_truth_image.transpose(2, 0, 1)

        # normalize
        input_image = input_image.astype(dtype=np.float32)
        input_image /= 255.0

        ground_truth_image = ground_truth_image.astype(dtype=np.float32)

        return input_image, ground_truth_image, info, tuple([i/2 for i in best_size]), best_resolution


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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.is_shuffle, num_workers=1)

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

        mixed_data[1::2, 0, :, :] = mixed_data[0::2, 2, :, :]

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
