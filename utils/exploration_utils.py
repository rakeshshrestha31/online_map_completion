#!/usr/bin/env python3
"""
@package docstring
utilities to help exploration
"""

# custom imports
import utils.vis_utils
from utils.vis_utils import save_image

# pytorch imports
import torch
from torch.autograd import Variable

# standard library imports
import os
import numpy as np
import typing


def compute_expected_information_gain(input: Variable, prediction: Variable,
                                      info: typing.List[dict]) \
    -> None: # typing.Tuple[typing.List[int], np.array]:
    """
    Computing information gain for each frontier group
    :param input: input to the network B x 4 x H x W (4 channels: unknown, free, obstacle, prediction mask)
    :param prediction: predicted occupancy map B x 1 x H x W (0 - free, 1 - occupied)
    :param info: list of dictionary containing groups of frontier points in "Frontiers" key
    :return: (information gain, image with expected information gain pixel set high
    """
    frontier_bb_mask = input[:, -1:, :, :].byte()
    unknown_mask = input[:, 0:1, 0, 0]

    frontier_points_mask = input.clone()
    frontier_points_mask[:, 0:-1, :, :] = 0

    for batch_idx in range(input.size(0)):
        for frontier_group in info[0]['Frontiers']:
            for frontier_point in frontier_group:
                input[batch_idx, 2, frontier_point[1], frontier_point[0]] = 1.0
                frontier_points_mask[batch_idx, 2, frontier_point[1], frontier_point[0]] = 1.0

    viz_data = [
        utils.vis_utils.get_transparancy_adjusted_input(frontier_points_mask),
        utils.vis_utils.get_transparancy_adjusted_input(
            utils.vis_utils.get_padded_occupancy_grid(prediction)
        ),
        utils.vis_utils.get_transparancy_adjusted_input(input)
    ]

    viz_data = torch.cat(viz_data, 0)
    save_image(viz_data.data.cpu(),
               os.path.join('/tmp/exploration_viz.png'),
               nrow=1)
