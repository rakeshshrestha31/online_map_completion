#!/usr/bin/env python3
"""
@package docstring
utilities to help exploration
"""

# custom imports
import utils.vis_utils
from utils.vis_utils import save_image
import utils.constants

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
    unknown_mask = input[:, 0:1, :, :].byte()

    cells_to_predict_mask = torch.mul(frontier_bb_mask, unknown_mask)

    frontier_points_mask = input.clone()
    frontier_points_mask[:, 0:-1, :, :] = 0

    predicted_obstacles = prediction.gt(utils.constants.OBSTACLE_THRESHOLD)
    predicted_obstacles = torch.mul(predicted_obstacles, cells_to_predict_mask)

    annotated_input = input.clone()

    for batch_idx in range(input.size(0)):
        for frontier_group in info[0]['Frontiers']:
            for frontier_point in frontier_group:
                annotated_input[batch_idx, :, frontier_point[1], frontier_point[0]] = 0.0
                frontier_points_mask[batch_idx, 2, frontier_point[1], frontier_point[0]] = 1.0

    annotated_input[:, 0, :, :] = torch.mul(annotated_input[:, 0, :, :], (1  - cells_to_predict_mask).float())
    annotated_input[:, 2:3, :, :] += predicted_obstacles.float()

    viz_data = [
        utils.vis_utils.get_transparancy_adjusted_input(
            utils.vis_utils.get_padded_occupancy_grid(prediction)
        ),
        utils.vis_utils.get_transparancy_adjusted_input(input),
        utils.vis_utils.get_transparancy_adjusted_input(annotated_input)
    ]

    viz_data = torch.cat(viz_data, 0)
    save_image(viz_data.data.cpu(),
               os.path.join('/tmp/exploration_viz.png'),
               nrow=1, padding=2)
