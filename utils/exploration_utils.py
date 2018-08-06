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
import cv2


def compute_expected_information_gain(input: Variable, prediction: Variable,
                                      info: typing.List[dict], filename: str) \
    -> typing.List[dict]:
    """
    Computing information gain for each frontier group (uncertain predictions are considered to be free)
    :param input: input to the network B x 4 x H x W (4 channels: unknown, free, obstacle, prediction mask)
    :param prediction: predicted occupancy map B x 1 x H x W (0 - free, 1 - occupied)
    :param info: list of dictionary containing groups of frontier points in "Frontiers" key
    :return: list of dict containing information gain and other stuffs for each frontier cluster
    """
    frontier_bb_mask = input[:, -1:, :, :].byte()
    unknown_mask = input[:, 0:1, :, :].byte()

    cells_to_predict_mask = torch.mul(frontier_bb_mask, unknown_mask)

    predicted_obstacles = prediction.gt(utils.constants.OBSTACLE_THRESHOLD)
    predicted_obstacles = torch.mul(predicted_obstacles, cells_to_predict_mask)

    # uncertain predictions are considered to be free
    flood_fillable_tensor = cells_to_predict_mask.clone()
    flood_fillable_tensor = torch.mul(flood_fillable_tensor, (1-predicted_obstacles))

    annotated_input = input.clone()

    for batch_idx in range(input.size(0)):
        for frontier_group in info[batch_idx]['Frontiers']:
            for frontier_point in frontier_group:
                flood_fillable_tensor[batch_idx, 0, frontier_point[1], frontier_point[0]] = 1
                annotated_input[batch_idx, :, frontier_point[1], frontier_point[0]] = 0.0

    annotated_input[:, 0:1, :, :] = torch.mul(annotated_input[:, 0:1, :, :], (1 - cells_to_predict_mask).float())
    annotated_input[:, 2:3, :, :] += predicted_obstacles.float()

    multi_channel_prediction = utils.vis_utils.get_padded_occupancy_grid(prediction)
    multi_channel_prediction[:, 3, :, :] = input[:, 3, :, :]

    info = flood_fill_frontiers(flood_fillable_tensor, info)

    flood_filled = flood_fillable_tensor.clone()

    for batch_idx in range(input.size(0)):
        flood_filled[batch_idx, :, :, :] = info[batch_idx]['flood_filled_mask']

    viz_data = [

        utils.vis_utils.get_transparancy_adjusted_input(input),

        utils.vis_utils.get_transparancy_adjusted_input(
            multi_channel_prediction,
        ),

        # utils.vis_utils.get_transparancy_adjusted_input(annotated_input),

        utils.vis_utils.get_transparancy_adjusted_input(torch.cat(
            [
                torch.zeros(flood_filled.size(0), 2, *(flood_filled.shape[2:])),
                flood_filled.float(),
                input[:, 3:, :, :]
            ],
            dim=1
        )),

        utils.vis_utils.get_transparancy_adjusted_input(torch.cat(
            [
                torch.zeros(flood_fillable_tensor.size(0), 2, *(flood_fillable_tensor.shape[2:])),
                flood_fillable_tensor.float(),
                input[:, 3:, :, :]
            ],
            dim=1
        )),
    ]

    viz_data = torch.cat(viz_data, 0)
    save_image(viz_data.data.cpu(),
               # os.path.join('/tmp/', filename+'.png'),
               filename,
               nrow=input.size(0), padding=2)

    return info


def flood_fill_frontiers(flood_fillable_tensor,
                         info: typing.List[dict]) -> typing.List[dict]:

    """
    :param flood_fillable_tensor: tensor with image that's directly flood fillable
    (i.e. unknown regions predicted to be unobstructed)
    :param info: list of dictionary containing groups of frontier points in "Frontiers" key
    :return:
    """
    output = []
    for batch_idx in range(flood_fillable_tensor.size(0)):
        np_array = flood_fillable_tensor[batch_idx, 0, :, :].cpu().numpy()

        for frontier_group in info[batch_idx]['Frontiers']:
            # the borders should be padded for flood fill
            flood_filled_mask = np.zeros((np_array.shape[0] + 2, np_array.shape[1] + 2), dtype=np.uint8)

            # flood-filled pixels are cleared for each frontier point in order to avoid repeated flood filling
            tmp_np_array = np_array.copy()

            for frontier_point in frontier_group:
                tmp_flood_filled_mask = np.zeros((np_array.shape[0] + 2, np_array.shape[1] + 2), dtype=np.uint8)

                if tmp_np_array[frontier_point[1], frontier_point[0]] != 1:
                    # the frontier point is already flood filled. No need to check again
                    continue

                cv2.floodFill(
                    tmp_np_array,
                    tmp_flood_filled_mask,
                    (frontier_point[0], frontier_point[1]),
                    1
                )
                flood_filled_mask += tmp_flood_filled_mask

                # flood-filled pixels are cleared for each frontier point in order to avoid repeated flood filling
                tmp_np_array = np.multiply(tmp_np_array, 1 - flood_filled_mask[1:-1, 1:-1])

            # the borders are padded for flood fill, remove them
            flood_filled_mask = flood_filled_mask[1:-1, 1:-1]

            flood_filled_mask = flood_filled_mask.clip(0, 1)
            output.append({
                'flood_filled_mask': torch.from_numpy(flood_filled_mask),
                'information_gain': np.sum(flood_filled_mask)
            })
    return output