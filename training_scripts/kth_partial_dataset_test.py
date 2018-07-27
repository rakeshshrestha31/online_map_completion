#!/usr/bin/env python3
"""
@package docstring
script for calculating test accuracy, precision and recall
"""

# custom libraries
from models.residual_fully_conv_vae import ResidualFullyConvVAE
from utils import nn_module_utils
import utils.constants
from utils.tensorboard_logger import Logger
import utils.vis_utils
from utils.vis_utils import save_image
from utils.model_visualize import make_dot, make_dot_from_trace
from utils.generator_utils import collate_without_batching_dict

from utils import loss_functions as custom_loss_functions
from data_generators.kth_partial_map_dataloader import PartialMapDataset
# from data_generators.kth_partial_map_dataloader_frontiers import PartialMapDataset

from utils.exploration_utils import compute_expected_information_gain
# pytorch imports
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms

# standard library imports
import os
import numpy as np
import argparse
import functools
import json

def compute_positives(ground_truth, prediction):
    """

    :param ground_truth: byte tensor, 1-positive, 0-negative
    :param prediction: byte tensor, 1-positive, 0-negative
    :return:  tensor with elements 1 if true positive prediction
    """
    return torch.eq(ground_truth, prediction) * ground_truth

def compute_model_stats(input, reconstructed_occupancy_grid, ground_truth_occupancy_grid):
    """
    number of true positives, false positives, true negatives, false negatives and total
    :param input: input to the network B x 4 x H x W (4 channels: unknown, free, obstacle, prediction mask)
    :param reconstructed_occupancy_grid: B x 1 x H x W (0 - free, 1 - occupied)
    :param ground_truth_occupancy_grid: B x 1 x H x W (0 - free, 1 - occupied)
    :return: dict of number of true positives, false positives, true negatives, false negatives + misc secondary stats
    """

    frontier_mask = input[:, -1:, :, :].byte()

    gt_positives = torch.mul(ground_truth_occupancy_grid.gt(utils.constants.OBSTACLE_THRESHOLD), frontier_mask)
    gt_negatives = torch.mul(ground_truth_occupancy_grid.lt(utils.constants.FREE_THRESHOLD), frontier_mask)

    # non-positives = negatives + uncertains, non-negatives = positives + uncertains
    gt_non_positives = 1 - gt_positives
    gt_non_negatives = 1 - gt_negatives

    reconstructed_positives = reconstructed_occupancy_grid.gt(utils.constants.OBSTACLE_THRESHOLD) * frontier_mask
    reconstructed_negatives = reconstructed_occupancy_grid.lt(utils.constants.FREE_THRESHOLD) * frontier_mask

    true_positives = compute_positives(gt_positives, reconstructed_positives)
    true_negatives = compute_positives(gt_negatives, reconstructed_negatives)

    false_positives = compute_positives(gt_non_positives, reconstructed_positives)
    false_negatives = compute_positives(gt_non_negatives, reconstructed_negatives)

    viz_data = [
        ground_truth_occupancy_grid,
        frontier_mask.float(),
        # gt_positives.float(),
        # gt_negatives.float(),

        reconstructed_occupancy_grid,
        # reconstructed_positives.float(),
        # reconstructed_negatives.float(),
        # true_positives.float(),
        # true_negatives.float(),
        # false_positives.float(),
        # false_negatives.float()
    ]
    viz_data = torch.cat(viz_data, 0)
    save_image(viz_data.data.cpu(),
               os.path.join('/tmp/stat_viz.png'),
               nrow=1)

    stats = {
        'true_positives': torch.sum(true_positives).item(),
        'false_positives': torch.sum(false_positives).item(),
        'true_negatives': torch.sum(true_negatives).item(),
        'false_negatives': torch.sum(false_negatives).item(),
        'total_positives': torch.sum(gt_positives).item(),
        'total_negatives': torch.sum(gt_negatives).item(),
    }

    # secondary stats for easy precision-recall calculations
    stats['total_predicted_positives'] = stats['true_positives'] + stats['false_positives']
    stats['total_predicted_negatives'] = stats['true_negatives'] + stats['false_negatives']
    stats['total_certain_positives'] = stats['true_positives'] + stats['false_negatives']
    stats['total_certain_negatives'] = stats['true_negatives'] + stats['false_positives']
    stats['total'] = stats['total_positives'] + stats['total_negatives']
    stats['total_unknowns'] = stats['total'] - stats['total_certain_positives'] - stats['total_certain_negatives']
    stats['total_knowns'] = stats['total'] - stats['total_unknowns']
    stats['total_unknown_positives'] = stats['total_positives'] - stats['total_certain_positives']
    stats['total_unknown_negatives'] = stats['total_negatives'] - stats['total_certain_negatives']

    return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE KTH partial map training')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--latent-size', type=int, default=20, metavar='N',
                        help='latent vector size')

    parser.add_argument('test_dataset', type=str, metavar='S',
                        help='test dataset directory')
    parser.add_argument('original_dataset', type=str, metavar='S',
                        help='original dataset directory (with XML annotations)')
    parser.add_argument('checkpoint', type=str, default="checkpoints/model_best_epoch_accuracy.pth.tar",
                        help='Default=checkpoints/model_best.pth.tar')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print('loading dataset')
    test_dataset = PartialMapDataset(args.test_dataset, args.original_dataset)
    print('test set:', len(test_dataset))

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_without_batching_dict, **kwargs
    )

    model = ResidualFullyConvVAE((utils.constants.TARGET_HEIGHT, utils.constants.TARGET_WIDTH),
                                 latent_encoding_channels=args.latent_size, skip_connection_type='concat')  # 'add')  #

    if args.cuda:
        model = model.cuda()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    checkpoint_absolute_path = os.path.abspath(args.checkpoint)
    checkpoint_dir = '/'.join(checkpoint_absolute_path.split('/')[0:-1])
    print('loading checkpoints from {}'.format(checkpoint_dir))

    if os.path.isfile(args.checkpoint):

        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'"
              .format(args.checkpoint))
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
        exit(0)

    model.train(False)
    batch_stats = []
    batch_kld_losses = []
    for batch_idx, (input, ground_truth, info) in enumerate(test_loader):
        if batch_idx % 10 == 0:
            print('Batch: {} [{}/{} ({:.0f}%)]\t'.format(
                batch_idx, batch_idx * input.size(0), len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

        input = Variable(input)
        ground_truth = Variable(ground_truth)
        if args.cuda:
            input = input.cuda()
            ground_truth = ground_truth.cuda()

        recon_batch, mu, logvariance = model(input)  # original_data) #
        batch_stats.append(compute_model_stats(input, recon_batch, ground_truth))
        batch_kld_losses.append(custom_loss_functions.kl_divergence_loss(mu, logvariance).item() / args.batch_size)

        # compute_expected_information_gain(input, recon_batch, info)
        compute_expected_information_gain(input, ground_truth, info)

        # print('batch stat', json.dumps(batch_stats[-1], indent=4), 'kld loss', batch_kld_losses[-1])
        print(json.dumps([i['Frontiers'] for i in info], indent=4))
        exit(0)


    overall_stats = functools.reduce(
        lambda sum, current: {i: sum[i] + current[i] for i in current},
        batch_stats,
        {i: 0 for i in batch_stats[0]}
    )

    average_kld_loss = np.mean(batch_kld_losses)

    print('average_kld_loss', average_kld_loss)
    print('overall_stats', json.dumps(overall_stats, indent=4))

    def divide(x: int, y: int):
        return x / y if y else float('NaN')

    print(json.dumps(
        {
            'accuracy':
                divide(overall_stats['true_positives'] + overall_stats['true_negatives'],
                       overall_stats['total']),

            'generous_accuracy':
                divide(overall_stats['true_positives'] + overall_stats['true_negatives'],
                       overall_stats['total_knowns']),

            'obstacle_precision':
                divide(overall_stats['true_positives'], overall_stats['total_predicted_positives']),

            'obstacle_recall':
                divide(overall_stats['true_positives'], overall_stats['total_positives']),

            'obstacle_generous_recall':
                divide(overall_stats['true_positives'], overall_stats['total_certain_positives']),

            'free_precision':
                divide(overall_stats['true_negatives'], overall_stats['total_predicted_negatives']),

            'free_recall':
                divide(overall_stats['true_negatives'], (overall_stats['total_negatives'])),

            'free_generous_recall':
                divide(overall_stats['true_negatives'], (overall_stats['total_certain_negatives'])),
        },
        indent=4
    ))
