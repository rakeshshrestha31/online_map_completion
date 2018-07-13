#!/usr/bin/env python3

"""
@package docstring
script for training KTH partial map dataset
"""

# custom libraries
from models.residual_fully_conv_vae import ResidualFullyConvVAE
from utils import nn_module_utils
import utils.constants
# from utils.tensorboard_logger import Logger
import utils.vis_utils
from utils.vis_utils import save_image
from utils.model_visualize import make_dot, make_dot_from_trace


from utils import loss_functions as custom_loss_functions
from data_generators.kth_partial_map_dataloader import PartialMapDataset

# pytorch imports
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms


# standard library imports
import argparse
import os
import numpy as np


def loss_function(input, reconstructed_occupancy_grid, ground_truth_occupancy_grid,
                  mu, logvar):
    """
    loss function
    :param input: input to the network B x 4 x H x W (4 channels: unknown, free, obstacle, prediction mask)
    :param reconstructed_occupancy_grid: B x 1 x H x W (0 - free, 1 - occupied)
    :param ground_truth_occupancy_grid: B x 1 x H x W (0 - free, 1 - occupied)
    :param mu: mean of the latent vector
    :param logvar: log variance of the latent vector
    :return: total loss, binary cross entropy loss and KL-divergence loss
    """
    OBSTACLE_WEIGHT = 130

    cost_mask = input[:, 1:2, :, :].clone()

    # TODO: different cost for obstacle and free space
    ## because the amount of free space is more, the network might be tempted to always predict free space
    # # cost of the free space (0) is mapped to 1 and that of obstacle is mapped to (OBSTACLE_WEIGHT+1)
    # cost_mask[:, :, :, :] = cost_mask[:, :, :, :] * OBSTACLE_WEIGHT + 1.0
    #
    # # no points for guessing what we already know
    # cost_mask[:, :, partial_map_start[0]:partial_map_end[0], partial_map_start[1]:partial_map_end[1]] = 0

    if args.cuda:
        cost_mask = cost_mask.cuda()

    BCE = F.binary_cross_entropy(
        reconstructed_occupancy_grid, ground_truth_occupancy_grid,
        weight=cost_mask, size_average=False
    )
    KLD = custom_loss_functions.kl_divergence_loss(mu, logvar)
    return BCE + KLD, BCE, KLD


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Rent 3D training')
    # generic training arguments
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='N',
                        help='learning rate')
    parser.add_argument('--regularizer-weight', type=float, default=5e-3, metavar='N',
                        help='regularizer weight')
    parser.add_argument('--checkpoint', type=str, default="checkpoints/model_best_epoch_accuracy.pth.tar",
                        help='Default=checkpoints/model_best.pth.tar')
    parser.add_argument('--temp-checkpoint', type=str, default="checkpoint.pth.tar",
                        help='Default=checkpoints/model_best.pth.tar')

    parser.add_argument('--load-weight', dest='is_load_weight', action='store_true', help='Load weight')
    parser.add_argument('--no-load-weight', dest='is_load_weight', action='store_false', help='Don\'t Load weight')
    parser.set_defaults(is_load_weight=False)

    # application specific arguments
    parser.add_argument('--latent-size', type=int, default=20, metavar='N',
                        help='latent vector size')
    parser.add_argument('--sampling-distance', type=float, default=25, metavar='N',
                        help='sampling distance for generating dataset')
    parser.add_argument('--map-resolution', type=float, default=0.2, metavar='N',
                        help='resolution of the map (m/pixel)')
    parser.add_argument('--augmented-rotations', type=int, default=10, metavar='N',
                        help='number of rotations to augment (min=1, no augmentation)')
    parser.add_argument('train_dataset', type=str, metavar='S',
                        help='train dataset directory')
    parser.add_argument('validation_dataset', type=str, metavar='S',
                        help='validation dataset directory')
    parser.add_argument('test_dataset', type=str, metavar='S',
                        help='test dataset directory')
    parser.add_argument('original_dataset', type=str, metavar='S',
                        help='original dataset directory (with XML annotations)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_dataset       = PartialMapDataset(args.train_dataset, args.original_dataset)
    test_dataset        = PartialMapDataset(args.test_dataset, args.original_dataset)
    validation_dataset  = PartialMapDataset(args.validation_dataset, args.original_dataset)

    kwargs = {'num_workers': 3, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs
    )

    model = ResidualFullyConvVAE((utils.constants.HEIGHT, utils.constants.WIDTH), latent_encoding_channels=args.latent_size, skip_connection_type='concat') # 'add')  #

    if args.cuda:
        model = model.cuda()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.regularizer_weight)

    # checkpoint_dir = 'checkpoints'
    checkpoint_absolute_path = os.path.abspath(args.checkpoint)
    checkpoint_dir = '/'.join(checkpoint_absolute_path.split('/')[0:-1])
    print('saving checkpoints at {}'.format(checkpoint_dir))

    if args.checkpoint and args.is_load_weight:
        if os.path.isfile(args.checkpoint):

            print("=> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(args.checkpoint))
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpoint))

    checkpoint_saver = nn_module_utils.CheckPointSaver([
        'epoch_loss', 'epoch_accuracy', 'batch_loss', 'batch_accuracy', 'test_loss'
    ], checkpoint_dir)

    # logger = Logger('./logs')
    os.system("mkdir -p ./results")

    def train(epoch):
        model.train()
        train_loss = 0
        train_reconstruction_loss = 0
        train_kld_loss = 0
        step = (epoch - 1) * len(train_loader.dataset) + 1
        for batch_idx, (input, ground_truth) in enumerate(train_loader):
            input = Variable(input)
            ground_truth = Variable(ground_truth)
            if args.cuda:
                input = input.cuda()
                ground_truth = ground_truth.cuda()

            optimizer.zero_grad()
            recon_batch, mu, logvariance = model(input)  # original_data) #
            loss, reconstruction_loss, kld_loss = loss_function(input, recon_batch, ground_truth, mu, logvariance)
            loss.backward()

            train_loss += loss.item()
            train_reconstruction_loss += reconstruction_loss.item()
            train_kld_loss += kld_loss.item()

            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * input.size(0), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item() / input.size(0)))

                checkpoint_saver.save_checkpoint({
                    'state_dict': model.state_dict(),

                    'batch_loss': loss.item(),
                    'batch_reconstruction_loss': reconstruction_loss.item(),
                    'batch_kld_loss': kld_loss.item(),

                    'optimizer': optimizer.state_dict(),
                }, args.temp_checkpoint)
                # logger.scalar_dict_summary({
                #     'batch_loss': float(loss.item())
                # }, step)
                step += batch_idx

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
        checkpoint_saver.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),

            'epoch_loss': train_loss,
            'epoch_reconstruction_loss': train_reconstruction_loss,
            'epoch_kld_loss': train_kld_loss,

            'optimizer': optimizer.state_dict(),
        })
        # logger.scalar_dict_summary({
        #     'epoch_loss': train_loss,
        #     'epoch_reconstruction_loss': train_reconstruction_loss,
        #     'epoch_kld_loss': train_kld_loss,
        # }, epoch)


    def test(epoch):
        model.eval()
        test_loss = 0
        test_reconstruction_loss = 0
        test_kld_loss = 0

        # test the lot
        # for i, (masked_data, original_data) in enumerate(test_loader):

        # only one batch test
        if True:
            i = 0
            data_iter = iter(test_loader)
            input, ground_truth = next(data_iter)

            if args.cuda:
                input = input.cuda()
                ground_truth = ground_truth.cuda()
            input = Variable(input, volatile=True)
            ground_truth = Variable(ground_truth, volatile=True)

            recon_batch, mu, logvariance = model(input)  # ground_truth) #
            loss, reconstruction_loss, kld_loss = loss_function(input, recon_batch, ground_truth, mu, logvariance)
            test_loss += loss.item()
            test_reconstruction_loss += reconstruction_loss.item()
            test_kld_loss += kld_loss.item()

            if epoch == 1:
                dot = make_dot(recon_batch, params=dict(list(model.named_parameters())))
                dot.render("model", '.')

            if i == 0:
                n = min(input.size(0), 32)
                # threshold the reconstructed data
                # recon_batch = (recon_batch >= 0.5).type_as(recon_batch)

                # TODO: sample a few from the latent space for viz
                # recon_batches = []  # [recon_batch]
                # for i in range(5):
                #     recon_batch2, _, _ = model(input)  # ground_truth) #
                #
                #     # TODO: visualize only frontier regions
                #     # recon_batch2[:, :, partial_map_start[0]:partial_map_end[0],
                #     # partial_map_start[1]:partial_map_end[1]] = \
                #     #     ground_truth[:, :, partial_map_start[0]:partial_map_end[0],
                #     #     partial_map_start[1]:partial_map_end[1]]
                #
                #     recon_batches.append(recon_batch2)
                recon, _, _ = model(input)

                input_for_viz = utils.vis_utils.get_transparancy_adjusted_input(input[:n])

                ground_truth_for_viz = utils.vis_utils.get_padded_occupancy_grid(ground_truth[:n])
                # ground_truth_for_viz[:, -1, :, :] = input_for_viz[:, -1, :, :]

                recon_for_viz = utils.vis_utils.get_padded_occupancy_grid(recon[:n])
                # recon_for_viz[:, -1, :, :] = input_for_viz[:, -1, :, :]

                comparison = torch.cat([
                    input_for_viz, ground_truth_for_viz, recon_for_viz
                ])

                save_image(comparison.data.cpu(),
                           os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                        'results/reconstruction_' + str(epoch) + '.png'),
                           nrow=n)

        # test_loss /= len(test_loader.dataset)
        test_loss /= args.batch_size

        print('====> Test set loss: {:.4f}'.format(test_loss))
        # logger.scalar_dict_summary({
        #     'test_loss': test_loss,
        #     'test_reconstruction_loss': test_reconstruction_loss,
        #     'test_kld_loss': test_kld_loss,
        # }, epoch)
        checkpoint_saver.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),

            'test_loss': test_loss,
            'test_reconstruction_loss': test_reconstruction_loss,
            'test_kld_loss': test_kld_loss,

            'optimizer': optimizer.state_dict(),
        })

    for epoch in range(1, args.epochs + 1):
        test(epoch)
        train(epoch)