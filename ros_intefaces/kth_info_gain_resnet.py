#!/usr/bin/env python3
"""
@package docstring
ROS service interface to getting expected information gain from map prediction

you should update python3 alternative to python3.5

Python3 ros dependencies:
$ sudo apt-get install python3-yaml
$ sudo pip3 install rospkg catkin_pkg
"""

# ros
import rospy
# import tf
from online_map_completion_msgs.srv import InfoGains, InfoGainsRequest, InfoGainsResponse
from std_msgs.msg import UInt64
from sensor_msgs.msg import Image

import cv_bridge
from cv_bridge import CvBridge, CvBridgeError

# custom libraries
from models.resnet_vae import ResnetVAE
import utils.constants
from utils.exploration_utils import compute_expected_information_gain

# pytorch imports
import torch
from torch.autograd import Variable

from data_generators.OneDataInfo import OneDataInfoBase

# standard library imports
import os
import numpy as np
import argparse
import cv2
import functools

class InfoGainServer:
    def __init__(self):
        service = rospy.Service('info_gain', InfoGains, self.service_handler)
        self.args = None
        self.model = None
        self.init_model()

        self.bridge = CvBridge()

        rospy.spin()

    def init_model(self):
        parser = argparse.ArgumentParser(description='VAE KTH partial map info gain ROS service server')

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
        parser.add_argument('--temp-checkpoint', type=str, default="checkpoint.pth.tar",
                            help='Default=checkpoints/model_best.pth.tar')
        parser.add_argument('--log-dir', type=str, default="./logs")
        parser.add_argument('--results-dir', type=str, default="./results")

        parser.add_argument('--load-weight', dest='is_load_weight', action='store_true', help='Load weight')
        parser.add_argument('--no-load-weight', dest='is_load_weight', action='store_false', help='Don\'t Load weight')
        parser.set_defaults(is_load_weight=False)

        # application specific arguments
        parser.add_argument('--latent-size', type=int, default=512, metavar='N',
                            help='latent vector size')
        parser.add_argument('--sampling-distance', type=float, default=25, metavar='N',
                            help='sampling distance for generating dataset')
        parser.add_argument('--map-resolution', type=float, default=0.2, metavar='N',
                            help='resolution of the map (m/pixel)')
        parser.add_argument('--augmented-rotations', type=int, default=10, metavar='N',
                            help='number of rotations to augment (min=1, no augmentation)')
        parser.add_argument("--resnet-version", type=int, default=18, metavar='N',
                            help='resnet version {18, 34, 50, 101, 152}')
        parser.add_argument('--pretraining', action='store_true', default=False,
                            help='Pretraining with input as ground truth')

        parser.add_argument('checkpoint', type=str, default="checkpoints/model_best_epoch_accuracy.pth.tar",
                            help='Default=checkpoints/model_best.pth.tar')

        self.args = parser.parse_args()
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()

        self.model = ResnetVAE(
            {
                'block_lengths': [1, 1, 1, 1],
                'block': 'BasicBlock'
            },
            latent_encoding_channels=self.args.latent_size, skip_connection_type='concat'
        )

        # self.model = ResidualFullyConvVAE((utils.constants.TARGET_HEIGHT, utils.constants.TARGET_WIDTH),
        #                              latent_encoding_channels=self.args.latent_size, skip_connection_type='concat')

        if self.args.cuda:
            self.model = self.model.cuda()
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        if os.path.isfile(self.args.checkpoint):

            print("=> loading checkpoint '{}'".format(self.args.checkpoint))
            checkpoint = torch.load(self.args.checkpoint)
            self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(self.args.checkpoint))
        else:
            print("=> no checkpoint found at '{}'".format(self.args.checkpoint))
            exit(0)

    def service_handler(self, input):
        print('service call received')
        response = InfoGainsResponse()

        ros_image = input.multi_channel_occupancy_grid
        np_arr = np.fromstring(ros_image.data, np.uint8).reshape(ros_image.height, ros_image.width, 3)
        gt_image = input.ground_truth
        np_arr_gt = np.fromstring(gt_image.data, np.uint8).reshape(gt_image.height, gt_image.width)
        print('image size', np_arr.shape)

        cv2.imwrite('/tmp/costmap_srv.png', np_arr)

        frontiers_data = OneDataInfoBase(
            np_arr,
            np_arr_gt,
            self._convert_frontier_info({
                'BoundingBoxes': input.frontier_rois,
                'Frontiers': input.frontier_clusters
            })
        )

        response.info_gains = []
        response.info_gains_gt = []
        batch_input = []
        batch_gt = []
        batch_info = []

        for i in range(len(frontiers_data)):
            image, gt, info = frontiers_data[i]
            image = torch.from_numpy(image)
            gt = torch.from_numpy(gt)
            batch_input.append(image)
            batch_gt.append(gt)
            batch_info.append(info)

        batch_input = torch.stack(batch_input)
        batch_input = Variable(batch_input)
        batch_gt = torch.stack(batch_gt)
        batch_gt = Variable(batch_gt)

        if self.args.cuda:
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()
        output, _, _ = self.model(batch_input)

        info_gain, info_gain_image = compute_expected_information_gain(batch_input.clone(), output, batch_info, '/tmp/info_gain_srv.png')
        info_gain_gt, info_gain_image_gt = compute_expected_information_gain(batch_input.clone(), batch_gt, batch_info, '/tmp/info_gain_srv_gt.png')

        for i in range(len(info_gain)):
            msg = UInt64(info_gain[i]['information_gain'])
            response.info_gains.append(msg)

        for i in range(len(info_gain_gt)):
            msg = UInt64(info_gain_gt[i]["information_gain"])
            response.info_gains_gt.append(msg)

        # info_gain_image = cv2.imread('/tmp/info_gain_srv.png', cv2.IMREAD_UNCHANGED)
        # info_gain_image_gt = cv2.imread('/tmp/info_gain_srv_gt.png', cv2.IMREAD_UNCHANGED)

        response.prediction = self.bridge.cv2_to_imgmsg(info_gain_image, "passthrough")
        response.prediction_gt = self.bridge.cv2_to_imgmsg(info_gain_image_gt, "passthrough")

        return response

    def _convert_frontier_info(self, frontier_info: dict):
        """
        Converts frontiers info from ROS types to simple arrays and dics
        :param frontier_info: dict of info in ros message type
        :return:
        """
        output = {}

        def convert_bounding_box(ros_bounding_box):
            return {
                'x': ros_bounding_box.x_offset,
                'y': ros_bounding_box.y_offset,
                'width': ros_bounding_box.width,
                'height': ros_bounding_box.height
            }

        def convert_frontier_clusters(ros_frontier_clusters):
            return list(
                map(
                    lambda pose: {
                        'x': pose.position.x,
                        'y': pose.position.y,
                        'yaw': 0
                    }, #tf.getYaw(pose.orientation)),
                    ros_frontier_clusters.poses
                )
            )

        output['BoundingBoxes'] = list(map(convert_bounding_box, frontier_info['BoundingBoxes']))
        output['Frontiers'] = list(map(convert_frontier_clusters, frontier_info['Frontiers']))

        return output


if __name__ == '__main__':
    rospy.init_node('info_gain_server', anonymous=True)
    server = InfoGainServer()

