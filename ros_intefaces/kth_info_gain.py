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

# custom libraries
from models.residual_fully_conv_vae import ResidualFullyConvVAE
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

        rospy.spin()

    def init_model(self):
        parser = argparse.ArgumentParser(description='VAE KTH partial map info gain ROS service server')

        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='enables CUDA training')
        parser.add_argument('--latent-size', type=int, default=20, metavar='N',
                            help='latent vector size')

        parser.add_argument('checkpoint', type=str, default="checkpoints/model_best_epoch_accuracy.pth.tar",
                            help='Default=checkpoints/model_best.pth.tar')

        self.args = parser.parse_args()
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()

        self.model = ResidualFullyConvVAE((utils.constants.TARGET_HEIGHT, utils.constants.TARGET_WIDTH),
                                     latent_encoding_channels=self.args.latent_size, skip_connection_type='concat')

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
        print('image size', np_arr.shape)

        cv2.imwrite('/tmp/costmap_srv.png', np_arr)

        frontiers_data = OneDataInfoBase(
            np_arr,
            self._convert_frontier_info({
                'BoundingBoxes': input.frontier_rois,
                'Frontiers': input.frontier_clusters
            })
        )

        response.info_gains = []
        for i in range(len(frontiers_data)):
            image, info = frontiers_data[i]
            image = torch.from_numpy(image).unsqueeze(0)
            image = Variable(image)

            if self.args.cuda:
                image = image.cuda()
            output, _, _ = self.model(image)

            info_gain = compute_expected_information_gain(image, output, [info], 'info_gain_srv{}'.format(i))
            msg = UInt64(info_gain[0]['information_gain'])
            response.info_gains.append(msg)

        # TODO: info gain for all frontiers

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

