#!/usr/bin/env python
"""
@package docstring
Test node for info gain service client
"""

import utils.system_utils as system_utils
from online_map_completion_msgs.srv import InfoGains, InfoGainsRequest

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import RegionOfInterest
# import tf
# import tf.transformations

import argparse
import cv2
import json

def call_info_gain(costmap_file, info_file):
    rospy.wait_for_service('info_gain')

    info = None
    with open(info_file, 'r') as f:
        info = json.load(f)

    try:
        info_gain_service = rospy.ServiceProxy('info_gain', InfoGains)
        request = InfoGainsRequest()

        request.multi_channel_occupancy_grid = get_ros_image(costmap_file)
        request.frontier_rois = get_ros_bounding_boxes(info['BoundingBoxes'])
        request.frontier_clusters = get_ros_frontier_clusters(info['Frontiers'])

        response = info_gain_service(request)
        return response.info_gains

    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

def get_ros_image(costmap_file):
    costmap_image = cv2.imread(costmap_file, cv2.IMREAD_COLOR)
    msg = Image()
    msg.header.stamp = rospy.Time.now()
    msg.data = costmap_image.tostring()
    msg.width = costmap_image.shape[1]
    msg.height = costmap_image.shape[0]
    return msg


def get_ros_frontier_clusters(frontier_clusters):
    def convert_to_ros_cluster(frontier_cluster):
        def convert_to_ros_pose(pose):
            ros_pose = Pose()
            ros_pose.position.x = pose['x']
            ros_pose.position.y = pose['y']
            # ros_pose.position.orientation = tf.transformations.quaternion_from_euler(0, 0, pose['yaw'])
            return ros_pose

        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.poses = list(
            map(lambda x: convert_to_ros_pose(x),
                frontier_cluster)
        )

        return pose_array

    return list(
        map(lambda x: convert_to_ros_cluster(x),
            frontier_clusters)
    )


def get_ros_bounding_boxes(bounding_boxes):
    def convert_to_ros_bounding_box(bounding_box):
        ros_bb = RegionOfInterest()
        ros_bb.x_offset = bounding_box['x']
        ros_bb.y_offset = bounding_box['y']
        ros_bb.width = bounding_box['width']
        ros_bb.height = bounding_box['height']
        return ros_bb

    return list(
        map(
            lambda x: convert_to_ros_bounding_box(x),
            bounding_boxes
        )
    )

if __name__ == '__main__':
    rospy.init_node('info_gain_client', anonymous=True)
    parser = argparse.ArgumentParser(description='VAE KTH partial map info gain ROS service test client')

    parser.add_argument('costmap', type=str, help="costmap image location")
    parser.add_argument('info_json', type=str, help="info json file location")

    args = parser.parse_args()

    while True:
        print('info gain: ', call_info_gain(args.costmap, args.info_json))
        rospy.sleep(5)


