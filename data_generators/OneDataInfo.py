
import re
import os
import json
import glob
import cv2
import numpy as np

import utils.constants as const
from kth.FloorPlanGraph import FloorPlanGraph


def clip_image(image, rect):
    return image[rect[0]: rect[0] + rect[2], rect[1]:rect[1] + rect[3]]


def clip_frontiers(frontiers, rect):
    frontiers_clipped = []
    for i in range(len(frontiers)):
        frontier_new = [frontiers[i][0] - rect[0],
                        frontiers[i][1] - rect[1],
                        frontiers[i][2]]
        if 0 <= frontier_new[0] < rect[2] and 0 <= frontier_new[1] < rect[3]:
            frontiers_clipped.append(frontier_new)
    return frontiers_clipped


def get_mask_image(rect, mask_size, image_size):
    """
    Generate mask image with size of image_size, and mask centered at rect center with size of mask_size
    :param rect: a rectangle used to indicate mask center
    :param mask_size: the mask area size
    :param image_size: image size
    :return: mask image
    """
    mask_image = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
    center = [int(rect[i] + rect[i+2] / 2.0) for i in range(2)]
    mask_half_size = [int(mask_size[i] / 2) for i in range(2)]
    start = [center[i] - mask_half_size[i] for i in range(2)]
    end = [start[i] + mask_size[i] for i in range(2)]
    start = [start[i] if start[i] >= 0 else 0 for i in range(2)]
    end = [end[i] if end[i] <= image_size[i] else image_size[i] for i in range(2)]
    mask_image[start[0]: end[0], start[1]: end[1]].fill(255)
    return mask_image


def get_mask(image_size, mask_size):
    """generate mask in the center of the image, mask size is defined
    in constants.predicted_width and constants.predicted_height

    :arg image_size the generated mask image size
    :return images
    """
    mask_size = [mask_size[i] if mask_size[i] < image_size[i] else image_size[i] for i in range(2)]
    center = [int(image_size[i] / 2) for i in range(2)]
    mask_half_size = [int(mask_size[i] / 2) for i in range(2)]
    start = [center[i] - mask_half_size[i] for i in range(2)]

    mask = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
    mask[start[0]:start[0]+mask_size[0], start[1]:start[1]+mask_size[1]].fill(255)

    return mask


def resize_frontiers(frontiers, ratio):
    """
    Similar to image resize, resize the frontier points
    :param frontiers:
    :param ratio:
    :return:
    """
    frontiers_resized = []
    for i in range(len(frontiers)):
        frontier_new = [int(round(frontiers[i][0] * ratio)),
                        int(round(frontiers[i][1] * ratio)),
                        frontiers[i][2]]
        frontiers_resized.append(frontier_new)
    return frontiers_resized


def resize_rect(rect, ratio):
    """
    Similar to image resize, resize the rectangle
    :param rect:
    :param ratio:
    :return:
    """
    return [int(round(rect[i] * ratio)) for i in range(4)]


def enlarge_rect(rect, rect_new_size):
    """
    Change the rectangle size while keep changed center is the same with original one
    :param rect: input rectangle
    :param rect_new_size: new rectangle size
    :return:
    """
    center = get_center(rect)
    start = [center[i] - int(rect_new_size[i] / 2) for i in range(2)]
    new_rect = [start[0], start[1], rect_new_size[0], rect_new_size[1]]
    return new_rect


def get_center(rect):
    center = (rect[0] + int(np.round(rect[2] / 2.0)), rect[1] + int(np.round(rect[3] / 2.0)))
    return center


def pad_frontiers(frontiers, original_size, new_size):
    pad_offset = [int((new_size[i] - original_size[i]) / 2.0) for i in range(2)]
    frontiers_pad = []
    for i in range(len(frontiers)):
        frontier_new = [frontiers[i][0] + pad_offset[0],
                        frontiers[i][1] + pad_offset[1],
                        frontiers[i][2]]
        frontiers_pad.append(frontier_new)
    return frontiers_pad


def pad_rect(rect, original_size, new_size):
    """
    Similar to pad a image, pad the rectangle in a image to corresponding rectangle in new image
    :param rect: rectangle in original image
    :param original_size: original image size
    :param new_size: new padded image size
    :return: new padded rectangle
    """
    start = [int((new_size[i] - original_size[i]) / 2.0) for i in range(2)]
    rect_new = np.copy(rect)
    rect_new[0] = rect[0] + start[0]
    rect_new[1] = rect[1] + start[1]
    return rect_new


def pad_image(original_image, new_size):
    original_size = original_image.shape
    start = [int((new_size[i] - original_size[i])/2.0) for i in range(2)]

    output_image = np.zeros((new_size[0], new_size[1]), dtype=np.uint8)
    output_image[start[0]:(start[0] + original_size[0]), start[1]:(start[1] + original_size[1])] = original_image
    return output_image


def pad_costmap(original_costmap, new_size):
    original_size = original_costmap.shape
    start = [int((new_size[i] - original_size[i]) / 2.0) for i in range(2)]

    output_image = np.zeros((new_size[0], new_size[1], 3), dtype=np.uint8)
    output_image[:, :, 0].fill(255)
    output_image[start[0]:(start[0] + original_size[0]),start[1]:(start[1] + original_size[1])] = original_costmap
    return output_image


def parse_bounding_boxes_and_frontiers(bounding_boxes_list, frontier_cluster_list):
    bounding_boxes = []
    frontier_clusters = []
    assert len(bounding_boxes_list) == len(frontier_cluster_list)

    for i in range(len(bounding_boxes_list)):
        bounding_box = bounding_boxes_list[i]
        # if bounding_box['height'] > const.FRONTIER_BOUNDING_BOX_MIN and \
        #         bounding_box['width'] > const.FRONTIER_BOUNDING_BOX_MIN:

        rect = [bounding_box['y'],
                bounding_box['x'],
                bounding_box['height'],
                bounding_box['width']]

        cluster = []
        for j in range(len(frontier_cluster_list[i])):
            frontier_dict = frontier_cluster_list[i][j]
            frontier = [frontier_dict['y'],
                        frontier_dict['x'],
                        frontier_dict['yaw']]
            cluster.append(frontier)

        frontier_clusters.append(cluster)
        bounding_boxes.append(rect)

    return bounding_boxes, frontier_clusters

class OneDataInfoBase:
    def __init__(self, costmap: np.core.multiarray, info: dict):
        """

        :param costmap_image: input to the network H x W x 3 (3 channels: unknown, free, obstacle)
        :param info: frontiers info dict
        """
        self.costmap = costmap
        self.frontier_bounding_boxes, self.frontier_cluster_points = \
            parse_bounding_boxes_and_frontiers(info["BoundingBoxes"], info["Frontiers"])

    def __len__(self):
        return len(self.frontier_bounding_boxes)

    def __getitem__(self, item):

        rect = self.frontier_bounding_boxes[item]
        frontiers = self.frontier_cluster_points[item]

        ratio = const.ORIGINAL_RESOLUTION / const.TARGET_RESOLUTION
        costmap_resized = cv2.resize(self.costmap, (0, 0), fx=ratio, fy=ratio, interpolation=const.RESIZE_INTERPOLATION)
        rect_resized = resize_rect(rect, ratio)
        frontiers_resized = resize_frontiers(frontiers, ratio)

        image_size = costmap_resized.shape
        pad_image_size = (image_size[0] + const.TARGET_HEIGHT, image_size[1] + const.TARGET_WIDTH)

        # get the mask size
        if const.MASK_SIZE_FROM_FRONTIER:
            mask_size = (int(rect_resized[2] * const.FRONTIER_MASK_RESIZE_FACTOR),
                         int(rect_resized[3] * const.FRONTIER_MASK_RESIZE_FACTOR))
        else:
            mask_size = [const.PREDICTION_HEIGHT, const.PREDICTION_WIDTH]

        costmap_pad = pad_costmap(costmap_resized, pad_image_size)
        rect_pad = pad_rect(rect_resized, image_size, pad_image_size)
        frontiers_pad = pad_frontiers(frontiers_resized, image_size, pad_image_size)
        mask_pad = get_mask_image(rect_pad, mask_size, pad_image_size)
        final_image_size = (const.TARGET_HEIGHT, const.TARGET_WIDTH)

        # get the Area to be cropped as input
        crop_rect = enlarge_rect(rect_pad, final_image_size)
        # crop the final map
        costmap_final = clip_image(costmap_pad, crop_rect)
        mask_final = clip_image(mask_pad, crop_rect)
        frontiers_final = clip_frontiers(frontiers_pad, crop_rect)

        # normalize the images
        input_image = np.dstack((costmap_final, mask_final)).astype(dtype=np.float32) / 255.0

        # H * W * D -> D * H * W
        input_image = input_image.transpose(2, 0, 1)

        # return extra information as a dict of list of frontiers with only one element + order is (x, y, ...)
        #  to be consistent with dataset with multiple frontiers in a single image
        return input_image, \
               {
                   'Frontiers': [[(i[1], i[0], i[2]) for i in frontiers_final]],
                   'BoundingBoxes': [[crop_rect[1], crop_rect[0], crop_rect[3], crop_rect[2]]]
               }

class OneDataInfo:
    def __init__(self, json_path):
        self.frontier_bounding_boxes = []
        self.frontier_cluster_points = []
        self.json_path = ""
        self.load(json_path)

    def load(self, json_path):
        self.json_path = json_path
        with open(self.json_path, 'r') as f:
            info = json.load(f)
            self.frontier_bounding_boxes, self.frontier_cluster_points = \
                parse_bounding_boxes_and_frontiers(info["BoundingBoxes"], info["Frontiers"])

    def get_map_name(self):
        json_files_split = self.json_path.split('/')
        return json_files_split[-3]

    def get_costmap_path(self):
        regex = re.compile(r'.*info(\d+).json')
        result = re.search(regex, self.json_path)
        if result is None:
            return ""
        else:
            index = result.group(1)
            costmap_files = os.path.join(
                '/'.join(self.json_path.split('/')[0:-1]),
                'costmap' + str(index) + '.png'
            )
            return costmap_files

    def get_bounding_box_img_path(self):
        regex = re.compile(r'.*info(\d+).json')
        regex_result = re.search(regex, self.json_path)
        if regex_result is None:
            return "", False
        else:
            index = regex_result.group(1)
            costmap_files = os.path.join(
                '/'.join(self.json_path.split('/')[0:-1]),
                'boundingBox' + str(index) + '.png'
            )
            return costmap_files, True

    def __len__(self):
        return len(self.frontier_bounding_boxes)

    def __getitem__(self, args):
        item, gt_dict = args

        costmap_path = self.get_costmap_path()
        costmap = cv2.imread(costmap_path)
        map_name = self.get_map_name()
        gt = gt_dict[map_name]
        rect = self.frontier_bounding_boxes[item]
        frontiers = self.frontier_cluster_points[item]

        ratio = const.ORIGINAL_RESOLUTION / const.TARGET_RESOLUTION
        costmap_resized = cv2.resize(costmap, (0, 0), fx=ratio, fy=ratio, interpolation=const.RESIZE_INTERPOLATION)
        gt_resized = cv2.resize(gt, (0, 0), fx=ratio, fy=ratio, interpolation=const.RESIZE_INTERPOLATION)
        rect_resized = resize_rect(rect, ratio)
        frontiers_resized = resize_frontiers(frontiers, ratio)

        image_size = costmap_resized.shape
        pad_image_size = (image_size[0] + const.TARGET_HEIGHT, image_size[1] + const.TARGET_WIDTH)

        # get the mask size
        if const.MASK_SIZE_FROM_FRONTIER:
            mask_size = (int(rect_resized[2] * const.FRONTIER_MASK_RESIZE_FACTOR),
                         int(rect_resized[3] * const.FRONTIER_MASK_RESIZE_FACTOR))
        else:
            mask_size = [const.PREDICTION_HEIGHT, const.PREDICTION_WIDTH]

        gt_pad = pad_image(gt_resized, pad_image_size)
        costmap_pad = pad_costmap(costmap_resized, pad_image_size)
        rect_pad = pad_rect(rect_resized, image_size, pad_image_size)
        frontiers_pad = pad_frontiers(frontiers_resized, image_size, pad_image_size)
        mask_pad = get_mask_image(rect_pad, mask_size, pad_image_size)
        final_image_size = (const.TARGET_HEIGHT, const.TARGET_WIDTH)

        # get the Area to be cropped as input
        crop_rect = enlarge_rect(rect_pad, final_image_size)
        # crop the final map
        costmap_final = clip_image(costmap_pad, crop_rect)
        gt_final = clip_image(gt_pad, crop_rect)
        mask_final = clip_image(mask_pad, crop_rect)
        frontiers_final = clip_frontiers(frontiers_pad, crop_rect)

        # normalize the images
        input_image = np.dstack((costmap_final, mask_final)).astype(dtype=np.float32) / 255.0
        input_gt = gt_final.astype(dtype=np.float32) / 255.0
        input_gt = np.expand_dims(input_gt, -1)

        # H * W * D -> D * H * W
        input_image = input_image.transpose(2, 0, 1)
        input_gt = input_gt.transpose(2, 0, 1)

        # return extra information as a dict of list of frontiers with only one element + order is (x, y, ...)
        #  to be consistent with dataset with multiple frontiers in a single image
        return input_image, input_gt, \
               {
                   'Frontiers': [[(i[1], i[0], i[2]) for i in frontiers_final]],
                   'BoundingBoxes': [[crop_rect[1], crop_rect[0], crop_rect[3], crop_rect[2]]]
               }


if __name__ == "__main__":
    original_dataset_dir = "/home/bird/data/kth_floorplan_clean_categories"
    xml_files = glob.glob(os.path.join(original_dataset_dir, '**', '*.xml'), recursive=True)
    xml_files_split = [i.split('/') for i in xml_files]
    xml_names = [split[-1][0:-4] for split in xml_files_split]
    floorplans = [(FloorPlanGraph(file_path=xml_file).to_image(0.1, (1360, 1020))*255).astype(dtype=np.uint8) for xml_file in xml_files]
    floorplan_dict = dict(zip(xml_names, floorplans))

    json_path = "/home/bird/data/floorplan_results_split/training/middle/50010539/1/info4.json"
    info = OneDataInfo(json_path)
    input, gt, frontiers = info.__getitem__(0, floorplan_dict)
    input_image = input.transpose(1, 2, 0)
    input_image[:, :, 3] = input_image[:, :, 3] * 0.5 + 0.5
    input_image[:, :, 2].fill(0)
    for frontier in frontiers:
        input_image[frontier[0], frontier[1], 2] = 1.0

    img_255 = (input_image * 255).astype(np.uint8)
    cv2.imwrite("/tmp/test.png", img_255)
    import matplotlib.pyplot as plt
    ff = plt.imshow(input_image)

    input_without_mask = input[0:3, :, :].transpose(1, 2, 0)
    input_mask = input[3, :, :]
    import matplotlib.pyplot as plt
    plt.imshow(input_without_mask)
    plt.imshow(input_mask)

