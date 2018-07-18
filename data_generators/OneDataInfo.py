
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


def get_mask(image_size, mask_size):
    """generate mask in the center of the image, mask size is defined
    in constants.predicted_width and constants.predicted_height

    :arg image_size the generated mask image size
    :return images
    """
    center = [int(image_size[i] / 2) for i in range(2)]
    mask_half_size = [int(mask_size[i] / 2) for i in range(2)]
    start = [center[i] - mask_half_size[i] for i in range(2)]

    mask = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
    mask[start[0]:start[0]+mask_size[0], start[1]:start[1]+mask_size[1]].fill(255)

    return mask


def enlarge_rect(rect, rect_new_size):
    center = get_center(rect)
    start = [center[i] - int(rect_new_size[i] / 2) for i in range(2)]
    new_rect = [start[0], start[1], rect_new_size[0], rect_new_size[1]]
    return new_rect


def get_center(rect):
    center = (rect[0] + int(np.round(rect[2] / 2.0)), rect[1] + int(np.round(rect[3] / 2.0)))
    return center


def pad_rect(rect, original_size, new_size):
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


def parse_bounding_boxes(bounding_boxes_list):
    bounding_boxes = []
    for bounding_box in bounding_boxes_list:
        if bounding_box['height'] > const.FRONTIER_BOUNDING_BOX_MIN and \
                bounding_box['width'] > const.FRONTIER_BOUNDING_BOX_MIN:

            rect = [bounding_box['y'],
                    bounding_box['x'],
                    bounding_box['height'],
                    bounding_box['width']]
            bounding_boxes.append(rect)

    return bounding_boxes


class OneDataInfo:
    def __init__(self, json_path):
        self.frontier_bounding_boxes = []
        self.json_path = ""
        self.load(json_path)

    def load(self, json_path):
        self.json_path = json_path
        with open(self.json_path, 'r') as f:
            info = json.load(f)
            self.frontier_bounding_boxes = parse_bounding_boxes(info["BoundingBoxes"])

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

    def __getitem__(self, item, gt_dict):
        costmap_path = self.get_costmap_path()
        costmap = cv2.imread(costmap_path)
        map_name = self.get_map_name()
        gt = gt_dict[map_name]
        rect = self.frontier_bounding_boxes[item]
        original_size = costmap.shape
        new_size = (original_size[0] + const.TARGET_HEIGHT, original_size[1] + const.TARGET_WIDTH)

        gt_pad = pad_image(gt, new_size)
        costmap_pad = pad_costmap(costmap, new_size)
        rect_pad = pad_rect(rect, original_size, new_size)

        final_image_size = (const.TARGET_HEIGHT, const.TARGET_WIDTH)
        mask_size = [const.PREDICTION_HEIGHT, const.PREDICTION_WIDTH]
        rect_final = enlarge_rect(rect_pad, final_image_size)

        costmap_final = clip_image(costmap_pad, rect_final)
        gt_final = clip_image(gt_pad, rect_final)
        mask_final = get_mask(final_image_size, mask_size)

        # normalize the images
        input_image = np.dstack((costmap_final, mask_final)).astype(dtype=np.float32) / 255.0
        input_gt = gt_final.astype(dtype=np.float32) / 255.0
        input_gt = np.expand_dims(input_gt, -1)

        # H * W * D -> D * H * W
        input_image = input_image.transpose(2, 0, 1)
        input_gt = input_gt.transpose(2, 0, 1)

        return input_image, input_gt


if __name__ == "__main__":
    original_dataset_dir = "/home/bird/data/floorplan_categories"
    xml_files = glob.glob(os.path.join(original_dataset_dir, '**', '*.xml'), recursive=True)
    xml_files_split = [i.split('/') for i in xml_files]
    xml_names = [split[-1][0:-4] for split in xml_files_split]
    floorplans = [(FloorPlanGraph(file_path=xml_file).to_image(0.1, (1360, 1020))*255).astype(dtype=np.uint8) for xml_file in xml_files]
    floorplan_dict = dict(zip(xml_names, floorplans))

    json_path = "/home/bird/data/results/middle/50056459/1/info4.json"
    info = OneDataInfo(json_path)
    info.__getitem__(0, floorplan_dict)
