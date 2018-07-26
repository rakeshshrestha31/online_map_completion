#!/usr/bin/env python3

import cv2

ORIGINAL_RESOLUTION = 0.1
TARGET_RESOLUTION = 0.1

ORIGINAL_WIDTH = 1360
ORIGINAL_HEIGHT = 1020

TARGET_WIDTH = 256
TARGET_HEIGHT = 256

KLD_WEIGHT = 0.01

FRONTIER_MASK_RESIZE_FACTOR = 2.5

RESIZE_INTERPOLATION = cv2.INTER_CUBIC

# used when MASK_SIZE_FROM_FRONTIER=FALSE
PREDICTION_WIDTH = 80
PREDICTION_HEIGHT = 80

# ignore very small frontier area (height or width < 5)
FRONTIER_BOUNDING_BOX_MIN = 5

# True: MASK SIZE is according to Frontier Area * FRONTIER_MASK_RESIZE_FACTOR
# False: MASK_SIZE is fixed at PREDICTION_WIDTH * PREDICTION_HEIGHT
MASK_SIZE_FROM_FRONTIER = True

MAX_USE_EXPLORE_AREA = False

# Considering loss only in unknown part
LOSS_USE_ONLY_UNKNOWN = True

# Considering variable cost weight for obstacle and free space

"""
VARIABLE_COST_WEIGHT: use different loss weight for free and obstacle area  
valid value [-1, 0, 1],
-1: free weight is higher than obstacle weight
1: obstacle weight is higher than free weight
0: free and obstacle weight is the same at 0.5
"""
VARIABLE_COST_WEIGHT = 1



