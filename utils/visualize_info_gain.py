
import json
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np


def visualize_info_gain_distribution(info_gains):
    max_value = np.max(info_gains)
    min_value = np.min(info_gains)

    if min_value < 500:
        min_value = 500

    total_num = len(info_gains)

    bin_num = 1000
    interval = (max_value - min_value) / bin_num

    x = np.arange(min_value, max_value, interval) + interval/2
    y = np.zeros(bin_num)
    for info_gain in info_gains:
        if info_gain < min_value:
            continue
        index = int(np.ceil((info_gain - min_value) / interval) - 1)
        y[index] += 1

    y = y / total_num

    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Visualize information gain')
    parser.add_argument('json_path', type=str, metavar='dir',
                        help='json file path')
    args = parser.parse_args()

    if os.path.isfile(args.json_path):
        with open(args.json_path, 'r') as f:
            data = json.load(f)
            info_gain = data["all_information_gain"]
            info_gain_gt = data["all_information_gain_gt"]
            visualize_info_gain_distribution(info_gain)
            visualize_info_gain_distribution(info_gain_gt)

