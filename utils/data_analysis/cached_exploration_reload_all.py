#!/usr/bin/env python3
import argparse
import json
import ijson
import functools

from exploration_efficiency_visualization import visualize_floorplan, compare_outputs, group_outputs, TRAJECTORY_LABEL, \
    SIM_TIME_LABEL
import exploration_efficiency_visualization

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Cached Exploration Results')
    parser.add_argument('cached_all_data', type=str, metavar='S',
                        help='cached average data (json file)')
    parser.add_argument('cached_avg_data', type=str, metavar='S',
                        help='cached average data (json file)')

    args = parser.parse_args()
    with open(args.cached_avg_data, 'r') as f:
        all_avg_data_dict = json.load(f)
    common_floorplan = list(next(iter(all_avg_data_dict.values())).keys())

    labels = list(all_avg_data_dict.keys())

    with open(args.cached_all_data, 'r') as f:
        # all_data_dict = json.load(f)
        json_stream = ijson.parse(f)
        for prefix, event, value in json_stream:
            prefixes = prefix.split('.')

            print(prefix, event, value)


