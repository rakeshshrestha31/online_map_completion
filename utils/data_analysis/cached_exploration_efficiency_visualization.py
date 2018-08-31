#!/usr/bin/env python3
import argparse
import json

from exploration_efficiency_visualization import visualize_floorplan, TRAJECTORY_LABEL, SIM_TIME_LABEL

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Cached Exploration Results')
    parser.add_argument('cached_avg_data', type=str, metavar='S',
                        help='cached average data (json file)')
    args = parser.parse_args()
    with open(args.cached_avg_data, 'r') as f:
        all_avg_data_dict = json.load(f)
    common_floorplan = list(next(iter(all_avg_data_dict.values())).keys())
    
    for floorplan in common_floorplan:
        for x_label in [TRAJECTORY_LABEL, SIM_TIME_LABEL]:
            visualize_floorplan(list(all_avg_data_dict.values()), list(all_avg_data_dict.keys()), floorplan, data_type=x_label)
    
