#!/usr/bin/env python3
import argparse
import json
import functools

from exploration_efficiency_visualization import visualize_floorplan, compare_outputs, group_outputs, TRAJECTORY_LABEL, SIM_TIME_LABEL
import exploration_efficiency_visualization
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Cached Exploration Results')
    parser.add_argument('cached_avg_data', type=str, metavar='S',
                        help='cached average data (json file)')
    args = parser.parse_args()
    with open(args.cached_avg_data, 'r') as f:
        all_avg_data_dict = json.load(f)
    common_floorplan = list(next(iter(all_avg_data_dict.values())).keys())
    
    outputs = []
    for floorplan in common_floorplan:
        for x_label in [SIM_TIME_LABEL]:
            label_data_tuples = sorted(list(all_avg_data_dict.items()))
            outputs.extend(visualize_floorplan(
                [i[1] for i in label_data_tuples], 
                [i[0] for i in label_data_tuples], 
                floorplan, 
                data_type=x_label
            ))
    grouped_outputs = group_outputs(outputs)
    exploration_efficiency_visualization.plot_grouped_avg_results(grouped_outputs)

    # outputs = sorted(outputs, key=functools.cmp_to_key(compare_outputs))
    print(json.dumps(grouped_outputs, indent=4))
    
