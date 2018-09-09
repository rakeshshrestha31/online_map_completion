#!/usr/bin/env python3
import argparse
import json
import functools
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

from exploration_efficiency_visualization import visualize_floorplan, compare_outputs, group_outputs, TRAJECTORY_LABEL, \
    SIM_TIME_LABEL, PERCENT_AREA_LABEL
import exploration_efficiency_visualization

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Cached Exploration Results')
    parser.add_argument('cached_exploration_data', type=str, metavar='S',
                        help='cached exploration data (json file)')
    args = parser.parse_args()
    with open(args.cached_exploration_data, 'r') as f:
        all_exploration_data_dict = json.load(f, object_pairs_hook=OrderedDict)

    common_floorplan_names = list(next(iter(all_exploration_data_dict.values())).keys())

    outputs = {
        floorplan_name: {
            algorithm: OrderedDict([
                (75, []), (80, []), (85, []), (100, [])
            ])
            for algorithm in all_exploration_data_dict.keys()
        }
        for floorplan_name in common_floorplan_names
    }

    for x_label in [SIM_TIME_LABEL]:
        x_label_alias, y_label_alias = exploration_efficiency_visualization.getXYLabel(x_label)
        for floorplan_name in common_floorplan_names:
            for algorthm_idx, algorithm in enumerate(all_exploration_data_dict.keys()):
                for one_run_idx, one_run_data in enumerate(all_exploration_data_dict[algorithm][floorplan_name]):
                    y = all_exploration_data_dict[algorithm][floorplan_name][one_run_idx][PERCENT_AREA_LABEL]
                    x = all_exploration_data_dict[algorithm][floorplan_name][one_run_idx][x_label]

                    x_array = np.asarray(x)
                    y_array = np.asarray(y)

                    plt.plot(
                        x_array, y_array,
                        label=algorithm, color=exploration_efficiency_visualization.COLORS[algorthm_idx]
                    )

                    for percentage in outputs[floorplan_name][algorithm].keys():
                        outputs[floorplan_name][algorithm][percentage].append(
                            exploration_efficiency_visualization.evaluate_percent_coverage(x_array, y_array, percentage)
                        )

                    plt.xlabel(x_label_alias)
                    plt.ylabel(y_label_alias)
                    plt.legend(loc='lower right')
                    plt.title("{}".format(floorplan_name))
                    plt.savefig('/tmp/{}_{}_{}.png'.format(floorplan_name, algorithm, one_run_idx))
                    plt.savefig('/tmp/{}_{}_{}.eps'.format(floorplan_name, algorithm, one_run_idx))
                    # plt.show()
                    plt.clf()


    for x_label in [SIM_TIME_LABEL]:
        x_label_alias, y_label_alias = exploration_efficiency_visualization.getXYLabel(x_label)
        for floorplan_name in common_floorplan_names:
            for algorthm_idx, algorithm in enumerate(all_exploration_data_dict.keys()):
                for percentage in outputs[floorplan_name][algorithm].keys():
                    x = outputs[floorplan_name][algorithm][percentage]
                    # for normalized hist
                    weights = np.ones_like(x) / float(len(x))
                    plt.hist(
                        x,
                        weights=weights,
                        label=algorithm,
                        color=exploration_efficiency_visualization.COLORS[algorthm_idx]
                    )
                    plt.title('hist_{}_{}_{}.png'.format(floorplan_name, algorithm, percentage))
                    plt.xlabel(str(percentage) + "% area coverage time")
                    plt.ylabel('probability')
                    plt.show()