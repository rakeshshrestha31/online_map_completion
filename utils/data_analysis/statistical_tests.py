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
    parser.add_argument('cached_arrival_time_data', type=str, metavar='S',
                        help='cached arrival time data (json file)')
    args = parser.parse_args()
    with open(args.cached_arrival_time_data, 'r') as f:
        all_arrival_time_data_dict = json.load(f, object_pairs_hook=OrderedDict)

    common_floorplan_names = list(
        # floorplan data
        next(iter(
                # eval_metric data
                next(iter(all_arrival_time_data_dict.values())).values()
        )).keys()
    )

    eval_metrics = list(next(iter(all_arrival_time_data_dict.values())).keys())
    percentages = list(
        # percentage data
        next(iter(
            # floorplan data
            next(iter(
                    # eval_metric data
                    next(iter(all_arrival_time_data_dict.values())).values()
                )
            ).values()
        ))
    )
    for eval_metric in eval_metrics:
        x_label_alias, y_label_alias = exploration_efficiency_visualization.getXYLabel(eval_metric)
        for floorplan_name in common_floorplan_names:
            for percentage in percentages:
                plt.hold(True)
                for algorthm_idx, algorithm in enumerate(all_arrival_time_data_dict.keys()):
                    percentage_data = all_arrival_time_data_dict[algorithm][eval_metric][floorplan_name][percentage]
                    min_arrival_time = max(percentage_data) * 0.0
                    filtered_percentage_data = list(filter(lambda x: x > min_arrival_time, percentage_data))

                    weights = np.ones_like(filtered_percentage_data) / float(len(filtered_percentage_data))
                    plt.hist(
                        filtered_percentage_data,
                        weights=weights,
                        label=algorithm,
                        color=exploration_efficiency_visualization.COLORS[algorthm_idx],
                        alpha=0.5
                    )

                plot_name = '{}_{}_{}_{}'.format(algorithm, floorplan_name, percentage, eval_metric)
                print(plot_name, filtered_percentage_data, percentage_data)
                plt.title('hist_'+plot_name)
                plt.xlabel(str(percentage) + "% area coverage time")
                plt.ylabel('probability')

                # plt.xlabel(x_label_alias)
                # plt.ylabel(y_label_alias)

                plt.legend(loc='lower right')
                plt.title("{}".format(plot_name))
                plt.savefig('/tmp/{}_{}_{}_{}.png'.format(floorplan_name, algorithm, percentage, eval_metric))
                plt.savefig('/tmp/{}_{}_{}_{}.eps'.format(floorplan_name, algorithm, percentage, eval_metric))
                plt.show()


    for x_label in [SIM_TIME_LABEL]:
        for x_label in [SIM_TIME_LABEL]:
            x_label_alias, y_label_alias = exploration_efficiency_visualization.getXYLabel(x_label)
            for floorplan_name in common_floorplan_names:
                for algorthm_idx, algorithm in enumerate(all_arrival_time_data_dict.keys()):
                    for one_run_idx, one_run_data in enumerate(all_arrival_time_data_dict[algorithm][floorplan_name]):
                        y = all_arrival_time_data_dict[algorithm][floorplan_name][one_run_idx][PERCENT_AREA_LABEL]
                        x = all_arrival_time_data_dict[algorithm][floorplan_name][one_run_idx][x_label]

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