#!/usr/bin/env python3
import argparse
import json
import functools
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from exploration_efficiency_visualization import visualize_floorplan, compare_outputs, group_outputs, TRAJECTORY_LABEL, \
    SIM_TIME_LABEL, PERCENT_AREA_LABEL
import exploration_efficiency_visualization

def show_histogram():
    for eval_metric in eval_metrics:
        x_label_alias, y_label_alias = exploration_efficiency_visualization.getXYLabel(eval_metric)
        for floorplan_name in common_floorplan_names:
            for percentage in percentages:
                if float(percentage) < 95:
                    continue

                plt.clf()
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
                        alpha=0.5,
                    )

                    # pdf_y, pdf_x = np.histogram(filtered_percentage_data)
                    # pdf_y = np.asarray(pdf_y) / sum(pdf_y)
                    # pdf_x  = pdf_x[1:]
                    # plt.plot(pdf_x, pdf_y,
                    #          label=algorithm, color=exploration_efficiency_visualization.COLORS[algorthm_idx],)

                    # print('{} {} {}: meandev:{}+/-{}, QD: {}'.format(
                    #     algorithm, floorplan_name, eval_metric,
                    #     np.mean(filtered_percentage_data),
                    #     np.std(filtered_percentage_data),
                    #     np.percentile(filtered_percentage_data, 75) - np.percentile(filtered_percentage_data, 25)
                    # ))


                plot_name = '{}_{}_{}'.format(floorplan_name, percentage, eval_metric)
                plt.title('hist_'+plot_name)
                plt.xlabel(str(percentage) + "% area coverage time")
                plt.ylabel('probability')

                # plt.xlabel(x_label_alias)
                # plt.ylabel(y_label_alias)

                plt.legend(loc='lower right')
                plt.title("{}".format(plot_name))
                plt.savefig('/tmp/{}_{}_{}_{}.png'.format(floorplan_name, algorithm, percentage, eval_metric))
                plt.savefig('/tmp/{}_{}_{}_{}.eps'.format(floorplan_name, algorithm, percentage, eval_metric))
                # plt.show()

def show_t_score(null_algorithm='250_info'):
    skip_floorplans = ['50052755', '50057023', '50055642']
    floorplan_names = list(filter(lambda x: x not in skip_floorplans, common_floorplan_names))

    for eval_metric in eval_metrics:
        t_test_data = {}
        for floorplan_name in floorplan_names:
            if floorplan_name in skip_floorplans:
                continue
            for percentage in percentages:
                if float(percentage) > 95:
                    continue

                if percentage not in t_test_data:
                    t_test_data[percentage] = {}

                null_data = all_arrival_time_data_dict[null_algorithm][eval_metric][floorplan_name][percentage]


                plt.clf()
                for algorthm_idx, algorithm in enumerate(all_arrival_time_data_dict.keys()):
                    if algorithm == null_algorithm:
                        continue
                    algorithm_data = all_arrival_time_data_dict[algorithm][eval_metric][floorplan_name][percentage]

                    t, p = scipy.stats.ttest_ind(null_data, algorithm_data, equal_var=False)
                    # 2xp cuz two tailed analysis
                    if algorithm not in t_test_data[percentage]:
                        t_test_data[percentage][algorithm] = {}

                    t_test_data[percentage][algorithm][floorplan_name] = {'t': t, 'p': 2 * p}

        for percentage in t_test_data.keys():
            floorplans_data = []
            for floorplan_name in floorplan_names:
                algorithms_data = []
                for algorithm in t_test_data[percentage].keys():
                    algorithms_data.append((algorithm, t_test_data[percentage][algorithm][floorplan_name]['t']))
                algorithms_data = OrderedDict(algorithms_data)
                floorplans_data.append((floorplan_name, algorithms_data))
            floorplans_data = OrderedDict(floorplans_data)

        avg_flooplans_data = [
            (floorplan_name, np.mean(list(floorplans_data[floorplan_name].values())))
            for floorplan_name in floorplans_data.keys()
        ]

        sorted_floorplans_data = OrderedDict(sorted(avg_flooplans_data, key=lambda x: x[1]))
        print(sorted_floorplans_data)

        for percentage in t_test_data.keys():
            plt.clf()
            for algorithm_idx, algorithm in enumerate(t_test_data[percentage].keys()):
                y = []
                for floorplan_name in sorted_floorplans_data.keys():
                    y.append(t_test_data[percentage][algorithm][floorplan_name]['t'])
                x = list(range(len(y)))
                plt.plot(x, y, label=algorithm, color=exploration_efficiency_visualization.COLORS[algorithm_idx])
                plt.scatter(x, y, color=exploration_efficiency_visualization.COLORS[algorithm_idx])
                plt.xticks(x, list(sorted_floorplans_data.keys()))

            # 90% interval: 1.734, 95%: -2.1
            plt.axhline(y=-1.734, linestyle='dotted')
            plt.xlabel('floor plans')
            plt.ylabel(eval_metric)
            plt.legend(loc='lower right')
            plt.title("{}".format(percentage + '_' + eval_metric))
            plt.show()


            # x = list(range(len(data)))
            # y = list(data.values())
            # x_ticks_label = list(data.keys())
            #
            # x_label, y_label = getXYLabel(type)
            # plt.xlabel("floor plans")
            # plt.ylabel(metric)
            # plt.plot(x, y, label=label, color=COLORS[idx])
            # plt.scatter(x, y, color=COLORS[idx])
            # plt.legend(loc='lower right')
            # # plt.title("{}".format(floorplan_name))
            #
            # plt.xticks(x, x_ticks_label)





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

    # show_histogram()
    show_t_score()

