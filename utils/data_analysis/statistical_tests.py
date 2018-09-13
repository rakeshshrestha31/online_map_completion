#!/usr/bin/env python3
import argparse
import json
import functools
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import os

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


                plot_name = '{}_{}_{}'.format(floorplan_name, percentage, x_label_alias)
                plt.title('hist_'+plot_name)
                plt.xlabel(str(percentage) + "% area coverage time")
                plt.ylabel('probability')

                # plt.xlabel(x_label_alias)
                # plt.ylabel(y_label_alias)

                plt.legend(loc='lower right')
                plt.title("{}".format(plot_name))
                plt.savefig('/tmp/{}_{}_{}_{}.png'.format(floorplan_name, algorithm, percentage, x_label_alias))
                plt.savefig('/tmp/{}_{}_{}_{}.eps'.format(floorplan_name, algorithm, percentage, x_label_alias))
                # plt.show()


def sort_floorplan(test_data, floorplan_names, test_type):
    for percentage in test_data.keys():
        floorplans_data = []
        for floorplan_name in floorplan_names:
            algorithms_data = []
            for algorithm in test_data[percentage].keys():
                algorithms_data.append((algorithm, test_data[percentage][algorithm][floorplan_name][test_type]))
            algorithms_data = OrderedDict(algorithms_data)
            floorplans_data.append((floorplan_name, algorithms_data))
        floorplans_data = OrderedDict(floorplans_data)

    avg_flooplans_data = [
        (floorplan_name, np.mean(list(floorplans_data[floorplan_name].values())))
        for floorplan_name in floorplans_data.keys()
    ]

    sorted_floorplans_data = OrderedDict(sorted(avg_flooplans_data, key=lambda x: x[1]))
    # print(sorted_floorplans_data)
    return sorted_floorplans_data


def filter_t_test_data(t_test_data, max_size, null_algorithm='250_info'):
    """
    filters the data to be the same size for all algorithms
    :param t_test_data:
    :return: updated t_test data
    """
    for percentage in sorted(t_test_data.keys()):
        for algorithm_idx, algorithm in enumerate(t_test_data[percentage].keys()):
            if algorithm not in t_test_data[percentage] or algorithm == null_algorithm:
                continue
            y = []
            for floorplan_name in t_test_data[percentage][algorithm].keys():
                # use the best tests from all the algorithms
                t_test_data[percentage][null_algorithm][floorplan_name]['data'].sort()
                null_data = t_test_data[percentage][null_algorithm][floorplan_name]['data'] \
                    = t_test_data[percentage][null_algorithm][floorplan_name]['data'][:max_size[percentage]]

                t_test_data[percentage][algorithm][floorplan_name]['data'].sort()
                algorithm_data = t_test_data[percentage][algorithm][floorplan_name]['data'] \
                    = t_test_data[percentage][algorithm][floorplan_name]['data'][:max_size[percentage]]

                t, p = scipy.stats.ttest_ind(algorithm_data, null_data, equal_var=False)
                # 2xp cuz two tailed analysis
                if algorithm not in t_test_data[percentage]:
                    t_test_data[percentage][algorithm] = {}
                Q3 = np.percentile(algorithm_data, 75)
                Q1 = np.percentile(algorithm_data, 25)

                t_test_data[percentage][algorithm][floorplan_name] = {
                    't': t,
                    'p': 2 * p,
                    'mean': np.mean(algorithm_data),
                    'median': np.median(algorithm_data),
                    'stddev': np.std(algorithm_data),
                    'Q1': Q1,
                    'Q3': Q3,
                    'data': algorithm_data
                }
    return t_test_data


def t_confidence_map(degrees_of_freedom):
    t_confidence_dict = {
        16: 2.12,
        18: 2.101,
        20: 2.086,
    }

    if degrees_of_freedom in t_confidence_dict:
        return t_confidence_dict[degrees_of_freedom]
    else:
        if degrees_of_freedom > 60:
            return 2.00
        elif degrees_of_freedom > 40:
            return 2.021
        elif degrees_of_freedom > 30:
            return 2.042



def show_t_score(null_algorithm='ig_hector'):
    """
    based on https://towardsdatascience.com/inferential-statistics-series-t-test-using-numpy-2718f8f9bf2f
    :param test_type:
    :param null_algorithm:
    :return:
    """
    plt.rcParams["figure.figsize"] = (24, 8) #8)
    plt.rcParams["savefig.dpi"] = 120

    percentages = ['75', '85', '95']
    skip_floorplans = ['50052755', '50057023', '50055642']

    floorplan_names = list(filter(lambda x: x not in skip_floorplans, common_floorplan_names))
    algorithms = list(all_arrival_time_data_dict.keys())

    for eval_metric in eval_metrics:
        x_label_alias, _ = exploration_efficiency_visualization.getXYLabel(eval_metric)
        t_test_data = {}
        min_expts = {percentage: float('inf') for percentage in percentages}
        for floorplan_name in floorplan_names:
            if floorplan_name in skip_floorplans:
                continue
            for percentage in percentages:
                # if float(percentage) > 95:
                #     continue

                if percentage not in t_test_data:
                    t_test_data[percentage] = {}

                null_data = all_arrival_time_data_dict[null_algorithm][eval_metric][floorplan_name][percentage]

                plt.clf()
                for algorthm_idx, algorithm in enumerate(all_arrival_time_data_dict.keys()):
                    # if algorithm == null_algorithm:
                    #     continue
                    algorithm_data = all_arrival_time_data_dict[algorithm][eval_metric][floorplan_name][percentage]
                    min_expts[percentage] = min(min_expts[percentage], len(algorithm_data))

                    t, p = scipy.stats.ttest_ind(algorithm_data, null_data, equal_var=False)
                    # 2xp cuz two tailed analysis
                    if algorithm not in t_test_data[percentage]:
                        t_test_data[percentage][algorithm] = {}
                    Q3 = np.percentile(algorithm_data, 75)
                    Q1 = np.percentile(algorithm_data, 25)
                    t_test_data[percentage][algorithm][floorplan_name] = {
                        't': t,
                        'p': 2 * p,
                        'mean': np.mean(algorithm_data),
                        'median':  np.median(algorithm_data),
                        'stddev': np.std(algorithm_data),
                        'Q1': Q1,
                        'Q3': Q3,
                        'data': algorithm_data
                    }

        print('min experiments:', min_expts)
        filter_t_test_data(t_test_data, min_expts, null_algorithm)

        sorted_floorplans_data = sort_floorplan(t_test_data, floorplan_names, 'mean') #test_type)

        for test_type in ['p', 't']:
            for percentage in sorted(t_test_data.keys()):
                degrees_of_freedom = 2 * min_expts[percentage] - 2
                critical_t = t_confidence_map(degrees_of_freedom)

                plt.clf()
                for algorithm_idx, algorithm in enumerate(algorithms):
                    if algorithm not in t_test_data[percentage]: # or algorithm == null_algorithm:
                        continue
                    y = []
                    for floorplan_name in sorted_floorplans_data.keys():
                        y.append(t_test_data[percentage][algorithm][floorplan_name][test_type])
                    x = list(range(len(y)))
                    plt.plot(x, y, label=algorithm, color=exploration_efficiency_visualization.COLORS[algorithm_idx])
                    plt.scatter(x, y, color=exploration_efficiency_visualization.COLORS[algorithm_idx])
                    plt.xticks(x, list(sorted_floorplans_data.keys()))

                # 90% interval: 1.734, 95%: 2.1
                # confidence_interval = 1.734 if test_type == 't' else 0.1
                confidence_interval = critical_t if test_type == 't' else 0.05
                plt.axhline(y=confidence_interval, linestyle='dotted')

                plt.xlabel('floor plans')
                plt.ylabel(test_type + ' score for ' + x_label_alias)
                # plt.legend(loc='lower right')
                plot_title = "{}_{}_{}".format(test_type+'test', percentage, x_label_alias)
                # plt.title(plot_title)
                plt.savefig(os.path.join('/tmp/', plot_title + '.png'))
                plt.savefig(os.path.join('/tmp/', plot_title + '.eps'))
                # plt.show()


        # plot the mean and stddevs
        for percentage in sorted(t_test_data.keys()):
            for average in ['mean', 'stddev']:
                plt.clf()
                for algorithm_idx, algorithm in enumerate(algorithms):
                    stats = ['mean', 'stddev'] if average == 'mean' else ['median', 'Q1', 'Q3']

                    algorithm_stats = {
                        i: [] for i in stats
                    }
                    for floorplan_name in sorted_floorplans_data.keys():
                        for stat in stats:
                            algorithm_stats[stat].append(t_test_data[percentage][algorithm][floorplan_name][stat])

                        if average == 'mean':
                            y = algorithm_stats['mean']
                            error = algorithm_stats['stddev']
                        else:
                            average = 'median'
                            y = algorithm_stats['median']
                            error_hi = np.asarray(algorithm_stats['Q3']) - np.asarray(algorithm_stats['median'])
                            error_low = np.asarray(algorithm_stats['median']) - np.asarray(algorithm_stats['Q1'])
                            error = np.stack((error_low, error_hi))

                        x = list(range(len(y)))

                    plt.plot(x, y, label=algorithm, color=exploration_efficiency_visualization.COLORS[algorithm_idx])
                    plt.scatter(x, y, color=exploration_efficiency_visualization.COLORS[algorithm_idx])
                    # plt.errorbar(x, y, yerr=error,
                    #              color=exploration_efficiency_visualization.COLORS[algorithm_idx], alpha=0.7,
                    #              capsize=20, elinewidth=3)

                plt.xticks(x, list(sorted_floorplans_data.keys()))
                plt.xlabel('floor plans')
                plt.ylabel(average + ' ' + x_label_alias)

                plt.legend(loc='lower right')

                plot_title = "{}_{}_{}".format(average+'test', percentage, x_label_alias)
                # plt.title(plot_title)
                plt.savefig(os.path.join('/tmp/', plot_title + '.png'))
                plt.savefig(os.path.join('/tmp/', plot_title + '.eps'))


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
    show_t_score('ig_cost_utility')

