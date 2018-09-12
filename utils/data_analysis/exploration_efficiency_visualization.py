import os
import glob
import re
import numpy as np
import json
import itertools
import functools
import matplotlib.pyplot as plt
import argparse
from collections import OrderedDict
import copy
from multiprocessing import Pool
import gc # garbage collector
import time

# custom import
import utils.constants as const
from utils.data_analysis.percentage_coverage_evaluator import PercentageCoverageEvaluator

PERCENT_AREA_LABEL = "PercentExploredArea"
AREA_LABEL = "ExploredArea"
TRAJECTORY_LABEL = "TrajectoryLens"
SIM_TIME_LABEL = "SimulationTimeCost"
SYS_TIME_LABEL = "SystemTimeCost"

COLORS = [
    'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'
]

X_LABELS = [TRAJECTORY_LABEL, SIM_TIME_LABEL, SYS_TIME_LABEL]
BIN_INTERVELS = [const.TRAJECTORY_BIN_INTERVAL, const.SIM_TIME_BIN_INTERVAL, const.SYS_TIME_BIN_INTERVAL]
# Y_LABEL = AREA_LABEL

ALL_LABELS = [PERCENT_AREA_LABEL, AREA_LABEL, TRAJECTORY_LABEL, SIM_TIME_LABEL, SYS_TIME_LABEL]


def STD_LOG(log):
    print('[LOG]', time.time(), log)


def STD_ERR(log):
    print('[ERR]', time.time(), log)

def poses_to_step_length(poses):
    # push 0 for the first one
    step_lens = [0]
    for idx in range(len(poses) - 1):
        position_1 = [poses[idx]["x"], poses[idx]["y"]]
        position_2 = [poses[idx + 1]["x"], poses[idx + 1]["y"]]
        step_len = np.linalg.norm(np.asarray(position_1, dtype=np.float) - np.asarray(position_2, dtype=np.float))
        step_len = const.ORIGINAL_RESOLUTION * step_len  # pixels to meters
        step_lens.append(step_len)
    return step_lens


def times_to_time_interval(times):
    step_time = [0]
    for idx in range(len(times) - 1):
        time_interval = times[idx + 1] - times[idx]
        step_time.append(time_interval)

    return step_time


def parse_info(info):
    if info["RobotPoses"] is None or info["SystemTimes"] is None or info["SimulationTimes"] is None:
        info["StepLens"] = []
        info["StepSystemTime"] = []
        info["StepSimulationTime"] = []
        return info

    step_lens = poses_to_step_length(info["RobotPoses"])
    step_time_sys = times_to_time_interval(info["SystemTimes"])
    step_time_simulation = times_to_time_interval(info["SimulationTimes"])

    info["StepLens"] = step_lens
    info["StepSystemTime"] = step_time_sys
    info["StepSimulationTime"] = step_time_simulation
    return info


def read_one_plan(json_path):
    with open(json_path, 'r') as f:
        info = json.load(f)
        info = parse_info(info)
        return info


def read_one_exploration(directory):
    info_files = glob.glob(os.path.join(directory, 'info*.json'))

    regex = re.compile(r'.*info(\d+).json')
    file_indices = np.uint([re.search(regex, info_file).group(1) for info_file in info_files])
    max_index = np.max(file_indices)
    # TODO: remove
    # max_index = min(10, max_index)

    # if no info file or info file is not record continuously, pass
    if max_index < 0 or len(file_indices) < max_index:
        return None

    one_exploration_data = []
    # sequentially read all json file
    for i in range(max_index):
        idx = i + 1
        file_path = os.path.join(directory, 'info{}.json'.format(idx))
        one_plan_data = read_one_plan(file_path)
        one_exploration_data.append(one_plan_data)
    return one_exploration_data


def aggregate_one_explore_data(one_explore_data, total_area=100.0):
    areas = []
    step_lens = []
    step_sys_times = []
    step_sim_times = []
    
    for idx in range(len(one_explore_data)):
        if not one_explore_data[idx]["ExploredArea"] or not one_explore_data[idx]["StepSystemTime"] \
                or not one_explore_data[idx]["StepSimulationTime"] or not one_explore_data[idx]["ExploredArea"]:
            continue
        else:

            areas.extend(one_explore_data[idx]["ExploredArea"])
            step_lens.extend(one_explore_data[idx]["StepLens"])
            step_sim_times.extend(one_explore_data[idx]["StepSimulationTime"])
            step_sys_times.extend(one_explore_data[idx]["StepSystemTime"])

    trajectory_lens = list(itertools.accumulate(step_lens))
    sim_time_cost = list(itertools.accumulate(step_sim_times))
    sys_time_cost = list(itertools.accumulate(step_sys_times))

    # convert to secs
    sim_time_cost = list(map(lambda x: float(x) / 1e3, sim_time_cost))
    sys_time_cost = list(map(lambda x: float(x) / 1e3, sys_time_cost))

    aggregation = dict()
    aggregation[AREA_LABEL] = areas
    aggregation[PERCENT_AREA_LABEL] = [i / total_area * 100 for i in areas]
    aggregation[TRAJECTORY_LABEL] = trajectory_lens
    aggregation[SIM_TIME_LABEL] = sim_time_cost
    aggregation[SYS_TIME_LABEL] = sys_time_cost

    downsampling = 5
    aggregation = {
        label: aggregation[label][::downsampling]
        for label in aggregation.keys()
    }

    return aggregation

class InfoDataset:
    # static member to hold ground truth data (so that we don't parse it again and again)
    ground_truth_data = {}

    def __init__(self, directory, repeat_times, original_dataset_dir, label=''):
        print('parsing directory {}'.format(directory))
        self.label = label
        self.data = dict()
        self.directory = directory
        self.repeat_times = repeat_times
        self.original_dataset_dir = original_dataset_dir
        self.ground_truth_data = {}
        self.exploration_data = {} 
        self.max_labels = {label: {} for label in ALL_LABELS}
        self.finish_times = {}

        # floorplan wise max for all the labels
        self.update_ground_truth_data()
        self.load(directory, repeat_times)

        # now done for each file separately
        # self.aggregate_exploration_data()
        self.update_max() 
        self.update_finish_time_data()


    def load(self, directory, repeat_times):
        for floorplan_name in os.listdir(directory):
            self.data[floorplan_name] = {
                'area': 100.0, 
                'repeat_runs_data': []
            }

            if floorplan_name in self.ground_truth_data:
                self.data[floorplan_name]['area'] = self.ground_truth_data[floorplan_name]['area']

            floorplan_dir = os.path.join(directory, floorplan_name)
            for num in range(repeat_times):
                repeat_dir = os.path.join(floorplan_dir, str(num + 1))
                if os.path.exists(repeat_dir):
                    self.data[floorplan_name]['repeat_runs_data'].append(read_one_exploration(repeat_dir))
                else:
                    print('[ERROR] directory not found {}'.format(repeat_dir))
                    self.data[floorplan_name]['repeat_runs_data'].append(None)
        
            # update for all runs
            self.aggregate_exploration_data(self.data[floorplan_name], floorplan_name)
            # self.update_max()
            # self.update_finish_time_data()
            filename = '{},{}.json'.format(self.label, floorplan_name)
            STD_LOG('writing for {}'.format(filename)) 
            with open(os.path.join(args.logdir, filename), 'w') as f:
                json.dump(self.exploration_data[floorplan_name], f, indent=4)
                # self.exploration_data[floorplan_name] = filename

            # clear unnecessaries
            self.data[floorplan_name]['repeat_runs_data'] = None
        
        with open(os.path.join(args.logdir, '{},max_labels.json'.format(self.label)), 'w') as f:
            json.dump(self.max_labels, f, indent=4)
        
        with open(os.path.join(args.logdir, '{},finish_times.json'.format(self.label)), 'w') as f:
            json.dump(self.finish_times, f, indent=4)
    
    def update_ground_truth_data(self):
        from kth.FloorPlanGraph import FloorPlanGraph
        for floorplan_name in os.listdir(self.directory):
            if floorplan_name not in self.ground_truth_data:
                xml_files = glob.glob(
                    os.path.join(self.original_dataset_dir, '**', '{}.xml'.format(floorplan_name)), 
                    recursive=True
                )
                if len(xml_files) != 1:
                    print('[ERROR] ground truth files for floor plan {}: {}'.format(floorplan_name, xml_files))
                    self.ground_truth_data[floorplan_name] = None
                else:
                    graph = FloorPlanGraph(file_path=xml_files[0])
                    self.ground_truth_data[floorplan_name] = {
                        'graph': graph,
                        'area': functools.reduce(lambda accumulator, x: x * accumulator, graph.get_real_size(), 1.0),
                        'dim': graph.get_real_size()
                    }

    def aggregate_exploration_data(self, data, floorplan_name):
        floorplan_data = self.data[floorplan_name]['repeat_runs_data']
        self.exploration_data[floorplan_name] = []
        for t in range(self.repeat_times):
            if floorplan_data[t] is None:
                self.exploration_data[floorplan_name].append(None)
            else:
                one_exploration_data = aggregate_one_explore_data(floorplan_data[t], self.data[floorplan_name]['area'])
                self.exploration_data[floorplan_name].append(one_exploration_data)
        


    def average_floorplan_data(self):
        """
        :return: average and all data
        """
        floorplan_names = self.data.keys()
        data_types = ALL_LABELS
        average_data = dict()

        for floorplan_name in floorplan_names:
            one_floorplan_data = self.exploration_data[floorplan_name]
            aggregate_floorplan = dict()
            for type in data_types:
                aggregate_floorplan[type] = []
                # for each repeat of floorplan
                for one_exploration_data in one_floorplan_data:
                    if one_exploration_data is not None:
                        aggregate_floorplan[type].extend(one_exploration_data[type])

            x_types = X_LABELS
            x_intervals = BIN_INTERVELS

            average_floorplan = {}

            # y_data_numpy = np.asarray(aggregate_floorplan[AREA_LABEL])
            y_data_numpy = np.asarray(aggregate_floorplan[PERCENT_AREA_LABEL])
            for idx in range(len(x_types)):
                if not aggregate_floorplan[x_types[idx]]:
                    average_floorplan[x_types[idx]] = None
                    print('[ERROR] {} exploration data does not exist'.format(x_types[idx]))
                    continue

                floorplan_data_numpy = np.asarray(aggregate_floorplan[x_types[idx]])
                max_x = floorplan_data_numpy.max()
                # split x data from 0 to max_x + x_interval into bins with x_interval
                bins = np.arange(0, max_x + x_intervals[idx], x_intervals[idx])
                # x is in the center of each bin
                x = [bins[i] + x_intervals[idx] / 2.0 for i in range(len(bins) - 1)]

                digitized = np.digitize(floorplan_data_numpy, bins)

                # get y bins data from average
                y_data_bins = [y_data_numpy[digitized == i].mean() for i in range(1, len(bins))]
                # y_data_bins = [y_data_numpy[digitized == i].median() for i in range(1, len(bins))]

                y_data_bins_std = [y_data_numpy[digitized == i].std() for i in range(1, len(bins))]
                y_q1 = [np.percentile(y_data_numpy[digitized == i], 25) for i in range(1, len(bins))]
                y_q3 = [np.percentile(y_data_numpy[digitized == i], 75) for i in range(1, len(bins))]

                # store x_bins and y_data_bins
                average_floorplan[x_types[idx]] = {
                    "x": x, 
                    "y": y_data_bins, 
                    "y_std": y_data_bins_std,
                    "y_q1": y_q1,
                    "y_q3": y_q3
                }

            # store into average_data with floorplan_name as key
            average_data[floorplan_name] = average_floorplan

        return average_data

    def update_finish_time_data(self, output_labels=[SIM_TIME_LABEL, TRAJECTORY_LABEL]):
        common_floorplan_names = self.exploration_data.keys()

        percentages = [75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100]
        
        for label in output_labels:
            if label not in self.finish_times:
                self.finish_times[label] = {}
        
        for x_label in output_labels:
            for floorplan_name in common_floorplan_names:
                  
                if type(self.exploration_data[floorplan_name]) == str:
                    # STD_ERR('exploration data for ' + floorplan_name + ' already cached')
                    continue
               
                if floorplan_name not in self.finish_times[x_label]:
                    self.finish_times[x_label][floorplan_name] = {
                        percent: [] 
                        for percent in percentages
                    }
                
                for one_run_idx, one_run_data in enumerate(self.exploration_data[floorplan_name]):
                    if self.exploration_data[floorplan_name][one_run_idx] is None:
                        continue
                    y = self.exploration_data[floorplan_name][one_run_idx][PERCENT_AREA_LABEL]
                    x = self.exploration_data[floorplan_name][one_run_idx][x_label]

                    x_array = np.asarray(x)
                    y_array = np.asarray(y)
                    percentage_evaluator = PercentageCoverageEvaluator(x_array, y_array)

                    for percentage in self.finish_times[x_label][floorplan_name].keys():
                        self.finish_times[x_label][floorplan_name][percentage].append(
                            percentage_evaluator.evaluate_percent_coverage(percentage)
                        )
                # sort the percentage to ordered dicts
                self.finish_times[x_label][floorplan_name] = OrderedDict(list(
                    sorted(self.finish_times[x_label][floorplan_name].items(), key=lambda x: x[0])
                ))


    def update_max(self):
        common_floorplan_names = self.exploration_data.keys()

        for floorplan_name in common_floorplan_names:
            for one_run_idx, one_run_data in enumerate(self.exploration_data[floorplan_name]):
                if type(self.exploration_data[floorplan_name]) == str:
                    # STD_ERR('update max: exploration data for ' + floorplan_name + ' already cached')
                    continue

                if self.exploration_data[floorplan_name][one_run_idx] is None or \
                    type(self.exploration_data[floorplan_name][one_run_idx]) is str:
                    continue
                for label in self.exploration_data[floorplan_name][one_run_idx].keys():
                    if label not in self.max_labels:
                        self.max_labels[label] = {}
                    if floorplan_name not in self.max_labels[label]:
                        self.max_labels[label][floorplan_name] = float('-inf')
                    x = self.exploration_data[floorplan_name][one_run_idx][label]
                    self.max_labels[label][floorplan_name] = max(x[-1], self.max_labels[label][floorplan_name])


    # deprecated!!!
    def average_dataset(self):
        floorplan_names = self.data.keys()
        average_floorplan_data = self.average_floorplan_data()
        average_data = dict()
        x_types = X_LABELS
        max_len_xs = dict()
        for x_type in x_types:
            max_len_xs[x_type] = []

        # get x with max length
        for floorplan_name in floorplan_names:
            for x_type in x_types:
                if average_floorplan_data[floorplan_name][x_type] and \
                        len(average_floorplan_data[floorplan_name][x_type]["x"]) > len(max_len_xs[x_type]):
                    max_len_xs[x_type] = np.asarray(average_floorplan_data[floorplan_name][x_type]["x"], dtype=np.float)

        ys = dict()
        nums = dict()

        for x_type in x_types:
            ys[x_type] = np.zeros(len(max_len_xs[x_type]), dtype=np.float)
            nums[x_type] = np.zeros(len(max_len_xs[x_type]), dtype=np.float)

        for floorplan_name in floorplan_names:
            for x_type in x_types:
                if average_floorplan_data[floorplan_name][x_type]:
                    lens = len(average_floorplan_data[floorplan_name][x_type]["y"])
                    ys[x_type][0:lens] += average_floorplan_data[floorplan_name][x_type]["y"]
                    nums[x_type][0:lens] += 1

        # average y data
        for x_type in x_types:
            min_value = np.min(nums[x_type])
            if np.abs(min_value) < 1e-6:
                print("divide zero")
            ys[x_type] = ys[x_type] / nums[x_type]
            average_data[x_type] = {"x": max_len_xs[x_type], "y": ys[x_type]}

        return average_data


def getXYLabel(type):
    x_label = ""
    # y_label = "Explored area ($\mathregular{m^2}$)"
    y_label = "Area coverage (%age)"

    if type == SIM_TIME_LABEL:
        x_label = "Simulation time (s)"
    elif type == TRAJECTORY_LABEL:
        x_label = "Trajectory length (m)"
    elif type == SYS_TIME_LABEL:
        x_label = "System time (s)"

    return x_label, y_label


def visualize_average_floorplan(data_ig_gt, data_ig, data_no_ig, floorplan_name, data_type):

    x_label, y_label = getXYLabel(data_type)

    ig_x = data_ig[floorplan_name][data_type]["x"]
    ig_y = data_ig[floorplan_name][data_type]["y"]

    ig_gt_x = data_ig_gt[floorplan_name][data_type]["x"]
    ig_gt_y = data_ig_gt[floorplan_name][data_type]["y"]

    no_ig_x = data_no_ig[floorplan_name][data_type]["x"]
    no_ig_y = data_no_ig[floorplan_name][data_type]["y"]

    plt.clf()
    plt.plot(ig_x, ig_y, label='info_gain')
    plt.plot(ig_gt_x, ig_gt_y, label='info_gain_gt')
    plt.plot(no_ig_x, no_ig_y, label='no_info_gain')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='lower right')
    plt.title("{}".format(floorplan_name))
    plt.show()


def visualize_floorplan(avg_tests, test_labels, floorplan_name, data_type):

    plt.clf()
    maxes = []
    max_coverage = functools.reduce(
        lambda acc, x: x if x > acc else acc,
        map(lambda x: x[floorplan_name][data_type]["y"][-1], avg_tests),
        float('-inf')
    )
    for idx in range(len(avg_tests)):
        x_data = np.asarray(avg_tests[idx][floorplan_name][data_type]["x"])
        y_data = np.asarray(avg_tests[idx][floorplan_name][data_type]["y"]) * 100 / max_coverage
        y_std = np.asarray(avg_tests[idx][floorplan_name][data_type]["y_std"]) * 100 / max_coverage
        y_q1 = np.asarray(avg_tests[idx][floorplan_name][data_type]["y_q1"]) * 100 / max_coverage
        y_q3 = np.asarray(avg_tests[idx][floorplan_name][data_type]["y_q3"]) * 100 / max_coverage
        
        # print('dispersion:', y_std, y_q1, y_q3)
        plt.plot(x_data, y_data, label= test_labels[idx], color=COLORS[idx])
        
        alpha = 0.25 # 0.5
        plt.fill_between(x_data, y_q1, y_q3, alpha=alpha, color=COLORS[idx])
        # plt.fill_between(x_data, np.asarray(y_data) - np.asarray(y_std), np.asarray(y_data) + np.asarray(y_std), alpha=alpha, color=COLORS[idx])

        plt.axvline(x=x_data[-1], linestyle='dotted', color=COLORS[idx])
        maxes.append(x_data[-1])

    outputs = [] 
    for idx in range(len(avg_tests)):
        x_data = np.asarray(avg_tests[idx][floorplan_name][data_type]["x"])
        y_data = np.asarray(avg_tests[idx][floorplan_name][data_type]["y"])  * 100 / max_coverage

        percentage_evaluator = PercentageCoverageEvaluator(x_data, y_data)

        output_json = OrderedDict([
            ('floorplan', floorplan_name),
            ('label', test_labels[idx]), 
            ('type', data_type), 
            ('max_x', x_data[-1]), 
            ('max_y', y_data[-1]),
            ('80%', percentage_evaluator.evaluate_percent_coverage(80)),
            ('85%', percentage_evaluator.evaluate_percent_coverage(85)),
            ('90%', percentage_evaluator.evaluate_percent_coverage(90)),
            ('95%', percentage_evaluator.evaluate_percent_coverage(95)),
            ('99%', percentage_evaluator.evaluate_percent_coverage(99))
        ])
        # print(json.dumps(output_json, indent=4))
        outputs.append(output_json)

    x_label, y_label = getXYLabel(data_type)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='lower right')
    plt.title("{}".format(floorplan_name))

    # show the maxes too
    x_ticks, x_ticks_labels = plt.xticks()
    x_ticks, x_ticks_labels = extend_ticks(x_ticks, x_ticks_labels, maxes)

    plt.xticks(x_ticks, x_ticks_labels)
    plt.savefig('/tmp/{}_{}_{}.png'.format(floorplan_name, data_type, "_".join(test_labels)))
    plt.savefig('/tmp/{}_{}_{}.eps'.format(floorplan_name, data_type, "_".join(test_labels)))
    # plt.show()

    return output


def compute_max_labels(all_tests):
    """
    finds max labels (evaluation metrics) according for floor plan comparing all test algorithms
    :param all_tests:
    :return:
    """
    assert(len(all_tests))
    new_max_labels = all_tests[0].max_labels

    common_floorplan_names = all_tests[0].data.keys()
    for i in range(len(all_tests)):
        common_floorplan_names = common_floorplan_names & all_tests[i].data.keys()

    for label in new_max_labels:
        for floorplan_name in common_floorplan_names: # new_max_labels[label]:
            for test in all_tests:
                if floorplan_name not in new_max_labels[label] or floorplan_name not in test.max_labels[label]:
                    STD_ERR('no floorplan ' + floorplan_name)
                    continue
                new_max_labels[label][floorplan_name] = max(test.max_labels[label][floorplan_name],
                                                            new_max_labels[label][floorplan_name])

    return new_max_labels

def renormalize_coverage(all_tests, max_labels):
    """
    renormalize percentage area coverage based on max coverable area
    :param all_tests:
    :return:
    """
    for test in all_tests:
        for floorplan in test.exploration_data.keys():
            for one_run_idx, one_run_data in enumerate(test.exploration_data[floorplan]):
                if test.exploration_data[floorplan][one_run_idx] is None:
                    continue
                test.exploration_data[floorplan][one_run_idx][PERCENT_AREA_LABEL] = \
                    [i * 100.0 / max_labels[AREA_LABEL][floorplan] for i in test.exploration_data[floorplan][one_run_idx][AREA_LABEL]]


def group_outputs(outputs):
    """
    groups outputs based on floor plan, label and type (metric)
    :param outputs: list of dicts containing evaluations of individual floor plans, labels and metric type
    :return: dict of dict of dict (keys floor plan, label and type) containing evaluations
    """
    # grouped according to labels to plot
    label_grouped = {}
    # grouped according to floorplan (to compute averages for sorting)
    floorplan_grouped = {}

    labels = set(map(lambda x: x['label'], outputs))
    floorplans = sorted(set(map(lambda x: x['label'], outputs)))

    for output in outputs:
        floorplan = output['floorplan']
        label = output['label']
        type = output['type']

        if type not in floorplan_grouped:
            floorplan_grouped[type] = {}

        for metric in output:
            if metric in ['floorplan', 'label', 'type']:
                # not real comparison metric
                continue

            if metric not in floorplan_grouped[type]:
                floorplan_grouped[type][metric] = {}

            if floorplan not in floorplan_grouped[type][metric]:
                floorplan_grouped[type][metric][floorplan] = {}

            if label not in floorplan_grouped[type][metric][floorplan]:
                floorplan_grouped[type][metric][floorplan][label] = output[metric]


        # if label not in floorplan_grouped[floorplan][type]:
        #     floorplan_grouped[floorplan][type][label] = []
        #
        # floorplan_grouped[floorplan][type][label].append(
        #     OrderedDict(sorted(
        #         {i: output[i] for i in output if i not in ['floorplan', 'label', 'type']}.items()
        #     ))
        # )

    floorplan_avg_metric = copy.deepcopy(floorplan_grouped)
    for type in floorplan_avg_metric:
        for metric in floorplan_avg_metric[type]:
            for floorplan in floorplan_avg_metric[type][metric]:
                floorplan_avg_metric[type][metric][floorplan] = \
                    np.mean(list(floorplan_avg_metric[type][metric][floorplan].values()))

    sorted_floorplan = copy.deepcopy(floorplan_avg_metric)
    for type in sorted_floorplan:
        for metric in sorted_floorplan[type]:
            sorted_floorplan[type][metric] = OrderedDict(sorted(list(sorted_floorplan[type][metric].items()), key=lambda x: x[1]))

    # group according to label (algorithm)
    for type in floorplan_grouped:
        label_grouped[type] = {}
        for metric in floorplan_grouped[type]:
            label_grouped[type][metric] = {}
            for floorplan in floorplan_grouped[type][metric]:
                for label in floorplan_grouped[type][metric][floorplan]:
                    if label not in label_grouped[type][metric]:
                        label_grouped[type][metric][label] = {}
                    label_grouped[type][metric][label][floorplan] = floorplan_grouped[type][metric][floorplan][label]

    for type in label_grouped:
        for metric in label_grouped[type]:
            for label in label_grouped[type][metric]:
                label_grouped[type][metric][label] = OrderedDict(
                    [
                        (floorplan, label_grouped[type][metric][label][floorplan]) for floorplan in sorted_floorplan[type][metric].keys()
                    ]
                )

    # print(json.dumps(floorplan_avg_metric, indent=4))
    # print(json.dumps(sorted_floorplan, indent=4))
    return label_grouped

def plot_grouped_avg_results(label_grouped):
    for type in label_grouped:
        for metric in label_grouped[type]:
            plt.clf()
            for idx, label in enumerate(label_grouped[type][metric]):
                data = label_grouped[type][metric][label]
                x = list(range(len(data)))
                y = list(data.values())
                x_ticks_label = list(data.keys())

                x_label, y_label = getXYLabel(type)
                plt.xlabel("floor plans")
                plt.ylabel(metric)
                plt.plot(x, y, label=label, color=COLORS[idx])
                plt.scatter(x, y, color=COLORS[idx])
                plt.legend(loc='lower right')
                # plt.title("{}".format(floorplan_name))

                plt.xticks(x, x_ticks_label)

            plt.savefig('/tmp/{}.png'.format(metric))
            plt.savefig('/tmp/{}.eps'.format(metric))
            # plt.show()



def extend_ticks(x_ticks, x_ticks_labels, new_ticks):
    x_ticks = x_ticks.tolist()

    # indices_to_remove = []
    # half_tick_interval = (x_ticks[1] - x_ticks[0]) / 2.
    # for new_tick in new_ticks:
    #     for tick_idx in range(len(x_ticks)):
    #         if abs(x_ticks[tick_idx] - new_tick) < half_tick_interval:
    #             indices_to_remove.append(tick_idx)

    # x_ticks = [x for i, x in enumerate(x_ticks) if i not in indices_to_remove]
    # x_ticks.extend(new_ticks)
    # x_ticks.sort()
    x_ticks_labels = list(map(lambda x: ("%g" % x), x_ticks))

    return x_ticks, x_ticks_labels


def compare_outputs(output1, output2):
    "compares the dictionary outputs for sorting"
    if output1['floorplan'] != output2['floorplan']:
        return output1['floorplan'] < output2['floorplan']
    else:
        return output1['label'] < output2['label']


def parse_directory(directory):
    # nonlocal repeat, original_dataset_dir
    return InfoDataset(directory, repeat, original_dataset_dir)


def parse_directories(directories):
    all_tests = []
    common_floorplans = set()
    # keep index of each algorithm's each floorplan for later 
    all_data_index = {}
    for idx, directory in enumerate(directories):
        one_test = InfoDataset(directory, repeat, original_dataset_dir, labels[idx])
        all_tests.append(one_test)
        all_data_index[labels[idx]] = list(one_test.data.keys())

        for floorplan in one_test.data.keys():
            common_floorplans.add(floorplan)

        with open(os.path.join(args.logdir, 'common_floorplans.json'), 'w') as f:
            json.dump(list(common_floorplans), f, indent=4)

        gc.collect()
        # all_tests.append(one_test)
        # all_avg_floorplan_results.append(one_test.average_floorplan_data())
        # all_exploration_data.append(one_test.exploration_data)
    with open(os.path.join(args.logdir, 'all_data_index.json'), 'w') as f:
        json.dump(all_data_index, f, indent=4)

    return all_tests

def parse_cached():
    with open(os.path.join(args.logdir, 'all_data_index.json'), 'r') as f:
        all_data_index = json.load(f)
    for algorithm_idx, algorithm in enumerate(all_data_index.keys()):
        for floorplan in all_data_index[algorithm]:
            pass 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Exploration Results')
    parser.add_argument('original_dataset_dir', type=str, metavar='S',
                        help='location of original dataset (xml files)')
    parser.add_argument('results_dirs', type=str, metavar='S', nargs="+",
                        help='result directories, each directory corresponding to an experiment')
    parser.add_argument('--result_labels', type=str, metavar='S', nargs="+",
                        help='result labels, corresponding to each result directories')
    parser.add_argument('--repeat_times', type=int, default=10, metavar='N',
                        help='repeat times for each floorplan')
    parser.add_argument('--num_workers', type=int, default=1, metavar='N',
                        help='num processes to spawn')
    parser.add_argument('--logdir', type=str, default='/tmp', metavar='S',
                        help='directory of temporary files')

    args = parser.parse_args()

    original_dataset_dir = args.original_dataset_dir
    directories = args.results_dirs
    labels = args.result_labels
    repeat = args.repeat_times

    all_tests = []
    # all_explore_data = []
    all_avg_floorplan_results = []
    all_exploration_data = []
    all_arrival_time_data = []

    # p = Pool(args.num_workers)

    # all_tests = p.map(
    #     parse_directory,
    #     directories
    # )
    # # all_tests = map(parse_directory, directories)

    # for one_test in all_tests:
    #     # all_avg_floorplan_results.append(one_test.average_floorplan_data())
    #     # all_exploration_data.append(one_test.exploration_data)
    #     all_arrival_time_data.append(one_test.finish_time_data())
    
    all_tests = parse_directories(directories)
    parse_cached()

    max_labels = compute_max_labels(all_tests)
    with open(os.path.join(args.logdir, 'all_max_labels.json'), 'w') as f:
        json.dump(max_labels, f, indent=4)
    renormalize_coverage(all_tests, max_labels)

    for one_test in all_tests:
        all_arrival_time_data.append(one_test.finish_times)

    common_floorplan = all_tests[0].data.keys()

    for i in range(len(all_tests)):
        common_floorplan = common_floorplan & all_tests[i].data.keys()



    # save the data to avoid recomputations
    all_arrival_time_data = OrderedDict(zip(
        labels,
        [{
            eval_metric: {
                floorplan: all_arrival_time_data[i][eval_metric][floorplan]
                for floorplan in common_floorplan
            }
            for eval_metric in all_arrival_time_data[i].keys()

        } for i in range(len(all_arrival_time_data))]
    ))

    # all_data_dict = OrderedDict(zip(
    #     labels,
    #     [{floorplan: i.data[floorplan] for floorplan in common_floorplan} for i in all_tests]
    # ))
    # all_avg_data_dict = OrderedDict(zip(
    #     labels,
    #     [{floorplan: i[floorplan] for floorplan in common_floorplan} for i in  all_avg_floorplan_results]
    # ))
    # all_exploration_data = OrderedDict(zip(
    #     labels,
    #     [{floorplan: i[floorplan] for floorplan in common_floorplan} for i in  all_exploration_data]
    # ))

    # # with open('/tmp/all_data.json', 'w') as f:
    # #     json.dump(all_data_dict, f, indent=4)
    # with open('/tmp/all_avg_data.json', 'w') as f:
    #     json.dump(all_avg_data_dict, f, indent=4)
    # with open('/tmp/all_exploration_data.json', 'w') as f:
    #     json.dump(all_exploration_data, f, indent=4)
    with open('/tmp/all_arrival_time_data.json', 'w') as f:
        json.dump(all_arrival_time_data, f, indent=4)
    exit(0)

    outputs = []
    for floorplan in common_floorplan:
        # for test_idx in range(len(all_tests)):
        #     print('label: {}, floorplan: {})'.format(labels[test_idx], floorplan))
        #     for run_idx in range(len(all_tests[test_idx].data[floorplan])):
        #         print('total time: {}'.format(all_tests[test_idx].data[floorplan][run_idx][-1]['SimulationTimes'][-1]))

        for x_label in [TRAJECTORY_LABEL, SIM_TIME_LABEL]:
            label_data_tuples = sorted(list(all_avg_data_dict.items()))
            outputs.extend(visualize_floorplan(
                [i[1] for i in label_data_tuples],
                [i[0] for i in label_data_tuples],
                floorplan,
                data_type=x_label
            ))

    sorted(outputs, key=functools.cmp_to_key(compare_outputs))
    print(json.dumps(outputs, indent=4))
