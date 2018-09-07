import os
import glob
import re
import numpy as np
import json
import itertools
import matplotlib.pyplot as plt
import argparse

# custom import
import utils.constants as const

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

ALL_LABELS = [AREA_LABEL, TRAJECTORY_LABEL, SIM_TIME_LABEL, SYS_TIME_LABEL]


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


def aggregate_one_explore_data(one_explore_data):
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
    aggregation[TRAJECTORY_LABEL] = trajectory_lens
    aggregation[SIM_TIME_LABEL] = sim_time_cost
    aggregation[SYS_TIME_LABEL] = sys_time_cost

    return aggregation


class InfoDataset:
    def __init__(self, directory, repeat_times):
        self.data = dict()
        self.directory = directory
        self.repeat_times = repeat_times
        self.load(directory, repeat_times)

    def load(self, directory, repeat_times):
        for floorplan_name in os.listdir(directory):
            self.data[floorplan_name] = []
            floorplan_dir = os.path.join(directory, floorplan_name)
            for num in range(repeat_times):
                repeat_dir = os.path.join(floorplan_dir, str(num + 1))
                if os.path.exists(repeat_dir):
                    self.data[floorplan_name].append(read_one_exploration(repeat_dir))
                else:
                    self.data[floorplan_name].append(None)

    def aggregate_exploration_data(self):
        floorplan_names = self.data.keys()
        exploration_data = dict()
        for floorplan_name in floorplan_names:
            floorplan_data = self.data[floorplan_name]
            exploration_data[floorplan_name] = []
            for t in range(self.repeat_times):
                if floorplan_data[t] is None:
                    exploration_data[floorplan_name].append(None)
                else:
                    one_exploration_data = aggregate_one_explore_data(floorplan_data[t])
                    exploration_data[floorplan_name].append(one_exploration_data)
        return exploration_data

    def average_floorplan_data(self):
        floorplan_names = self.data.keys()
        exploration_data = self.aggregate_exploration_data()
        data_types = ALL_LABELS
        average_data = dict()
        for floorplan_name in floorplan_names:
            one_floorplan_data = exploration_data[floorplan_name]
            aggregate_floorplan = dict()
            for type in data_types:
                aggregate_floorplan[type] = []
                # for each repeat of floorplan
                for one_exploration_data in one_floorplan_data:
                    if one_exploration_data is not None:
                        aggregate_floorplan[type].extend(one_exploration_data[type])

            x_types = X_LABELS
            x_intervals = BIN_INTERVELS

            average_floorplan = dict()
            y_data_numpy = np.asarray(aggregate_floorplan[AREA_LABEL])
            for idx in range(len(x_types)):
                if not aggregate_floorplan[x_types[idx]]:
                    average_floorplan[x_types[idx]] = None
                    continue

                floorplan_data_numpy = np.asarray(aggregate_floorplan[x_types[idx]])
                max_x = floorplan_data_numpy.max()
                # split x data from 0 to max_x + x_interval into bins with x_interval
                bins = np.arange(0, max_x + x_intervals[idx], x_intervals[idx])
                # x is in the center of each bin
                x = [bins[i] + x_intervals[idx] / 2 for i in range(len(bins) - 1)]

                digitized = np.digitize(floorplan_data_numpy, bins)

                # get y bins data from average
                # y_data_bins = [y_data_numpy[digitized == i].mean() for i in range(1, len(bins))]
                y_data_bins = [y_data_numpy[digitized == i].mean() for i in range(1, len(bins))]

                y_data_bins_std = [y_data_numpy[digitized == i].std() for i in range(1, len(bins))]
                y_q1 = [np.percentile(y_data_numpy[digitized == i], 25) for i in range(1, len(bins))]
                y_q3 = [np.percentile(y_data_numpy[digitized == i], 75) for i in range(1, len(bins))]

                # store x_bins and y_data_bins
                average_floorplan[x_types[idx]] = {"x": x, "y": y_data_bins}

            # store into average_data with floorplan_name as key
            average_data[floorplan_name] = average_floorplan

        return average_data

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
    y_label = ""
    if type == SIM_TIME_LABEL:
        x_label = "Simulation time (s)"
        y_label = "Explored area ($\mathregular{m^2}$)"
    elif type == TRAJECTORY_LABEL:
        x_label = "Trajectory length (m)"
        y_label = "Explored area ($\mathregular{m^2}$)"
    elif type == SYS_TIME_LABEL:
        x_label = "System time (s)"
        y_label = "Explored area ($\mathregular{m^2}$)"

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
    for idx in range(len(avg_tests)):
        x_data = avg_tests[idx][floorplan_name][data_type]["x"]
        y_data = avg_tests[idx][floorplan_name][data_type]["y"]
        plt.plot(x_data, y_data, label= test_labels[idx], color=COLORS[idx])
        plt.axvline(x=x_data[-1], linestyle='dotted', color=COLORS[idx])
        maxes.append(x_data[-1])
        print('floorplan: {}, label: {}, type: {}, max: {}'.format(floorplan_name, test_labels[idx], data_type, x_data[-1]))

    x_label, y_label = getXYLabel(data_type)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='lower right')
    plt.title("{}".format(floorplan_name))

    # show the maxes too
    x_ticks, x_ticks_labels = plt.xticks()
    x_ticks, x_ticks_labels = extend_ticks(x_ticks, x_ticks_labels, maxes)

    plt.xticks(x_ticks, x_ticks_labels)
    plt.savefig('/tmp/' + floorplan_name + '_' + data_type + '.png')
    # plt.show()


def extend_ticks(x_ticks, x_ticks_labels, new_ticks):
    x_ticks = x_ticks.tolist()

    indices_to_remove = []
    half_tick_interval = (x_ticks[1] - x_ticks[0]) / 2.
    for new_tick in new_ticks:
        for tick_idx in range(len(x_ticks)):
            if abs(x_ticks[tick_idx] - new_tick) < half_tick_interval:
                indices_to_remove.append(tick_idx)

    x_ticks = [x for i, x in enumerate(x_ticks) if i not in indices_to_remove]
    x_ticks.extend(new_ticks)
    x_ticks.sort()
    x_ticks_labels = list(map(lambda x: ("%g" % x), x_ticks))

    return x_ticks, x_ticks_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Exploration Results')
    parser.add_argument('results_dirs', type=str, metavar='S', nargs="+",
                        help='result directories, each directory corresponding to an experiment')
    parser.add_argument('--result_labels', type=str, metavar='S', nargs="+",
                        help='result labels, corresponding to each result directories')
    parser.add_argument('--repeat_times', type=int, default=10, metavar='N',
                        help='repeat times for each floorplan')

    args = parser.parse_args()

    directories = args.results_dirs
    labels = args.result_labels
    repeat = args.repeat_times

    all_tests = []
    # all_explore_data = []
    all_avg_floorplan_results = []

    for directory in directories:
        one_test = InfoDataset(directory, repeat)
        all_tests.append(one_test)
        # all_explore_data.append(one_test.aggregate_exploration_data())
        all_avg_floorplan_results.append(one_test.average_floorplan_data())

    common_floorplan = all_tests[0].data.keys()

    for i in range(len(all_tests)):
        common_floorplan = common_floorplan & all_tests[i].data.keys()

    for floorplan in common_floorplan:
        # for test_idx in range(len(all_tests)):
        #     print('label: {}, floorplan: {})'.format(labels[test_idx], floorplan))
        #     for run_idx in range(len(all_tests[test_idx].data[floorplan])):
        #         print('total time: {}'.format(all_tests[test_idx].data[floorplan][run_idx][-1]['SimulationTimes'][-1]))

        for x_label in [TRAJECTORY_LABEL, SIM_TIME_LABEL]:
            visualize_floorplan(all_avg_floorplan_results, labels, floorplan, data_type=x_label)




    # dir_info_gain = args.results_dirs
    # dir_info_gain_gt = args.info_gain_gt_results_dir
    # dir_no_info_gain = args.no_info_gain_results_dir
    #
    #
    # data_no_ig = InfoDataset(dir_no_info_gain, repeat)
    # data_ig = InfoDataset(dir_info_gain, repeat)
    # data_ig_gt = InfoDataset(dir_info_gain_gt, repeat)
    #
    # explore_data_no_ig = data_no_ig.aggregate_exploration_data()
    # explore_data_ig = data_ig.aggregate_exploration_data()
    # explore_data_ig_gt = data_ig_gt.aggregate_exploration_data()
    #
    # avg_floorplan_data_no_ig = data_no_ig.average_floorplan_data()
    # avg_floorplan_data_ig = data_ig.average_floorplan_data()
    # avg_floorplan_data_ig_gt = data_ig_gt.average_floorplan_data()

    # avg_dataset_ig = data_ig.average_dataset()
    # avg_dataset_no_ig = data_no_ig.average_dataset()
    # plt.clf()
    # plt.plot(avg_dataset_ig["TrajectoryLens"]["x"], avg_dataset_ig["TrajectoryLens"]["y"], label='average_dataset_ig')
    # plt.plot(avg_dataset_no_ig["TrajectoryLens"]["x"], avg_dataset_no_ig["TrajectoryLens"]["y"],
    #          label='average_dataset_no_ig')
    # plt.show()

    # keys_no_ig = explore_data_no_ig.keys()
    # keys_ig = explore_data_ig.keys()
    # keys_ig_gt = explore_data_ig_gt.keys()
    #
    # keys_common = keys_no_ig & keys_ig & keys_ig_gt
    #
    # for key in keys_common:
    #
    #     visualize_average_floorplan(data_ig=avg_floorplan_data_ig, data_ig_gt=avg_floorplan_data_ig_gt,
    #                                 data_no_ig=avg_floorplan_data_no_ig, floorplan_name=key, data_type=SIM_TIME_LABEL)
    #
    #     visualize_average_floorplan(data_ig=avg_floorplan_data_ig, data_ig_gt=avg_floorplan_data_ig_gt,
    #                                 data_no_ig=avg_floorplan_data_no_ig, floorplan_name=key, data_type=TRAJECTORY_LABEL)
    #
    #     floorplan_data_no_ig = explore_data_no_ig[key]
    #     floorplan_data_ig = explore_data_ig[key]
    #     floorplan_data_ig_gt = explore_data_ig_gt[key]

        # for t in range(repeat):
        #     one_exploration_no_ig = floorplan_data_no_ig[t]
        #     one_exploration_ig = floorplan_data_ig[t]
        #
        #     if one_exploration_no_ig is None or one_exploration_ig is None:
        #         continue
        #     else:
        #         plt.clf()
        #         plt.plot(one_exploration_ig["TrajectoryLens"], one_exploration_ig["ExploredArea"],
        #                  label='info_gain')
        #         plt.plot(one_exploration_no_ig["TrajectoryLens"], one_exploration_no_ig["ExploredArea"],
        #                  label='no_info_gain')
        #         # plt.plot(one_exploration_ig["SimulationTimeCost"], one_exploration_ig["ExploredArea"],
        #         #          label='info_gain')
        #         # plt.plot(one_exploration_no_ig["SimulationTimeCost"], one_exploration_no_ig["ExploredArea"],
        #         #          label='no_info_gain')
        #         # plt.plot(one_exploration_ig["SystemTimeCost"], one_exploration_ig["ExploredArea"],
        #         #          label='info_gain')
        #         # plt.plot(one_exploration_no_ig["SystemTimeCost"], one_exploration_no_ig["ExploredArea"],
        #         #          label='no_info_gain')
        #         plt.legend(loc='lower right')
        #         plt.title("{}-{}".format(key, t + 1))
        #         plt.show()
