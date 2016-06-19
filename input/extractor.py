# -*- coding: utf-8 -*-
__author__ = 'bk'

import numpy as np
import os

# Used when the next value is delay-step-ahead value.
NEXT = 1

random_number = np.random.randint(100000)
np_rng = np.random.RandomState(random_number)

# put the path to folder
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_root_path = os.path.join(BASE_DIR, 'resources/data/')

# the properties of input used
training_time = 120  # sec
testing_time  = 60   # sec
freq          = 50   # Hz, Data acquisition frequency
delay_time    = 0.2  # sec, system latency time
delay_step    = int(delay_time * freq)  # delay step
history_size  = 2
num_of_axis   = 3
training_size = training_time * freq
testing_size = testing_time * freq


final_test_list = ['t_DB15_Fx2', 't_DB14_Fx3', 't_DB24_Fx4', 't_DB15_Fx1', 't_DB36_Fx3',
                   't_DB30_Fx3', 't_DB36_Fx1', 't_DB14_Fx2', 't_DB46_Fx2', 't_DB46_Fx3',
                   't_DB32_Fx1', 't_DB46_Fx1', 't_DB01_Fx1', 't_DB29_Fx1', 't_DB01_Fx3',
                   't_DB01_Fx2', 't_DB30_Fx4', 't_DB02_Fx1', 't_DB30_Fx1', 't_DB44_Fx2',
                   't_DB44_Fx3', 't_DB12_Fx1', 't_DB04_Fx3b', 't_DB08_Fx2', 't_DB34_Fx3',
                   't_DB14_Fx5', 't_DB39_Fx3b', 't_DB35_Fx1', 't_DB40_Fx2', 't_DB16_Fx4',
                   't_DB24_Fx1', 't_DB03_Fx1a', 't_DB23_Fx3', 't_DB33_Fx2', 't_DB35_Fx2',
                   't_DB35_Fx3', 't_DB20_Fx3', 't_DB32_Fx3', 't_DB07_Fx5', 't_DB07_Fx4',
                   't_DB07_Fx3', 't_DB05_rtbronchus', 't_DB07_Fx1', 't_DB06_Fx1', 't_DB28_Fx2',
                   't_DB28_Fx3', 't_DB43_Fx5', 't_DB43_Fx4', 't_DB19_Fx2', 't_DB25_Fx2',
                   't_DB25_Fx1', 't_DB21_Fx3', 't_DB21_Fx2', 't_DB21_Fx1', 't_DB32_Fx2',
                   't_DB16_Fx3', 't_DB16_Fx2', 't_DB41_Fx2', 't_DB23_Fx1', 't_DB31_Fx1',
                   't_DB09_Fx3', 't_DB45_Fx2', 't_DB27_Fx1', 't_DB23_Fx2', 't_DB41_Fx1',
                   't_DB41_Fx3', 't_DB27_Fx2', 't_DB04_Fx3a', 't_DB06_Fx2', 't_DB33_FX1',
                   't_DB04_Fx4', 't_DB05_LUL', 't_DB11_Fx3', 't_DB28_Fx1', 't_DB26_Fx1',
                   't_DB26_Fx2', 't_DB26_Fx3', 't_DB05_RLL', 't_DB31_Fx2', 't_DB43_Fx1',
                   't_DB09_Fx1', 't_DB10_Fx3', 't_DB10_Fx2', 't_DB10_Fx1', 't_DB11_Fx1']

small_test_file_list = ['t_DB13_Fx1', 't_DB34_Fx2', 't_DB18_Fx1']

# notorious input set
outlier_file_list = ['t_DB39_Fx3a', 't_DB17_Fx1', 't_DB33_Fx3', 't_DB45_Fx3', ' t_DB39_Fx3a',
                     't_DB42_Fx1a', 't_DB09_Fx2', 't_DB22_Fx2', 't_DB42_Fx1b', 't_DB42_Fx2', 't_DB40_Fx1']

complex_pattern_file_list = ['t_DB17_Fx1', 't_DB44_Fx1', 't_DB37_Fx1', 't_DB25_Fx3', 't_DB43_Fx3']

# chaotic_pattern_file_list = ['t_DB22_Fx3', 't_DB03_Fx1b']

big_fluctuation_file_list = ['t_DB45_Fx1', 't_DB22_Fx1', 't_DB22_Fx2']

# 't_DB42_Fx1a', too big fluctuation
chaotic_pattern_file_list_total = ['t_DB44_Fx1', 't_DB22_Fx3', 't_DB17_Fx2',
                                   't_DB05_RLL_Fx1']

chaotic_pattern_file_list_train = ['t_DB17_Fx2', 't_DB40_Fx1', 't_DB22_Fx3', 't_DB17_Fx3',
                                   't_DB36_Fx2_liver', 't_DB45_Fx1']

chaotic_pattern_file_list_test = ['t_DB42_Fx2', 't_DB43_Fx2', 't_DB29_Fx2_hilum', 't_DB22_Fx3',
                                  't_DB15_Fx3', 't_DB17_Fx1']

file_list = {'small_test': small_test_file_list,
             'final_test': final_test_list,
             'outlier': outlier_file_list,
             'complex': complex_pattern_file_list,
             'chaotic_t': chaotic_pattern_file_list_total,
             'chaotic_tr': chaotic_pattern_file_list_train,
             'chaotic_te': chaotic_pattern_file_list_test,
             'fluctuation': big_fluctuation_file_list,}


def get_file_list(attribute='total'):
    if attribute == 'total':
        return [data_root_path + data_file_path for data_file_path in os.listdir(data_root_path)]

    elif attribute == 'normal':
        abnormal_file_list = [element for key in file_list for element in file_list[key]]
        return [data_root_path + data_file_path for data_file_path in os.listdir(data_root_path)
                if data_file_path.split('.')[0] not in abnormal_file_list]

    else:
        return [data_root_path + data_file_path for data_file_path in os.listdir(data_root_path)
                if data_file_path.split('.')[0] in file_list[attribute]]


def load_raw_data(path, normalize):
    data_non_parallel = [[], [], []]
    print 'now, dive into '+path+'...'
    with open(path, mode='r') as data_file:
        for data_row in data_file:
            try:
                x, y, z, _ = [float(element) for element in data_row.split()]

                data_non_parallel[0].append(x)
                data_non_parallel[1].append(y)
                data_non_parallel[2].append(z)

            except ValueError:
                continue

    if normalize:
        data_non_parallel = [normalize_vector(col_data) for col_data in data_non_parallel]

    data_non_parallel = np.array(data_non_parallel)
    data_parallel = np.array([row_data for row_data in data_non_parallel.T])

    return data_parallel, data_non_parallel


def normalize_vector(target_vector):
    target_vector_np = np.array(target_vector)
    std = np.std(target_vector_np)
    mean = np.mean(target_vector_np)

    return (target_vector_np - mean)/std


def create_test_init_point(total_raw_data):
    return 7000 # np_rng.randint(training_size+1, len(total_raw_data) - testing_size)


def pre_processing_data(total_raw_data, test_init_point, parallel=False):

    if parallel:
        x_train, y_train, x_test, y_test = [], [], [], []

        train = total_raw_data[:training_size, :]
        test = total_raw_data[test_init_point:test_init_point+testing_size, :]

        for row_idx in range(0, len(train) - history_size - delay_step):
            x_train.append([mov_data for h_step in range(0, history_size) for mov_data in train[row_idx + h_step]])
            y_train.append(train[row_idx + history_size, :])

        for row_idx in range(0, len(test) - history_size - delay_step):
            x_test.append([mov_data for h_step in range(0, history_size) for mov_data in test[row_idx + h_step]])
            y_test.append(test[row_idx + history_size + delay_step, :])

        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

    else:
        # movement matrix as [x, y, z]
        x_train = [[], [], []]
        y_train = [[], [], []]

        x_test = [[], [], []]
        y_test = [[], [], []]

        for axis_index in range(0, num_of_axis):
            train_data = total_raw_data[axis_index][:training_size]

            # get random experiment sample from input.
            test_data = total_raw_data[axis_index][test_init_point: test_init_point + testing_size]

            for row_index in range(0, len(train_data) - history_size - delay_step):
                x_train[axis_index].append(train_data[row_index:(row_index + history_size)])

                # use one step ahead value to training the regression model
                index_train_y = row_index + history_size + delay_step
                y_train[axis_index].append(train_data[index_train_y])

            for row_index in range(0, len(test_data) - history_size - delay_step):
                x_test[axis_index].append(test_data[row_index:(row_index + history_size)])

                # use the number of delay_step ahead value to training the regression model
                y_test[axis_index].append(test_data[row_index + history_size + delay_step])

        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

# chaotic ['t_DB22_Fx3', 't_DB03_Fx1b']
