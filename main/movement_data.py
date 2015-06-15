__author__ = 'bk'

import operator
import numpy as np
import os

# Used when the next value is delay-step-ahead value.
NEXT = 1

random_number = np.random.randint(100000)
np_rng = np.random.RandomState(random_number)

# put the path to folder
data_root_path = '/home/bk/KAIST/Brian IT/data/'

# the properties of data used
training_time = 15  # sec
testing_time  = 60  # sec
freq          = 50  # Hz, Data acquisition frequency
delay_time    = 0.2 # sec, system latency time
delay_step    = int(delay_time * freq) # delay step
history_size  = 2
num_of_axis   = 3
training_size = training_time * freq
testing_size = testing_time * freq


nan_file_list = ['t_DB29_Fx4_hilum', 't_DB03_Fx2', 't_DB44_Fx1']

test_file_list = ['t_DB13_Fx1', 't_DB34_Fx2', 't_DB18_Fx1']

# notorious data set
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

file_list = {'test': test_file_list, 'outlier': outlier_file_list,
             'complex': complex_pattern_file_list,
             'chaotic_t': chaotic_pattern_file_list_total,
             'chaotic_tr': chaotic_pattern_file_list_train,
             'chaotic_te': chaotic_pattern_file_list_test,
             'fluctuation': big_fluctuation_file_list, 'nan': nan_file_list}


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

def preprocessing_data(total_raw_data, test_init_point, parallel=False):

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

            # get random test sample from data.
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

if __name__ == "__main__":
    std_dict_train = {}
    std_dict_test = {}
    # std_dict_x = {}
    # std_dict_y = {}
    # std_dict_z = {}
    for f_name in get_file_list():
        _, raw_data_n_p = load_raw_data(f_name, False)
        _, y_train, _, y_test = preprocessing_data(raw_data_n_p, create_test_init_point(raw_data_n_p))
        std_dict_train[f_name] = sum([np.std(axis_data) for axis_data in y_train])
        std_dict_test[f_name] = sum([np.std(axis_data) for axis_data in y_test])
        # std_dict_x[f_name], std_dict_y[f_name], std_dict_z[f_name] = [np.std(axis_data) for axis_data in raw_data_n_p]

    # sorted_x = sorted(std_dict_x, key=operator.itemgetter(1))
    # sorted_y = sorted(std_dict_y, key=operator.itemgetter(1))
    # sorted_z = sorted(std_dict_z, key=operator.itemgetter(1))

    sorted_std_key_train = sorted(std_dict_train, key=std_dict_train.get, reverse=True)
    sorted_std_key_test = sorted(std_dict_test, key=std_dict_test.get, reverse=True)

    print [key.split('/')[-1].split('.')[0] for key in sorted_std_key_train if std_dict_train[key] > 3]
    print '================================'
    print ''

    print [key.split('/')[-1].split('.')[0] for key in sorted_std_key_test if std_dict_test[key] > 3]



# chaotic ['t_DB22_Fx3', 't_DB03_Fx1b']


