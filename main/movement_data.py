__author__ = 'bk'


import numpy as np
import os
from random import randint

# put the path to folder
data_root_path = '/home/bk/KAIST/Brian IT/data/'

# the properties of data used
training_time = 15  # sec
testing_time  = 60  # sec
freq          = 50  # Hz, Data acquisition frequency
delay_time    = 0.2 # sec, system latency time
delay_step    = int(delay_time * freq) # delay step
history_size  = 30
num_of_axis   = 3
training_size = training_time * freq
testing_size = testing_time * freq


normal_file_list = ['t_DB13_Fx1', 't_DB26_Fx3']

# notorious data set
outlier_file_list = ['t_DB39_Fx3a', 't_DB17_Fx1', 't_DB33_Fx3', 't_DB45_Fx3', ' t_DB39_Fx3a',
                     't_DB42_Fx1a', 't_DB09_Fx2', 't_DB22_Fx2', 't_DB42_Fx1b', 't_DB42_Fx2', 't_DB40_Fx1']

complex_pattern_file_list = ['t_DB17_Fx1', 't_DB44_Fx1', 't_DB37_Fx1', 't_DB25_Fx3', 't_DB43_Fx3']

chaotic_pattern_file_list = ['t_DB22_Fx3', 't_DB03_Fx1b']

big_fluctuation_file_list = ['t_DB45_Fx1', 't_DB22_Fx1', 't_DB22_Fx2']


file_list = {'normal': normal_file_list, 'outlier': outlier_file_list,
             'complex': complex_pattern_file_list, 'chaotic': chaotic_pattern_file_list,
             'fluctuation': big_fluctuation_file_list}


def get_file_list(attribute='total'):
    if attribute == 'total':
        return [data_root_path + data_file_path for data_file_path in os.listdir(data_root_path)]
    else:
        return [data_root_path + data_file_path for data_file_path in os.listdir(data_root_path)
                if data_file_path.split('.')[0] in file_list[attribute]]

def load_raw_data(path):
    data = [[], [], []]
    print 'now, dive into '+path+'...'
    with open(path, mode='r') as data_file:
        for data_row in data_file:
            try:
                x, y, z, dummy = [float(element) for element in data_row.split()]
                data[0].append(x)
                data[1].append(y)
                data[2].append(z)

            except ValueError:
                continue
    return data

def normalize_vector(target_vector):
    target_vector_np = np.array(target_vector)
    std = np.std(target_vector_np)
    mean = np.mean(target_vector_np)

    return (target_vector_np - mean)/std

def preprocessing_data(total_raw_data, normalize=False):
    # movement matrix as [x, y, z]
    x_train = [[], [], []]
    y_train = [[], [], []]

    x_test = [[], [], []]
    y_test = [[], [], []]

    for axis_index in range(0, num_of_axis):
        train_data = total_raw_data[axis_index][:training_size]

        # get random test sample from data.
        test_init_point = randint(training_size+1, len(total_raw_data[axis_index]) - testing_size)
        test_data = total_raw_data[axis_index][test_init_point: test_init_point + testing_size]

        training_session = training_size - history_size

        # end of training session becomes the start of testing session
        testing_session = training_size

        max_iteration = training_size + testing_size - delay_step - history_size + 1

        for index in range(0, max_iteration):
            if index < training_session:
                index_train = index
                x_train[axis_index].append(train_data[index_train:(index_train + history_size)])

                # use one step ahead value to training the regression model
                index_train_y = index_train + history_size
                y_train[axis_index].append(train_data[index_train_y])

            elif index >= testing_session:
                index_test = index - training_size
                x_test[axis_index].append(test_data[index_test:(index_test + history_size)])

                # use the number of delay_step ahead value to training the regression model
                index_test_y = index_test + history_size + delay_step - 1
                y_test[axis_index].append(test_data[index_test_y])

    if normalize:
        x_train_norm = [normalize_vector(x_train_axis) for x_train_axis in x_train]
        y_train_norm = [normalize_vector(y_train_axis) for y_train_axis in y_train]
        x_test_norm = [normalize_vector(x_test_axis) for x_test_axis in x_test]
        y_test_norm = [normalize_vector(y_test_axis) for y_test_axis in y_test]

        return x_train_norm, y_train_norm, x_test_norm, y_test_norm

    else:
        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def load_data_set(path_to_data, normalize=True):
    raw_data = load_raw_data(path_to_data)

    # Split the data and the targets into training/testing sets
    return preprocessing_data(raw_data, normalize)

if __name__ == "__main__":
    test_file_list = [file_path for file_path in get_file_list('normal')]
    x_train, y_train, x_test, y_test = load_data_set(test_file_list[0], False)

    print len(x_test[0])


