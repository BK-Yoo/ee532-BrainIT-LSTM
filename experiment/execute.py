# -*- coding: utf-8 -*-
__author__ = 'bk'

from experiment import utils
from input import extractor as md
from model import reg
from model.raiman import raiman_lstm

lstm_models = {'bk': None, 'cho': None, 'raiman': raiman_lstm}

file_attribute = ['normal', 'outlier', 'complex', 'chaotic', 'fluctuation']
file_format = '.png'

# input parameter
normalize = False
delay_step = 10
history_size = 10
slice_step = 1

# lstm parameter
# 1,2 stack, non parallel, 100 seconds train, 0.93
# 1_1 stack, 150 seconds, 10 history, 50 epochs 0.6

# 2 stack, 100 seconds train, 0.52
stack = 1
parallel = True
training_method = 'adadelta'
rho = 0.62

epochs = 50


def do_experiment(f_attr):
    md.delay_step = delay_step
    md.history_size = history_size

    for data_file_path in md.get_file_list(f_attr):

        raw_data_parallel, raw_data_non_parallel = md.load_raw_data(data_file_path, normalize)
        train_init_point = md.create_test_init_point(raw_data_parallel)

        lstm_raw_data = raw_data_parallel if parallel else raw_data_non_parallel
        model = train_lstm('raiman')
        model.train_lstm(stack, parallel,
                        training_method, rho, epochs, slice_step,
                        lstm_raw_data, train_init_point, data_file_path, f_attr)

        reg.train_regression(raw_data_non_parallel, train_init_point, data_file_path, f_attr)


def train_lstm(model):
    if model == "bk":
        model = None
    elif model == 'cho':
        model = None
    elif model == 'raiman':
        model = lstm_models[model]
    return model


if __name__ == '__main__':
    do_experiment('small_test')
    utils.show_final_error_result()
    utils.init_error()
