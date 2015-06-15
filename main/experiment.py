__author__ = 'bk'


import movement_data as md
import lstm
import reg
import utils

file_attribute = ['normal', 'outlier', 'complex', 'chaotic', 'fluctuation']
file_format = '.png'

# data parameter
normalize = False
delay_step = 10
history_size = 10
slice_step = 1

# lstm parameter
stack = 1
parallel = True
rho = 0.93 # non parallel to 0.93
epochs = 50

def do_experiment(f_attr):
    md.delay_step = delay_step
    md.history_size = history_size

    for data_file_path in md.get_file_list(f_attr):

        raw_data_parallel, raw_data_non_parallel = md.load_raw_data(data_file_path, normalize)
        train_init_point = md.create_test_init_point(raw_data_parallel)

        lstm_raw_data = raw_data_parallel if parallel else raw_data_non_parallel
        lstm.train_lstm(stack, parallel, rho, epochs, slice_step,
                        lstm_raw_data, train_init_point, data_file_path, f_attr)

        reg.train_regression(raw_data_non_parallel, train_init_point, data_file_path, f_attr)

if __name__ == '__main__':

    # for attr in file_attribute:
    #     do_experiment(attr)
    do_experiment('normal')
    utils.show_final_error_result()
    utils.init_error()
