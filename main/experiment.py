__author__ = 'bk'


import movement_data as md
import lstm
import reg
import utils

file_attribute = ['normal', 'outlier', 'complex', 'chaotic', 'fluctuation', 'complex']
file_format = '.png'

history_size = 20
stack = 2
rho = 0.93
epochs = 70
slice_step = 1

def do_experiment(f_attr):
    md.history_size = history_size

    for data_file_path in md.get_file_list(f_attr):
        x_train, y_train, x_test, y_test = md.load_data_set(data_file_path, True)

        lstm.train_lstm(stack, rho, epochs, slice_step,
                        x_train, y_train, x_test, y_test,
                        data_file_path, f_attr)

        reg.train_regression(x_train, y_train, x_test, y_test,
                             data_file_path, f_attr)

if __name__ == '__main__':

    for attr in file_attribute:
        do_experiment(attr)

        utils.show_final_error_result()
        utils.init_error()
