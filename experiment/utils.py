# -*- coding: utf-8 -*-
__author__ = 'bk'

import os

import matplotlib.pyplot as plt
import numpy as np

import model
from input import extractor as md

saved_file_format = '.png'
axis_str = ['x', 'y', 'z']

# rmse, map, mae
error = {'lstm': [{}, {}, {}], 'reg': [{}, {}, {}]}
result_dir = '../result/'


def init_error():
    error = {'lstm': [{}, {}, {}], 'reg': [{}, {}, {}]}


def predict_multi_step_ahead(model, test_input, lead_time, parallel=False):
    predicted_value = None

    if parallel:
        for index_cal in range(0, lead_time):
            predicted_value = model.predict(test_input)
            np.append(test_input[:, 1:], predicted_value, 1)

    else:
        for index_cal in range(0, lead_time):
            predicted_value = model.predict(test_input)
            test_input = np.append(test_input[:, 1:], np.array([predicted_value]).T, 1)

    return predicted_value


def make_metadata(model_structure, *metadata):
    history_length = 'Using ' + str(md.history_size) + ' stpes'
    predict_length = 'Predict ' + str(md.delay_step) + ' steps'

    if model_structure == model.REGR:
        return ' / '.join(['Regression', history_length, predict_length])

    else:
        stack = 'Stacked: ' + str(metadata[0])
        batch_size = 'batch_size: ' + str(metadata[1])
        training = 'Training: ' + str(metadata[2]) + ' times'

        return ' / '.join(['LSTM', history_length, predict_length, stack, batch_size, training])


def make_result_file_dir(model_structure, axis_idx, pred_error, file_path, file_attribute, *metadata):
    used_data_file = file_path.split('/')[-1].split('.')[0]

    if model_structure == model.REGR:
        f_name = '_'.join([str(pred_error), used_data_file, axis_str[axis_idx],
                           str(md.history_size), str(md.delay_step)])
        result_path = result_dir + 'control_group/'

    else:
        f_name = '_'.join([str(pred_error), used_data_file, axis_str[axis_idx],
                           str(md.history_size), str(md.delay_step),
                           str(metadata[0]), str(metadata[1]), str(metadata[2])])
        result_path = result_dir + 'experimental_group/'

    return used_data_file, result_path + file_attribute + '/', f_name + saved_file_format


def save_result(model, axis_index, cal_y, real_y, file_name, sav_dir, saved_file_name, additional_data):
    # Plot outputs
    fig = plt.figure()
    fig.suptitle('_'.join([file_name, axis_str[axis_index]]), fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)

    fig.subplots_adjust(top=0.85)

    ax.set_title(additional_data, fontsize=10)

    ax.set_xlabel('time(/20ms)')
    ax.set_ylabel('movement(mm)')

    ax.plot(cal_y, color='red', linestyle='-', label='prediction', linewidth=1)
    ax.plot(real_y, color='black', linestyle='--', label='actual', linewidth=0.7)

    ax.text(100, 10, 'red line is prediction', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad':10})

    ax.text(100, 13, 'RMSE: ' + str(error[model][0][file_name][axis_index]) +'mm', style='italic',
        bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    ax.axis([0, 3000, -16, 16])

    if not os.path.exists(sav_dir):
        os.makedirs(sav_dir)

    plt.savefig(sav_dir + saved_file_name, bbox_inches='tight', Frameon=True)


def calculate_error_measure(cal_y, real_y):
    rmse = np.sqrt(np.mean((cal_y - real_y) ** 2))
    mae = np.mean(np.abs(cal_y - real_y))
    mape = np.mean(np.abs(np.divide(cal_y - real_y, real_y)) * 100)
    return rmse, mae, mape


def save_to_error_trunk(model, axis, file_name, *errors):
    for idx, value in enumerate(errors):
        # to block the case which divided by zero
        if file_name not in error[model][idx]:
            error[model][idx][file_name] = []

        insert_val = value if value < 5000 else 5000
        error[model][idx][file_name].insert(axis, insert_val)


def show_measure(model, axis_index, file_name):
    # The mean square error
    print ('Root Mean Squared Error: %.6f' % error[model][0][file_name][axis_index])
    # The Mean Absolute Error
    print ('Mean Absolute Error: %.6f' % error[model][1][file_name][axis_index])
    # The Mean Absolute Percentage Error
    print ('Mean Absolute Percentage Error: %.6f' % error[model][2][file_name][axis_index])


def show_final_error_result():
    # The mean square error

    for model in error:
        error_array = []

        for index in range(0, 3):
            error_array.append([element for key in error[model][index]
                                for element in error[model][index][key]])

        for err_list in error_array:
            err_list.sort(reverse=True)

        total_rmse = np.mean(error_array[0])
        total_mae = np.mean(error_array[1])
        total_mape = np.mean(error_array[2])

        print ''
        print '############' + model + '############'
        print 'Root Mean Squared Error: %.6f' % total_rmse
        print 'Top 100'
        print error_array[0][:100]

        print 'Mean Absolute Error: %.6f' % total_mae
        print 'Top 100'
        print error_array[1][:100]

        print 'Mean Absolute Percentage Error: %.6f' % total_mape
        print ('Top 100')
        print error_array[2][:100]

        print '######################'
        print ''