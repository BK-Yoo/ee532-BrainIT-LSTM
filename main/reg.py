__author__ = 'bk'

import utils
from sklearn import linear_model
import movement_data as md

REGR = "reg"

def train_regression(raw_data, train_point, data_file_path, file_attribute):

    x_train, y_train, x_test, y_test = md.preprocessing_data(raw_data, train_point)

    for axis_index in range(0, 3):
        regr_model = linear_model.LinearRegression()

        # Train the model using the training sets
        regr_model.fit(x_train[axis_index], y_train[axis_index])

        prediction = utils.predict_multi_step_ahead(regr_model, x_test[axis_index], md.NEXT)

        rmse, mae, mape = utils.calculate_error_measure(prediction, y_test[axis_index])
        file_name, file_dir, result_file = utils.make_result_file_dir(REGR, axis_index, rmse,
                                                                      data_file_path, file_attribute)
        utils.save_to_error_trunk(REGR, axis_index, file_name, rmse, mae, mape)
        utils.show_measure(REGR, axis_index, file_name)
        utils.save_result(REGR, axis_index,
                          prediction, y_test[axis_index], file_name, file_dir, result_file,
                          utils.make_metadata(REGR))
