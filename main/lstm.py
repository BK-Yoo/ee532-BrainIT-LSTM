"""
=====KAIST EE532 Brain IT Project=====

Using LSTM for making lung movement time-series
With Lung Movement Data(Composed of 3 axes data, x, y, z)

Using Raiman's theano_rnn library and referencing his tutorial.
[here](https://github.com/JonathanRaiman/theano_lstm)

@author: BK Yoo
@date: June 5 2015

##############################   ATTACK PLAN   #################################
# First, analyze nonsensical LSTM,
### get deep into how theano functions are used to train and build the model.

# Think about how to change the architecture of LSTM for x,y,z axis data.
### Analyze the code from thenano_lstm and Figure out how to put movement data.

# Train it and get a result.(By Saturday)
### Check error and analyze it.

"""

import theano
import theano.tensor as T

from theano_lstm import *

import movement_data as md
import datetime
import utils

lstm = 'lstm'

def has_hidden(layer):
    """
    Whether a layer has a trainable
    initial hidden state.
    """
    return hasattr(layer, 'initial_hidden_state')

def matrixify(vector, n):
    return T.repeat(T.shape_padleft(vector), n, axis=0)

def initial_state(layer, dimensions=None):
    """
    Initalizes the recurrence relation with an initial hidden state
    if needed, else replaces with a "None" to tell Theano that
    the network **will** return something, but it does not need
    to send it to the next step of the recurrence
    """
    if dimensions is None:
        return layer.initial_hidden_state if has_hidden(layer) else None
    else:
        return matrixify(layer.initial_hidden_state, dimensions) if has_hidden(layer) else None

def initial_state_with_taps(layer, dimensions=None):
    """Optionally wrap tensor variable into a dict with taps=[-1]"""
    state = initial_state(layer, dimensions)
    if state is not None:
        return dict(initial=state, taps=[-1])
    else:
        return None

class Model:
    """
    Simple predictive model for forecasting words from
    sequence using LSTMs. Choose how many LSTMs to stack
    what size their memory should be, and how many
    words can be predicted.
    """
    def __init__(self, hidden_size, input_size, output_size, rho, stack_size=1,  celltype=LSTM):
        self.rho = rho

        # declare model
        self.model = StackedCells(input_size, celltype=celltype, layers=[hidden_size] * stack_size)

        # self.model.layers.insert(0, Embedding(vocab_size, input_size))
        # add a classifier:
        self.model.layers.append(Layer(hidden_size, output_size, activation=lambda x: x))

        # inputs are matrices of indices,
        # each row is a sentence, each column a timestep
        self.input_mat = T.fmatrix("input_mat")

        # timestep for theano loop
        self.timestep = 1

        # set target value of prediction
        self.target_vec = T.fvector("target_vet")

        # create symbolic variables for prediction:
        self.predictions = self.create_prediction()

        # create gradient training functions:
        self.cost = self.create_cost_fun()
        self.pred_fun = self.create_predict_function()
        self.update_fun = self.create_training_function()

    @property
    def params(self):
        return self.model.params

    def initialize(self):
        self.model.initialize()

    def create_prediction(self):
        def step(idx, *states):
            # print "model input dimension: ", idx.ndim
            # print "hidden state: ", states
            # print "hidden state dimension: ", states[-1].ndim

            new_states = self.model.forward(idx, states)

            # print "result of step function:", [new_states[-1]] + new_states[:-1]
            return [new_states[-1]] + new_states[:-1]

        num_examples = self.input_mat.shape[0]
        result, updates = theano.scan(fn=step,
                                      n_steps=self.timestep,
                                      outputs_info=[dict(initial=self.input_mat, taps=[-1])] +
                                                   [initial_state_with_taps(layer, num_examples)
                                                    for layer in self.model.layers])
        return result[0][0][:]

    def create_cost_fun(self):
        return (self.predictions.T - self.target_vec).norm(L=2)

    def create_predict_function(self):
        return theano.function(
            inputs=[self.input_mat],
            outputs=self.predictions,
            allow_input_downcast=True
        )

    def create_training_function(self):
        updates, _, _, _, _ = create_optimization_updates(self.cost, self.params, rho=self.rho, method="adadelta")
        return theano.function(
            inputs=[self.input_mat, self.target_vec],
            outputs=self.cost,
            updates=updates,
            allow_input_downcast=True)

    def update(self, input_x, output_y):
        self.update_fun(input_x, output_y)

    def __call__(self, x):
        return self.pred_fun(x)

    def predict(self, x_mat):
        # return np.array([self.pred_fun(x) for x in x_mat])
        return self.pred_fun(x_mat)[:, 0]

def build_movement_lstm_model(stack, rho):
    return [Model(
        input_size=md.history_size,
        hidden_size=md.history_size,
        output_size=1,
        rho=rho,
        stack_size=stack, # make this bigger, but makes compilation slow
        celltype=LSTM # use RNN or LSTM
        ) for i in range(0, 3)]


def train_lstm(stack, rho, epoch, slice, x_train, y_train, x_test, y_test, data_file_path, file_attribute):

    # construct model & theano functions:
    models = build_movement_lstm_model(stack, rho)

    for axis_index in range(0, 3):
        x_train_data = x_train[axis_index]
        y_train_data = y_train[axis_index]

        start_time = datetime.datetime.now()

        # minibatch training:
        for i in range(1, epoch):

            for slice_idx in range(0, len(y_train_data)):

                error = models[axis_index].update_fun(x_train_data[slice_idx: slice_idx+slice, :],
                                                      y_train_data[slice_idx: slice_idx+slice])

            print "Epoch: ", i, " error: ", error

        end_time = datetime.datetime.now()
        print "Epoch time: ", (end_time-start_time).total_seconds(), "seconds"

        prediction = utils.predict_multi_step_ahead(models[axis_index], x_test[axis_index], md.delay_step)
        rmse, mae, mape = utils.calculate_error_measure(prediction, y_test[axis_index])

        file_name, file_dir, result_file = utils.make_result_file_dir(lstm, axis_index, rmse,
                                                                      data_file_path, file_attribute,
                                                                      stack, slice, epoch)

        utils.save_to_error_trunk(lstm, axis_index, file_name, rmse, mae, mape)
        utils.show_measure(lstm, axis_index, file_name)
        utils.save_result(lstm, axis_index, prediction, y_test[axis_index], file_name, file_dir, result_file,
                          utils.make_metadata(lstm, stack, slice, epoch))

    ##################################################################
    # training LSTM for all the data and test model to each data file#
    ##################################################################

    # print "=================training==================="
    #
    # train_data = {}
    # test_data = {}
    # models = build_movement_lstm_model()
    #
    # for file_path in md.get_file_list():
    #     x_train, y_train, x_test, y_test = md.load_data_set(file_path, False)
    #     train_data[file_path] = (x_train, y_train)
    #     test_data[file_path] = (x_test, y_test)
    #     break
    #
    # for file_key in train_data:
    #     x_train_data, y_train_data = train_data[file_key]
    #     x_test_data, y_test_data = test_data[file_key]
    #
    #     for axis_index in range(0, 3):
    #         # train:
    #         start_time = datetime.datetime.now()
    #
    #         # minibatch training:
    #         for i in range(1, training_times):
    #             for slice_idx in range(0, len(y_train_data[axis_index])):
    #
    #                 error = models[axis_index].update_fun(x_train_data[axis_index][slice_idx: slice_idx+slice_step, :],
    #                                                       y_train_data[axis_index][slice_idx: slice_idx+slice_step])
    #
    #             # for ele in models[axis_index].params:
    #             #     if 1 in T.isnan(ele).eval():
    #             #         print "NaN!!", i
    #
    #         end_time = datetime.datetime.now()
    #
    #         print "Training for ", (end_time-start_time).total_seconds(), "seconds"
    #
    #         prediction = utils.predict_multi_step_ahead(models[axis_index], x_test_data[axis_index], md.delay_step)
    #         rmse, mae, mape = utils.calculate_error_measure(prediction, y_test_data[axis_index])
    #
    #         file_name, file_dir = utils.make_result_file_dir(lstm, axis_index, mape, file_key,
    #                                                          stack_layer, slice_step, training_times)
    #
    #         utils.save_to_error_trunk(file_name, rmse, mae, mape)
    #
    #         # print [ele.eval() for ele in models[axis_index].params]
    #
    #         utils.show_measure(file_name)
    #         utils.save_result(prediction, y_test_data[axis_index], file_name, file_dir,
    #         utils.make_metadata(lstm, stack_layer, slice_step, training_times))
    #
    # utils.show_final_error_result()
