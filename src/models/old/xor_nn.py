import keras
import matplotlib
import numpy as np
from keras.layers import Flatten, Input, concatenate, \
    Lambda, Reshape
from keras.models import Model
from keras import backend as K
from tqdm import trange
from sklearn.linear_model import LassoLarsCV

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import Hadamard, \
    euclidean_distance_output_shape, \
    neg_mean_euc_dist


def create_nn(input_shape, num_forms=2):
    print(input_shape)

    all_inputs = []
    formx_x = []
    forms = []

    for _ in range(num_forms):
        input = Input(shape=input_shape)
        h_object = Hadamard()
        x = h_object(input)
        #x = keras.layers.BatchNormalization()(x)
        formx_x.append(x)
        all_inputs.append(input)
        forms.append(h_object)

    input = Input(shape=input_shape)

    distance = concatenate([Lambda(neg_mean_euc_dist,
                                   output_shape=euclidean_distance_output_shape)(
        [c, input])
        for c in formx_x])

    distanceModel = Model(inputs=[input] + all_inputs,
                          outputs=[distance])
    x = keras.layers.concatenate([distance])
    x = keras.layers.Dense(1)(x)
    #x = keras.layers.Dense(1)(distance)

    model = Model(inputs=[input] + all_inputs, outputs=[x])

    model.compile(loss=keras.losses.mean_squared_error,
                  #optimizer=keras.optimizers.SGD(lr=0.01, momentum =0.9, nesterov = True),
                  optimizer=keras.optimizers.Adam(lr=0.0001),

                  metrics=['mse'])

    return model, distanceModel, forms


if __name__ == '__main__':
    x = np.array([[1, 1],
                  [0, 1],
                  [1, 0],
                  [0, 0]
                  ])

    y = np.array([
        [0],
        [1],
        [1],
        [0],
    ])

    num_forms = 1
    model, distanceModel, forms = create_nn(2, num_forms=num_forms)
    print(x.shape, y.shape)
    # model.fit([x] + [ones]*num_forms, y, epochs = 10000)
    epochs = 50000
    one = np.ones(shape=x.shape)
    #n_strength = 0.01

    # reps = 10000
    # x_rep = np.repeat(x, reps, axis=0)
    # y_rep = np.repeat(y, reps, axis=0)
    # ones_rep = np.ones(shape=x_rep.shape)
    # one_inputs = [ones_rep for _ in range(num_forms)]
    # print(x_rep.shape, y_rep.shape)
    # model.fit(x=[x_rep] + one_inputs, y=y_rep, epochs=1, validation_split=0.1)
    # # exit()

    with trange(epochs) as t:
        x_input = x  # + n_strength * np.random.normal(size=x.shape)
        ones = [one for _ in range(num_forms)]
        for i in t:
            r = model.train_on_batch(x=[x_input] + ones,
                                     y=y)
            t.set_description(
                'Epoch %i, loss_dist %.3f' % (i, r[-1],))

    ones = np.ones(shape=x.shape)
    y_hat = model.predict([x] + [ones] * num_forms)
    print(y_hat)
    for form in forms:
        print(np.array(form.get_example()))
