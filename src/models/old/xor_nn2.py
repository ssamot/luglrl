import keras
import matplotlib
import numpy as np
from keras.layers import Flatten, Input, concatenate, \
    Lambda, Reshape, add
from keras.models import Model
from keras import backend as K
from tqdm import trange
from sklearn.linear_model import LassoLarsCV

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import InverseEucledian


def create_nn(input_shape, num_forms=2):
    print(input_shape)

    formx_x = []
    forms = []



    input = Input(shape=input_shape)

    for _ in range(num_forms):
        h_object = InverseEucledian()
        x = h_object(input)
        #x = keras.layers.BatchNormalization()(x)
        formx_x.append(x)
        forms.append(h_object)

    distance = add(formx_x)

    distanceModel = Model(inputs=[input],
                          outputs=[distance])
    #x = keras.layers.concatenate([distance])
    x = keras.layers.Dense(1)(distance)
    #x = keras.layers.Dense(1)(distance)

    model = Model(inputs=[input], outputs=[x])

    model.compile(loss=keras.losses.mean_squared_error,
                  #optimizer=keras.optimizers.SGD(lr=0.01, momentum =0.9, nesterov = True),
                  optimizer=keras.optimizers.Adam(lr=0.001),
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

    num_forms = 3
    model, distanceModel, forms = create_nn(2, num_forms=num_forms)
    print(x.shape, y.shape)
    epochs = 10000

    model.fit([x], y, epochs=30000, verbose=False)

    # with trange(epochs) as t:
    #     x_input = x  # + n_strength * np.random.normal(size=x.shape)
    #     for i in t:
    #         r = model.train_on_batch(x=[x_input],
    #                                  y=y)
    #         t.set_description(
    #             'Epoch %i, loss_dist %.3f' % (i, r[-1],))



    ones = np.ones(shape=x.shape)
    y_hat = model.predict([x])
    print(y_hat)
    for form in forms:
        print(np.array(form.get_example()))
