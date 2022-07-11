# Imports
from keras.models import Sequential
from keras.layers import Dense
import keras



def build_model(y_axis):
    input_shape = (y_axis,)
    model = Sequential()
    model.add(Dense(30*y_axis, input_shape=input_shape, activation='leaky_relu'))
    #model.add(Dense(50, activation='leaky_relu'))
    #model.add(Dense(50, activation='leaky_relu'))
    model.add(Dense(1, activation='linear'))
    #callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    model.compile(loss=keras.losses.mean_squared_error,
                  #optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9,nesterov=True),
                  optimizer=keras.optimizers.Adam(), jit_compile=False,

                  )
    return model


def build_nn_socratic(input_shape, num_forms=2):
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
    x = keras.layers.concatenate([distance, input])
    x = keras.layers.Dense(1)(x)
    #x = keras.layers.Dense(1)(distance)

    model = Model(inputs=[input] + all_inputs, outputs=[x])

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.SGD(lr=0.1, momentum =0.9, nesterov = True),
                  #optimizer=keras.optimizers.Adam(lr=0.1),

                  metrics=['mse'])

    return model, distanceModel, forms

