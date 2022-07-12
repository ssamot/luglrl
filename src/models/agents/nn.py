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

