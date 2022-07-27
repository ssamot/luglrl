
import tensorflow as tf
from keras import backend as K
import keras
import numpy as np


def custom_reg(weight_matrix):
    # we like our weights large!
    clipped_matrix = tf.clip_by_value(weight_matrix,-10,10)
    return -0.01*tf.reduce_sum(clipped_matrix*clipped_matrix)


class Hadamard(keras.layers.Layer):



    def __init__(self, units=1):
        super(Hadamard, self).__init__()
        self.units = units

    def build(self, input_shape):
        init = tf.constant_initializer(np.random.randint(2, size = input_shape[1:]))
        self.w = self.add_weight(
            shape=input_shape[1:],
            initializer="glorot_uniform",
            trainable=True,
            regularizer=custom_reg
        )

    def call(self, inputs):
        k = tf.multiply(inputs, ((self.w)))
        return k

    def get_example(self):
        print(self.w)
        return ((self.w))


def neg_mean_euc_dist(vects):
    x,y = vects
    d = tf.norm(x-y,keepdims=True,axis = -1) + K.epsilon()
    return -d



def neg_manhattan_dist(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.abs(x - y), axis=1, keepdims=True)

    return -sum_square

def cos_similiary(vects):
    x, y = vects
    y_true = tf.math.l2_normalize(x, axis=-1)
    y_pred = tf.math.l2_normalize(y, axis=-1)
    return -tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)

def euclidean_distance_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def pad_up_to(t, max_in_dims, constant_values):
    s = tf.shape(t)
    paddings = [[0, m-s[i]] for (i,m) in enumerate(max_in_dims)]
    return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)