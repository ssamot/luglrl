import keras.backend as K
import numpy as np


class NNWeightHelper:
    def __init__(self, model):
        self.model = model
        self.init_weights = self.model.trainable_weights


    def _set_trainable_weight(self, model, weights):
        """Sets the weights of the model.

        # Arguments
            model: a keras neural network model
            weights: A list of Numpy arrays with shapes and types matching
                the output of `model.trainable_weights`.
        """
        tuples = []
        for layer in model.layers:
            num_param = len(layer.trainable_weights)
            layer_weights = weights[:num_param]
            for sw, w in zip(layer.trainable_weights, layer_weights):
                tuples.append((sw, w))
            weights = weights[num_param:]
        K.batch_set_value(tuples)


    def set_weights(self, weights):
        new_weights = []
        total_consumed = 0
        for w in self.init_weights:
            w = np.array(w)
            layer_shape, layer_size = w.shape, w.size
            chunk = weights[total_consumed:total_consumed + layer_size]
            total_consumed += layer_size
            new_weights.append(np.array(chunk.reshape(layer_shape)))

        self._set_trainable_weight(self.model, new_weights)


    def get_weights(self):
        W_list = self.model.trainable_weights
        #print(W_list, "weer")
        #exit()
        W_flattened_list = [np.array(k, copy =True).flatten() for k in W_list]
        W = np.concatenate(W_flattened_list)
        return W