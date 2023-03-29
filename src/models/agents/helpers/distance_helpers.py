import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

class NormalDistributionRegularizer(regularizers.Regularizer):
    def __init__(self, l2=0.01, n_samples=1000):
        self.l2 = l2
        self.n_samples = n_samples

    def __call__(self, x):
        # Sample the feature values
        #samples = tf.random.shuffle(x)[:self.n_samples]

        # Calculate the empirical CDF
        sorted_samples = tf.sort(x)
        cdf = tf.range(1, self.n_samples + 1, dtype=tf.float32) / self.n_samples

        # Calculate the KS statistic
        ks_statistic = K.max(K.abs(cdf - tf.math.erf(sorted_samples / tf.sqrt(2))))

        # Calculate the regularizer term
        regularizer = self.l2 * K.square(ks_statistic)

        return regularizer


import keras.backend as K
from keras.layers import Layer


class GaussianRegularizer(Layer):
    def __init__(self, sigma=1.0, **kwargs):
        super(GaussianRegularizer, self).__init__(**kwargs)
        self.sigma = sigma

    def call(self, inputs):
        # Compute the mean and variance of the activation distribution
        mean = K.mean(inputs, axis=0, keepdims=True)
        var = K.var(inputs, axis=0, keepdims=True)

        # Compute the MMD between the activation distribution and a Gaussian distribution
        mmd = K.square(mean) + var - 2 * K.sqrt(var) * K.exp(
            -K.square(mean) / (2 * K.square(self.sigma)))
        mmd = K.mean(mmd)

        # Add the MMD regularizer to the layer's loss
        self.add_loss(mmd)

        # Return the inputs as the layer's output
        return inputs

    def get_config(self):
        config = {'sigma': self.sigma}
        base_config = super(GaussianRegularizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


import keras.backend as K
from keras.layers import Layer


class GaussianRegularizer(Layer):
    def __init__(self, sigma=1.0, **kwargs):
        super(GaussianRegularizer, self).__init__(**kwargs)
        self.sigma = sigma

    def call(self, inputs):
        # Compute the mean and covariance of the activation distribution
        mean = K.mean(inputs, axis=0, keepdims=True)
        cov = K.cov(inputs, rowvar=False, bias=True)

        # Add diagonal penalty to the covariance matrix
        cov = cov + K.eye(cov.shape[-1]) * 1e-8

        # Compute the log-likelihood of the activation distribution under a Gaussian model
        n = K.cast(K.shape(inputs)[0], K.floatx())
        logdet = K.log(K.abs(K.det(cov)))
        logp = -0.5 * (
                    n * K.log(2 * K.constant(K.epsilon())) + n * logdet + K.sum(
                K.dot(inputs - mean, K.linalg.inv(cov)) * (inputs - mean),
                axis=1))
        logp = K.mean(logp)

        # Add the log-likelihood regularizer to the layer's loss
        self.add_loss(logp)

        # Return the inputs as the layer's output
        return inputs

    def get_config(self):
        config = {'sigma': self.sigma}
        base_config = super(GaussianRegularizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


import keras.backend as K
from keras.layers import Layer

class MMDRegularizer(Layer):
    def __init__(self, sigma=1.0, **kwargs):
        super(MMDRegularizer, self).__init__(**kwargs)
        self.sigma = sigma

    def call(self, inputs):
        # Compute the mean of the activation distribution
        mean = K.mean(inputs, axis=0, keepdims=True)

        # Compute the squared distances between the activations and the mean
        dists = K.sum(K.square(inputs - mean), axis=1, keepdims=True)

        # Compute the MMD between the activation distribution and a Gaussian distribution
        n = K.cast(K.shape(inputs)[0], K.floatx())
        mmd = (1 / (n * (n - 1))) * K.sum(K.exp(-0.5 * dists / self.sigma**2)) - (2 / n) * K.sum(K.exp(-0.5 * K.square(K.sqrt(2 / self.sigma) * K.abs(K.expand_dims(mean, axis=0) - inputs))))
        mmd = K.maximum(mmd, 0)

        # Add the MMD regularizer to the layer's loss
        self.add_loss(mmd)

        # Return the inputs as the layer's output
        return inputs

    def get_config(self):
        config = {'sigma': self.sigma}
        base_config = super(MMDRegularizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def mmd_regularizer(sigma=1.0, weight=1.0):
    def mmd(P):
        n_p = K.shape(P)[0]
        sum_p = K.sum(K.square(P), axis=1, keepdims=True)
        dot_prod = K.dot(P, K.transpose(P))
        dist = K.expand_dims(sum_p, axis=1) - 2.0 * dot_prod + K.expand_dims(sum_p, axis=0)
        kernel = K.exp(-dist / (2.0 * sigma ** 2))
        mmd = K.mean(kernel) - 2.0 / n_p * K.mean(K.exp(-K.square(P) / (2.0 * sigma ** 2)))
        return mmd * weight

    return mmd



import tensorflow as tf

def mmd(x, kernel_fn):
    n = tf.shape(x)[0]
    kxx = kernel_fn(x, x)
    kxy = kernel_fn(x, tf.zeros_like(x))
    kyy = kernel_fn(tf.zeros_like(x), tf.zeros_like(x))
    return tf.reduce_mean(kxx) - 2 * tf.reduce_mean(kxy) + tf.reduce_mean(kyy)

def gaussian_kernel(x1, x2, sigma=1.0):
    sq_norms1 = tf.reduce_sum(tf.square(x1), axis=1, keepdims=True)
    sq_norms2 = tf.reduce_sum(tf.square(x2), axis=1, keepdims=True)
    dot_prods = tf.matmul(x1, tf.transpose(x2))
    dists_sq = sq_norms1 + tf.transpose(sq_norms2) - 2 * dot_prods
    return tf.exp(-dists_sq / (2 * sigma ** 2))

def linear_kernel(x1, x2):
    return tf.matmul(x1, tf.transpose(x2))
