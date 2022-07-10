import keras
from keras import layers
import keras.backend as K
import numpy as np
from tqdm import trange
import shutil
from pathlib import Path
from scipy.spatial import distance
from sklearn.metrics import mean_absolute_error
import tensorflow as tf

np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)


def get_model(n_inputs, encoding_dim=4):
    decoder_layers = [
        layers.Dense(1024, activation='leaky_relu'),
        layers.BatchNormalization(),
        layers.Dense(n_inputs, activation='sigmoid'),
    ]

    def get_decoded(decoded):
        for layer in decoder_layers:
            decoded = layer(decoded)
        return decoded

    input_img = keras.Input(shape=(n_inputs,))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(32, activation='leaky_relu')(input_img)
    for _ in range(1):
        encoded = layers.Dense(32, activation='leaky_relu')(encoded)

    encoded = layers.Dense(encoding_dim, activation="sigmoid")(encoded)


    decoded = get_decoded(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)
    encoder = keras.Model(input_img, encoded, name="encoder")

    encoded_input = keras.Input(shape=(encoding_dim,))

    decoder = keras.Model(encoded_input, get_decoded(encoded_input),
                          name="decoder")

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder, encoder, decoder


def save_model(model, model_name):
    model.save(f"{project_dir}/models/{model_name}_tmp.keras")
    shutil.copy(f"{project_dir}/models/{model_name}_tmp.keras",
                f"{project_dir}/models/{model_name}.keras")
    shutil.os.remove(f"{project_dir}/models/{model_name}_tmp.keras")


def get_model_with_comparison(n_inputs, encoder, decoder):
    def euclidean_distance(vects):
        x, y = vects
        sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1,
                                        keepdims=True)
        return tf.math.sqrt(
            tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

    def manhattan_distance(vects):
        x, y = vects
        sum_square = tf.math.reduce_sum(tf.math.abs(x - y), axis=1,
                                        keepdims=True)
        return sum_square



    input_left = keras.Input(shape=(n_inputs,), name="input_left")
    input_right = keras.Input(shape=(n_inputs,), name="input_right")

    encoded_input_left = encoder(input_left)
    encoded_input_right = encoder(input_right)


    merge_layer = layers.Lambda(euclidean_distance)(
        [encoded_input_left, encoded_input_right])



    #optimizer = keras.optimizers.SGD(lr = 0.0001, momentum = 0.5, nesterov = True)

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model = keras.Model([input_left, input_right],
                        [decoder(encoded_input_left), merge_layer])
    model.compile(optimizer=optimizer, loss=["binary_crossentropy", 'mae'],
                  loss_weights=[1, 1], jit_compile=False)

    return model


if __name__ == '__main__':
    n_inputs = 6
    epochs = 50000
    encoding_dim = 2
    autoencoder, encoder, decoder = get_model(n_inputs,
                                              encoding_dim=encoding_dim)
    model = get_model_with_comparison(n_inputs, encoder, decoder)
    print(model.summary())

    def generate_batch():
        import graycode
        codes = graycode.gen_gray_codes(6)
        ps = []

        for i, code in enumerate(codes):
            code = graycode.tc_to_gray_code(i)
            c = '{:06b}'.format(graycode.tc_to_gray_code(i))
            p = np.fromstring(",".join(c), count=6, dtype="int", sep=",")
            # print(c, '{:06b}'.format(graycode.tc_to_gray_code(i)), )
            ps.append(p)

        left = []
        right = []
        ds = []
        # for j in range(0, len(ps)):
        #     l = ps[j]
        #     r = ps[0]
        #     d = distance.hamming(l, r)
        #     print(d, l, r)
        #     # exit()
        #
        #     left.append(l)
        #     right.append(r)
        #     ds.append(d)


        for i in range(len(ps)):
            for j in range(len(ps)):
                l = ps[i]
                r = ps[j]
                d = distance.hamming(l, r)
                left.append(l)
                right.append(r)
                ds.append(d)

        left = np.array(left)#[:1000]
        right = np.array(right)#[:1000]
        d = np.array(ds)#[:1000]
        print(left.shape,right.shape,d.shape)
        return left,right,d


    project_dir = Path(__file__).resolve().parents[2]

    input_left, input_right, d = generate_batch()
    with trange(epochs) as t:


        for i in t:
            r = model.train_on_batch(x=[input_left, input_right],
                                     y=[input_left, d])

            #hat = model.predict([input_left, input_right])[1]
            #print(hat, mean_absolute_error(d,hat), r)
            t.set_description(
                'Epoch %i, loss_auto %.3f, loss_dist %.3f' % (i, r[1], r[2]))

            if (i % 1000 == 0):
                save_model(model, "autoencoder_test")
                save_model(decoder, "decoder_test")
                save_model(encoder, "encoder_test")
                print(d[:32])
                #print(input_left[:32], input_right[:32])
                d_hat = (model.predict([input_left[:32], input_right[:32]]))[1].T[0]
                print(d_hat)
                print(mean_absolute_error(d[:32],d_hat ))
