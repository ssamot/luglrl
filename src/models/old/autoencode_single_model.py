import keras
from keras import layers
import keras.backend as K
import numpy as np
from tqdm import trange
import shutil
from pathlib import Path
from scipy.spatial import distance as ed
from sklearn.metrics import mean_absolute_error
import tensorflow_probability as tfp
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)


def get_model(n_inputs, encoding_dim=4):


    # This is the size of our encoded representations
    # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # This is our input image
    input_img = keras.Input(shape=(n_inputs,))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(1024, activation='elu')(input_img)
    # for _ in range(10):
    #     encoded = layers.Dense(32, activation='leaky_relu')(encoded)

    encoded = layers.Dense(encoding_dim, activation = "linear")(encoded)
    encoded = layers.Activation("sigmoid")(encoded)


    encoder = keras.Model(input_img, encoded, name="encoder")



    return encoder


def save_model(model, model_name):
    model.save(f"{project_dir}/models/{model_name}_tmp.keras")
    shutil.copy(f"{project_dir}/models/{model_name}_tmp.keras",
                f"{project_dir}/models/{model_name}.keras")
    shutil.os.remove(f"{project_dir}/models/{model_name}_tmp.keras")


def get_model_with_comparison(n_inputs, encoder):
    def euclidean_distance(vects):

        x, y = vects
        sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1,
                                        keepdims=True)
        return tf.math.sqrt(
            tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

    # def cosine_distance(left,right):
    #     cosine_loss = tf.keras.losses.CosineSimilarity(axis=1,
    #                                      reduction=tf.keras.losses.Reduction.NONE)
    #     return -cosine_loss(left, right)


    input_left = keras.Input(shape=(n_inputs,), name="input_left")
    input_right = keras.Input(shape=(n_inputs,), name="input_right")
    #b_input_left = layers.BatchNormalization()(input_left)
    #b_input_right = layers.BatchNormalization()(input_right)

    encoded_input_left = encoder(input_left)
    encoded_input_right = encoder(input_right)

    # diff = keras.layers.subtract([encoded_input_left, encoded_input_right])
    # d = keras.layers.multiply([diff,diff])
    # d = K.sqrt(K.sum(d,axis = -1))

    merge_layer = layers.Lambda(euclidean_distance)([encoded_input_left, encoded_input_right])
    final = layers.Dense(encoding_dim)(merge_layer)
    normal_layer = layers.BatchNormalization()(merge_layer)

    #d = euclidean_distance(encoded_input_left, encoded_input_right)

    #d = keras.layers.dot([encoded_input_left + 1, encoded_input_right + 1], axes = -1, normalize=True)

    #d = encoded_input_left * encoded_input_right

    model = keras.Model([input_left, input_right], normal_layer)
    #optimizer = keras.optimizers.SGD(lr = 0.01, momentum = 0.5, nesterov = True)


    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mae", jit_compile=False)
    return model


if __name__ == '__main__':
    n_inputs = 6
    epochs = 50000
    encoding_dim = 2
    encoder = get_model(n_inputs,encoding_dim=encoding_dim)
    model = get_model_with_comparison(n_inputs, encoder)
    print(model.summary())

    def generate_batch():
        import graycode
        codes = graycode.gen_gray_codes(n_inputs)
        ps = []

        for i, code in enumerate(codes):
            code = graycode.tc_to_gray_code(i)
            c = '{:06b}'.format(graycode.tc_to_gray_code(i))
            p = np.fromstring(",".join(c), count=n_inputs, dtype="int", sep=",")
            # print(c, '{:06b}'.format(graycode.tc_to_gray_code(i)), )
            ps.append(p)

        left = []
        right = []
        ds = []
        from scipy import spatial
        #for i in range(len(ps)):
        for j in range(0, len(ps)):
            l = ps[j]
            r = ps[0]
            d = spatial.distance.hamming(l, r)
            print(d, l, r)
            #exit()

            left.append(l)
            right.append(r)
            ds.append(d)
        left = np.array(left)#[:200]
        right = np.array(right)#[:200]
        d = np.array(ds)#[:200]
        d = d[:,np.newaxis]
        #scaler = MinMaxScaler(feature_range=(-1,1))
        #scaler = StandardScaler()
        #scaler.fit(np.concatenate([left,right]))
        #left = scaler.transform(left)
        #right = scaler.transform(right)
        #d = StandardScaler().fit_transform(d)

        print(left.shape,right.shape,d.shape)
        return left,right,d

    # def generate_batch(n_inputs, batch_size=128):
    #     r = np.random.random()
    #     input_left = np.random.randint(2, size=(batch_size, n_inputs))
    #     mask = np.random.binomial(1, r, size=input_left.shape)
    #     input_right = input_left * mask
    #     # d = np.sum((input_left != input_right), axis = -1)
    #     # d = ed.euclidean(input_left, input_right)
    #     diff = (input_left - input_right) ** 2
    #     # print(diff)codes
    #     s = np.sum(diff, axis=1)
    #     # print(s)
    #     # d = np.sqrt()
    #     d = s / n_inputs
    #     # print(input_left, input_right)
    #     # print(d)
    #     # exit()
    #     return input_left, input_right, d[:, np.newaxis]
    #
    #
    # def generate_diverse_batch(n_inputs, batch_size, size):
    #     input_left = []
    #     input_right = []
    #     d = []
    #     for _ in range(size):
    #         il, ir, id = generate_batch(n_inputs, batch_size)
    #         input_left.append(il)
    #         input_right.append(ir)
    #         d.append(id)
    #
    #     return np.concatenate(input_left), np.concatenate(
    #         input_right), np.concatenate(d)


    project_dir = Path(__file__).resolve().parents[2]

    input_left, input_right, d = generate_batch()
    with trange(epochs) as t:


        for i in t:
            r = model.train_on_batch(x=[input_left, input_right],
                                   y=[d])
            # r = model.fit(x=[input_left, input_right],
            #                          y=[d], verbose=True, epochs = 100000)
            # #hat = model.predict([input_left, input_right], verbose = False)
            #print(hat)
            #exit()
            #print(hat, , r)
            t.set_description(
                'Epoch %i, loss_dist %.3f' % (i, r,))

            if (i % 1000 == 0):
                save_model(model, "autoencoder_test")
                save_model(encoder, "encoder_test")
                print(d[:32].T[0])
                #print(input_left[:32], input_right[:32])
                d_hat = (model.predict([input_left[:32], input_right[:32]])).T[0]
                print(d_hat)
                print(mean_absolute_error(d[:32],d_hat ))
