import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.datasets import mnist, cifar100, cifar10

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

matplotlib.use("Agg")



def load_minst():
    num_classes = 10
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # x_train = x_train.mean(-1)
    # x_test = x_test.mean(-1)

    print(x_train.shape)
    img_rows, img_cols = x_train.shape[1], x_train.shape[2]  # , x_train.shape[3]
    channels = 1

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)




    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], x_train.min(), 'train samples')
    print(x_test.shape[0], x_test.min(), 'test samples')



    return x_train.reshape([-1,28*28]), y_train, x_test.reshape([-1,28*28]), y_test, img_rows, img_cols


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, img_rows, img_cols = load_minst()
    print(x_train.shape)


    # ds = np.linalg.norm(x - np.array([1,0]), axis = -1)
    # ds = ds[:,np.newaxis]
    #
    # one = np.ones(shape=x.shape)
    # n_strength = 0.01
    # x = np.concatenate([x,ds], axis = -1)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train,y_train)

    y_test_hat = clf.predict(x_test)
    acc = accuracy_score(y_test,y_test_hat )
    print("acc", acc)


    # print(y_hat)
    # for form in forms:
    #     print(np.array(form.get_example()))
