import matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

matplotlib.use("Agg")

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

    from scipy.spatial import distance
    print(x.shape, y.shape)
    # model.fit([x] + [ones]*num_forms, y, epochs = 10000)
    epochs = 5000
    # ds = []
    # for xi in x:
    #     ds.append(distance.euclidean(xi, [1,0]))
    # ds = np.array(ds)[:,np.newaxis]


    ds = np.linalg.norm(x - np.array([1,0]), axis = -1)
    ds = ds[:,np.newaxis]

    one = np.ones(shape=x.shape)
    n_strength = 0.01
    x = np.concatenate([x,ds], axis = -1)
    clf = LinearRegression()
    clf.fit(x,y)
    print(clf.coef_)
    print(clf.predict(x))

    # print(y_hat)
    # for form in forms:
    #     print(np.array(form.get_example()))
