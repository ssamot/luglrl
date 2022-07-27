import matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

matplotlib.use("Agg")

if __name__ == '__main__':
    X = np.array([[1, 1],
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
    # model.fit([x] + [ones]*num_forms, y, epochs = 10000)
    epochs = 5000
    r = np.array(list(range(0,len(X))))
    np.random.shuffle(r)
    X = X[r]
    y = y[r]

    maximum_distances = 10
    n_samples = 100

    best = [[] for _ in range(maximum_distances)]
    best_score = [[-10000] for _ in range(maximum_distances)]
    from scipy.spatial.distance import cdist

    for dist in range(1, maximum_distances):
        for _ in range(n_samples):
            n_distances = dist
            if(n_distances > len(X)):
                break
            sampled = np.random.choice(r, size = n_distances, replace=False)
            X_dst = cdist(X, X[sampled])
            clf = LinearRegression()
            clf.fit(X_dst, y)
            score = clf.score(X_dst,y)
            if(score > best_score[n_distances]):
                best_score[n_distances] = score
                best[n_distances] = sampled

    print(best[1:len(X)+1])
    print(best_score[1:len(X)+1])

    #print(clf.coef_)
    #print(clf.predict(X_dss))

