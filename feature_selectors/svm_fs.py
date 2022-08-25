import numpy as np
from sklearn.svm import SVC


def svm_fs(X, y, svm_max_iter=10_000_000, kernel='linear'):
    # our implementation of RFE-SVM score function
    X = np.array(X)
    y = np.array(y)

    X_0 = X.copy()
    s = list(range(X.shape[1]))
    r = []

    while s:
        svm = SVC(kernel=kernel, max_iter=svm_max_iter)
        svm.fit(X_0[:, s], y)

        alphas = np.zeros(len(X))
        alphas[svm.support_] = svm.dual_coef_.mean(axis=0)
        w = alphas @ X_0[:, s]
        c = w ** 2

        f = np.argmin(c)
        r.append(s[f])
        s.remove(s[f])

    r = np.array(r)[::-1]

    # make scores
    t = np.array([x[0] for x in sorted(enumerate(r), key=lambda x: x[1])])
    return 1 - t / max(t)


def rbf_svm_fs(X, y, svm_max_iter=10_000_000):
    # same rfe svm logic, but with rbf kernel
    return svm_fs(X, y, svm_max_iter=svm_max_iter, kernel='rbf')


def poly_svm_fs(X, y, svm_max_iter=10_000_000):
    # same rfe svm logic, but with poly kernel
    return svm_fs(X, y, svm_max_iter=svm_max_iter, kernel='poly')


def svm_fs_New(X, y, svm_max_iter=10_000_000, kernel='linear', step_frac=0.1):
    # our implementation of RFE-SVM score function, with our modifications
    X = np.array(X)
    y = np.array(y)

    X_0 = X.copy()
    s = list(range(X.shape[1]))
    r = []

    while s:
        svm = SVC(kernel=kernel, max_iter=svm_max_iter)
        svm.fit(X_0[:, s], y)

        alphas = np.zeros(len(X))
        alphas[svm.support_] = svm.dual_coef_.mean(axis=0)
        w = alphas @ X_0[:, s]
        c = w ** 2

        # main modification - remove 10% of the features each iteration compared to 1 feature at the original RFE-SVM
        f = np.argsort(c)[:1 + int(len(c) * step_frac)]
        for f_i in f:
            r.append(s[f_i])
        s = [x for idx, x in enumerate(s) if idx not in f]

    r = np.array(r)[::-1]

    # make scores
    t = np.array([x[0] for x in sorted(enumerate(r), key=lambda x: x[1])])
    return 1 - t / max(t)


def rbf_svm_fs_New(X, y, svm_max_iter=10_000_000):
    # same rfe svm logic with our modifications, but with rbf kernel
    return svm_fs_New(X, y, svm_max_iter=svm_max_iter, kernel='rbf')


def poly_svm_fs_New(X, y, svm_max_iter=10_000_000):
    # same rfe svm logic with our modifications, but with poly kernel
    return svm_fs_New(X, y, svm_max_iter=svm_max_iter, kernel='poly')
