from sklearn.svm import SVC
import numpy as np
from config import *
from data_utils.timer import Timer


class SVM:
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None):

        self.svm = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma,
                       coef0=coef0, shrinking=shrinking, probability=probability,
                       tol=tol, cache_size=cache_size, class_weight=class_weight,
                       verbose=verbose, max_iter=max_iter,
                       decision_function_shape=decision_function_shape,
                       random_state=random_state)

    def train(self, x, y):
        self.svm.fit(x, y)

    def score(self, xt, yt):
        return self.svm.score(xt, yt)

    def predict(self, sample):
        return self.svm.predict(sample) == 1

    def predict_prob(self, x):
        return self.predict_prob(x)


if __name__ == '__main__':
    rbf_svm = SVM(kernel='rbf', gamma='scale', probability=True)
    train_data = np.load(DATA_PATH + "\\hog\\train.npz")
    x, y = train_data['feature'], train_data['label']
    print("number of training sample = %d" % x.shape[0])
    timer = Timer()
    timer.start()
    rbf_svm.train(x, y)
    timer.stop()

    test_data = np.load(DATA_PATH + "\\hog\\test.npz")
    xt, yt = test_data['feature'], test_data['label']
    acc = rbf_svm.score(xt, yt)
    print("mean acc = {:.4f}".format(acc))

    sample = xt[0].reshape(1, xt.shape[1])
    proba = rbf_svm.predict_proba(sample)

    print("proba is ", proba)