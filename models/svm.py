from sklearn.svm import SVC
import numpy as np
from config import *
from data_utils.timer import Timer


class SVM:
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, shrinking=True, probability=True,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None):

        self.name = "svm_" + kernel
        self.svm = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma,
                       coef0=coef0, shrinking=shrinking, probability=probability,
                       tol=tol, cache_size=cache_size, class_weight=class_weight,
                       verbose=verbose, max_iter=max_iter,
                       decision_function_shape=decision_function_shape,
                       random_state=random_state)

    def train(self, x, y, save=False):
        self.svm.fit(x, y)
        if save:
            self.save()

    def get_support_vectors(self):
        return self.svm.support_vectors_

    def score(self, xt, yt):
        return self.svm.score(xt, yt)

    def predict(self, sample):
        if len(sample.shape) == 1:
            sample = sample.reshape(1, sample.shape[0])
        return self.svm.predict(sample) == 1

    def predict_proba(self, sample):
        if len(sample.shape) == 1:
            sample = sample.reshape(1, sample.shape[0])
        pred = self.svm.predict_proba(sample)
        return pred[0][1]

    def save(self, save_dir="..\\checkpoints"):
        save_path = os.path.join(save_dir, self.name + ".pkl")
        with open(save_path, 'wb') as fd:
            pickle.dump(self, fd)

    @staticmethod
    def load(file="..\\checkpoints\\svm_rbf.pkl"):
        with open(file, 'rb') as fd:
            instance = pickle.load(fd)
        return instance


if __name__ == '__main__':
    svm = SVM(kernel='linear', gamma='scale')
    train_data = np.load(DATA_PATH + "\\hog\\train.npz")
    x, y = train_data['feature'], train_data['label']
    print("number of training sample = %d" % x.shape[0])
    timer = Timer()
    timer.start()
    svm.train(x, y, save=True)
    timer.stop()

    test_data = np.load(DATA_PATH + "\\hog\\test.npz")
    xt, yt = test_data['feature'], test_data['label']
    acc = svm.score(xt, yt)
    print("mean acc = {:.4f}".format(acc))

    # axises = [89, 450, 678]
    # for axis in axises:
    #     sup_vec = svm.get_support_vectors()
    #     import matplotlib.pyplot as plt
    #     plt.hist(sup_vec[:, axis], bins=40, facecolor='blue', edgecolor='black')
    #     plt.title("histogram of %s support vector" % svm.name)
    #     plt.savefig(os.path.join(WORKSPACE, "images", "sup_vec_%s_axis%d.png" % (svm.name, axis)))
