import numpy as np
from config import *


class LDA:
    def __init__(self, p):
        self.p = p
        self.beta = np.zeros((p, 1), dtype=np.float32)
        self.threshold = 0
        self.mu_pos = 0
        self.mu_neg = 0
        self.var_pos = 0
        self.var_neg = 0
        self.inter_var = 0
        self.intra_var = 0

    def get_stat(self, x):
        # input x [n, p]
        n, p = x.shape
        mu = np.sum(x, axis=0) / n
        x = x - mu
        if p == 1:
            return mu[0], np.sqrt(np.sum(x*x) / n)

        return mu.reshape(p, 1), np.dot(x.T, x) / n

    def train(self, pos, neg):
        # positive example and negative example for training
        # return the \beta parameter the inter_class variance and the intra_class variance
        n_pos, n_neg = pos.shape[0], neg.shape[0]
        mu_pos, cov_pos = self.get_stat(pos)
        mu_neg, cov_neg = self.get_stat(neg)
        sw = n_pos * cov_pos + n_neg * cov_neg
        mu_diff = mu_pos - mu_neg
        self.beta = np.dot(np.linalg.pinv(sw), mu_diff)
        self.inter_var = np.power(np.dot(mu_diff.T, self.beta), 2)[0][0]
        var_pos = np.dot(np.dot(self.beta.T, cov_pos), self.beta)[0][0]
        var_neg = np.dot(np.dot(self.beta.T, cov_neg), self.beta)[0][0]
        self.intra_var = (n_pos * var_pos + n_neg * var_neg)

        self.mu_pos, self.var_pos = self.get_stat(np.dot(pos, self.beta))
        self.mu_neg, self.var_neg = self.get_stat(np.dot(neg, self.beta))

    def predict(self, x):
        proj = np.dot(x, self.beta)
        pred_pos = np.exp(-np.power((proj - self.mu_pos) / self.var_pos, 2)) / self.var_pos
        pred_neg = np.exp(-np.power((proj - self.mu_neg) / self.var_neg, 2)) / self.var_neg
        pred = (pred_neg < pred_pos).astype(np.float32)
        return pred

    def score(self, x, y_true):
        y_pred = self.predict(x)
        return np.sum(y_pred == y_true) / y_true.shape[0]


if __name__ == '__main__':
    lda = LDA(900)
    train_pos = np.load(DATA_PATH + "\\hog\\train_positive.npz")
    train_neg = np.load(DATA_PATH + "\\hog\\train_negative.npz")
    lda.train(train_pos['feature'], train_neg['feature'])

    print("the inter-class variance = {:.8e}, the intra-class variance = {:.8e}".
          format(lda.inter_var, lda.intra_var))

    test_data = np.load(DATA_PATH + "\\hog\\test.npz")
    xt, yt = test_data['feature'], test_data['label']
    acc = lda.score(xt, yt)
    print("accuracy = %.4f" % acc)




