import numpy as np
from sklearn.metrics import accuracy_score
from config import *


class LogisticModel:
    """
    binary logistic regression: output dimension = 1
    using logistic loss
    """

    def __init__(self, input_shape=900, learning_rate=0.001, regularizer=0.01):
        self.input_shape = input_shape
        self.w = np.zeros((input_shape,), dtype=np.float32)
        self.b = 0.0
        self.r = regularizer
        self.lr = learning_rate

    def forward(self, x, y):
        """
        calculate loss with regularizer
        """
        logit = np.dot(x, self.w) + self.b  # [n, ]
        loss = np.sum(np.log(1 + np.exp(-y * logit))) + self.r * np.sum(
            self.w * self.w)
        return loss / x.shape[0]

    def backward(self, x, y, langevin):
        """
        backward propagation, update weight and bias
        """
        # calculate deriviative
        logit = np.dot(x, self.w) + self.b
        _exp = np.exp(-y * logit)
        _div = -y * _exp / (1 + _exp)

        del_w = np.sum(_div * x.T, axis=1).reshape(
            self.input_shape) + 2 * self.r * self.w
        del_b = np.sum(_div)
        # update
        self.w -= self.lr * del_w
        self.b -= self.lr * del_b
        if langevin:
            eps_w = np.random.normal(loc=0, scale=np.sqrt(self.lr),
                                     size=(self.input_shape,))
            self.w += eps_w
            eps_b = np.random.normal(loc=0, scale=np.sqrt(self.lr))
            self.b += eps_b

    def predict_prob(self, x):
        logit = np.dot(x, self.w) + self.b
        return 1 / (1 + np.exp(-logit))

    def predict(self, x):
        threshold = 0.9
        pred = self.predict_prob(x) > threshold
        return pred

    def score(self, x_test, y_test):
        y_pred = (np.dot(x_test, self.w) + self.b) > 0
        y_test = y_test > 0
        return accuracy_score(y_test, y_pred)

    def train(self, x, y, x_test, y_test, batch_size, epoch, langevin=False):
        n = x.shape[0]  # total training samples
        batch = n // batch_size
        rest = n - batch * batch_size
        for e in range(epoch):
            for i in range(batch):
                self.backward(x[i: i + batch_size],
                              y[i: i + batch_size], langevin)
            if rest > 0:
                self.backward(x[-rest:], y[-rest:], langevin)
            loss = self.forward(x, y)
            acc = self.score(x_test, y_test)
            print("epoch[%d], loss=%.4f, acc=%.3f" % (e, loss, acc))


if __name__ == '__main__':
    train_data = np.load(DATA_PATH + "\\hog\\train.npz")
    x, y = train_data['feature'], train_data['label']
    test_data = np.load(DATA_PATH + "\\hog\\test.npz")
    xt, yt = test_data['feature'], test_data['label']
    model = LogisticModel(900, 0.01, 0.01)
    model.train(x, y, xt, yt, batch_size=200, epoch=5, langevin=True)



