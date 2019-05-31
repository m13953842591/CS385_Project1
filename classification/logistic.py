import numpy as np 
import matplotlib as plt 
from sklearn.metrics import accuracy_score
from global_var import * 


class LogisticModel:
    """
    binary logistic regression: output dimension = 1
    using logistic loss
    """
    def __init__(self, input_shape=900, learning_rate=0.01, regularizer=0.01):
        self.input_shape = input_shape
        self.w = np.zeros((input_shape,), dtype=np.float32)
        self.b = 0.0
        self.r = regularizer
        self.lr = learning_rate

    def forward(self, x, y):
        """
        calculate loss with regularizer
        """
        logit = np.dot(x, self.w) + self.b #[n, ]
        loss = np.sum(np.log(1+np.exp(-y * logit))) + self.r * np.sum(self.w*self.w)
        return loss / x.shape[0]

    def backward(self, x, y):
        """
        backward propagation, update weight and bias
        """
        # calculate deriviative
        logit = np.dot(x, self.w) + self.b
        _exp = np.exp(-y*logit)
        del_w = np.sum(_exp * (-y) / (1+_exp) * x.T, axis=1).reshape(self.input_shape) + 2*self.r*self.w
        del_b = np.sum(_exp * (-y) / (1+_exp))
        # update
        self.w -= self.lr * del_w
        self.b -= self.lr * del_b

    def eval(self, x_test, y_test):
        y_pred = (np.dot(x_test, self.w) + self.b) > 0
        y_test = y_test > 0
        return accuracy_score(y_test, y_pred)
    
    def fit(self, x, y, x_test, y_test, batch_size, epoch):
        n = x.shape[0] # total training samples
        batch = n // batch_size
        rest = n - batch * batch_size
        for e in range(epoch):
            for i in range(batch):
                self.backward(x[i: i+batch_size], y[i:i+batch_size])
            if rest > 0:
                self.backward(x[-rest: ], y[-rest:])
            loss = self.forward(x, y)
            acc = self.eval(x_test, y_test)
            print("epoch[%d], loss=%.4f, acc=%.3f"%(e, loss, acc))


if __name__ == '__main__':
    train_data = np.load(HOG_TRAIN)
    x, y = train_data['feature'], train_data['label']
    test_data = np.load(HOG_TEST)
    xt, yt = test_data['feature'], test_data['label']
    model = LogisticModel(900, 0.01, 0.01)
    model.fit(x, y, xt, yt, batch_size=50, epoch=5)


