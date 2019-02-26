import numpy as np


class LogisticClassifier:
    def __init__(self, parameter_dict):
        dim = parameter_dict["Dimension"]
        assert(dim > 0)
        self._dim = dim
        self._weight = np.random.randn(dim, 1)
        self._bias = np.random.rand()
        self._beta = np.random.randn(dim + 1, 1)

    def loss(self, x, labels):
        assert(type(x) == np.ndarray and type(labels) == np.ndarray)
        assert(x.shape[0] == labels.shape[0])
        assert(labels.shape[1] == 1)
        p = self.predict(x)
        loss_val = labels * np.log(p + 0.0001) + (1 - labels) * np.log(1 - p + 0.0001)
        loss_val = loss_val.sum()
        return loss_val

    def predict(self, x):
        assert(type(x) == np.ndarray)
        assert(x.shape[1] == self._dim)
        linear = np.dot(x, self._weight) + self._bias
        y = 1 - 1 / (1 + np.exp(linear))
        return y

    def fit(self, x, labels, iterations=1000):
        for cyc in range(iterations):
            dbeta = self._first_order_differentiation(x, labels)
            d2beta = self._second_order_differentiation(x)
            beta = np.concatenate((self._weight, np.reshape(self._bias, (1, 1))), axis=0)
            beta -= np.dot(np.linalg.inv(d2beta + 0.001 * np.eye(self._dim + 1, self._dim + 1)), dbeta).reshape(beta.shape[0], 1)
            self._weight = beta[:-1, :]
            self._bias = beta[-1]

    def _first_order_differentiation(self, x, labels):
        assert(type(x) == np.ndarray)
        assert(type(labels) == np.ndarray)
        assert(labels.shape[0] == x.shape[0])
        assert(labels.shape[1] == 1)
        predication = self.predict(x)
        x_add = np.concatenate((x, np.full((x.shape[0], 1), 1)), axis=1)
        dbeta = (x_add * (labels - predication)).sum(axis=0).transpose() * -1

        return dbeta

    def _second_order_differentiation(self, x):
        assert(type(x) == np.ndarray)
        predication = self.predict(x)
        x_add = np.concatenate((x, np.full((x.shape[0], 1), 1)), axis=1)
        d2beta = np.zeros((self._dim + 1, self._dim + 1))
        for i in range(x.shape[0]):
            xl = x_add[i, :]
            d2beta += np.dot(xl.transpose(), xl) * predication[i] * (1 - predication[i])
        return d2beta


