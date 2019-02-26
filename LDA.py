import numpy as np


class LDAClassifier:
    def __init__(self, parameter_dict):
        self._dims = parameter_dict["Dimension"]
        self._weight = np.random.randn(self._dims, 1)
        self._positive_value = None
        self._negative_value = None

    def _within_class_scatter(self, x_negative, x_positive, mean_negative, mean_positive):
        sigma = np.zeros((self._dims, self._dims))
        for i in range(x_negative.shape[0]):
            temp = x_negative[i, :].transpose() - mean_negative
            temp = temp.reshape((-1, 1))
            sigma += np.dot(temp, temp.transpose())
        for i in range(x_positive.shape[0]):
            temp = x_positive[i, :].transpose() - mean_positive
            temp = temp.reshape((-1, 1))
            sigma += np.dot(temp, temp.transpose())
        return sigma

    def _between_class_scatter(self, mean_negative, mean_positive):
        temp = mean_negative - mean_positive
        return np.dot(temp, temp.transpose())

    @staticmethod
    def _divide_data(x, labels):
        positive_data = x[labels.reshape(-1), :]
        negative_data = x[~labels.reshape(-1), :]
        return {
            'Positive Data': positive_data,
            'Negative Data': negative_data
        }

    def rayleigh_quotient(self, x, labels):
        assert(type(x) == np.ndarray)
        assert(len(x.shape) == 2)
        assert(x.shape[1] == self._dims)
        data_div_dict = self._divide_data(x, labels)
        positive_x, negative_x = data_div_dict['Positive Data'], data_div_dict['Negative Data']
        positive_mean = np.mean(positive_x, axis=0).transpose()
        negative_mean = np.mean(negative_x, axis=0).transpose()
        within_class_scatter = self._within_class_scatter(negative_x, positive_x, negative_mean, positive_mean)
        between_class_scatter = self._between_class_scatter(negative_mean, positive_mean)
        numerator = np.dot(self._weight.transpose(), np.dot(between_class_scatter, self._weight))
        denominator = np.dot(self._weight.transpose(), np.dot(within_class_scatter, self._weight))
        return numerator / denominator

    def fit(self, x, labels, iterations):
        assert(type(x) == np.ndarray)
        assert(len(x.shape) == 2)
        assert(x.shape[1] == self._dims)
        data_div_dict = self._divide_data(x, labels)
        positive_x, negative_x = data_div_dict['Positive Data'], data_div_dict['Negative Data']
        positive_mean = np.mean(positive_x, axis=0).transpose()
        negative_mean = np.mean(negative_x, axis=0).transpose()
        within_class_scatter = self._within_class_scatter(negative_x, positive_x, negative_mean, positive_mean)
        u, sigma, vt = np.linalg.svd(within_class_scatter)
        sigma_inv = np.diag(1 / sigma)
        within_class_scatter_inv = np.dot(vt.transpose(), np.dot(sigma_inv, u.transpose()))
        self._weight = np.dot(within_class_scatter_inv, negative_mean - positive_mean)
        self._positive_value = np.dot(self._weight.transpose(), positive_mean)
        self._negative_value = np.dot(self._weight.transpose(), negative_mean)

    def predict(self, x):
        assert(type(x) == np.ndarray)
        assert(len(x.shape) == 2)
        assert(x.shape[1] == self._dims)
        if self._negative_value is None or self._positive_value is None:
            return None
        projected_value = np.dot(x, self._weight).reshape((-1, 1))
        distance_to_negative = np.abs(projected_value - self._negative_value)
        distance_to_positive = np.abs(projected_value - self._positive_value)
        distance_ratio = distance_to_negative / distance_to_positive
        distance_ratio = distance_ratio / np.max(distance_ratio)
        return distance_ratio
