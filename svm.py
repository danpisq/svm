import numpy as np
from sklearn.utils import shuffle


class SVM:
    def __init__(self, regularization_strength: float = 0.5, learning_rate: float = 1e-2):
        self.regularization_strength = regularization_strength
        self.weights = None
        self.learning_rate = learning_rate

    def fit(self, features: np.ndarray, target: np.ndarray, max_epochs: int = 5000) -> None:
        weights = np.zeros(features.shape[1] + 1)
        # sgd
        for epoch in range(max_epochs):
            features, target = shuffle(features, target) # shuffle to prevent repeating update cycles
            print(f'Epoch {epoch}, loss: {self.compute_cost(features, target, weights)}')
            for ind, x in enumerate(features):
                ascent = self.calculate_cost_gradient(x, target[ind], weights)
                weights = weights - (self.learning_rate * ascent)
        self.weights = weights

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = self._add_bias_to_features(features)
        y_test_predicted = np.array([])
        for i in range(features.shape[0]):
            yp = np.sign(np.dot(self.weights, features[i]))  # model
            y_test_predicted = np.append(y_test_predicted, yp)
        return y_test_predicted

    def compute_cost(self, features, target, weights):
        # calculate hinge loss
        features = self._add_bias_to_features(features)
        distances = 1 - target * (np.dot(features, weights))
        distances[distances < 0] = 0
        hinge_loss = self.regularization_strength * np.sum(distances) / features.shape[0]
        return 0.5 * np.dot(weights, weights) + hinge_loss

    def calculate_cost_gradient(self, features_batch, target_batch, weights):
        # if only one example is passed
        if type(target_batch) == np.float64:
            target_batch = np.array([target_batch])
            features_batch = np.array([features_batch])
        features_batch = self._add_bias_to_features(features_batch)
        distance = 1 - (target_batch * np.dot(features_batch, weights))
        dw = np.zeros(len(weights))

        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = weights
            else:
                di = weights - (self.regularization_strength * target_batch[ind] * features_batch[ind])
            dw += di
        dw = dw / len(target_batch) # average
        return dw

    @staticmethod
    def _add_bias_to_features(features):
        return np.append(features, np.ones((features.shape[0], 1)), axis=1)
