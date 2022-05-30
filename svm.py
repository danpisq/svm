import numpy as np
from sklearn.utils import shuffle


class SVM:
    def __init__(self, regularization_strength: float = 0.5, learning_rate: float = 1e-2):
        self.regularization_strength = regularization_strength
        self.weights = None
        self.learning_rate = learning_rate

    def fit(self, features: np.ndarray, target: np.ndarray, max_epochs: int = 5000, logging=False) -> None:
        weights = np.zeros(features.shape[1] + 1)
        # sgd
        for epoch in range(max_epochs):
            features, target = shuffle(features, target)
            if logging:
                print(f'Epoch {epoch}, loss: {self.compute_cost(features, target, weights)}')
            ascent = self.compute_cost_gradient(features, target, weights)

            weights -= self.learning_rate * ascent

        self.weights = weights

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = self._add_bias_to_features(features)
        y_test_predicted = np.sign(np.dot(features, self.weights))
        return y_test_predicted

    def compute_cost(self, features, target, weights):
        # calculate hinge loss
        features = self._add_bias_to_features(features)
        distances = 1 - target * (np.dot(features, weights))
        distances[distances < 0] = 0
        hinge_loss = self.regularization_strength * np.sum(distances) / features.shape[0]
        return 0.5 * np.dot(weights, weights) + hinge_loss

    def compute_cost_gradient(self, features_batch, target_batch, weights):
        # if only one example is passed
        if type(target_batch) == np.float64:
            target_batch = np.array([target_batch])
            features_batch = np.array([features_batch])
        features_batch = self._add_bias_to_features(features_batch)
        negative_distance_mask = self.calculate_negative_distance_mask(weights, features_batch, target_batch)

        negative_distance_derivative = self.partial_derivative_for_negative_distance(negative_distance_mask, weights)
        positive_distance_derivative = self.partial_derivative_for_positive_distance(
            weights,
            negative_distance_mask,
            features_batch,
            target_batch
        )

        samples_derivative = negative_distance_derivative + positive_distance_derivative
        batch_derivative = np.average(samples_derivative, axis=0)
        return batch_derivative

    @staticmethod
    def calculate_negative_distance_mask(weights, features_batch, target_batch):
        distance = 1 - (target_batch * np.dot(features_batch, weights))
        return np.tile(np.reshape(distance <= 0., (-1, 1)), (1, features_batch.shape[1]))

    @staticmethod
    def partial_derivative_for_negative_distance(weights, distance_smaller_than_zero):
        distance_mask = distance_smaller_than_zero.astype(np.float)
        return np.multiply(distance_mask, weights)

    def partial_derivative_for_positive_distance(self, weights, distance_smaller_than_zero, features_batch, target_batch):
        distance_mask = (~distance_smaller_than_zero).astype(np.float)
        weights_tiled = np.tile(weights, (features_batch.shape[0], 1))
        target_tiled = np.tile(target_batch.reshape((-1, 1)), (1, features_batch.shape[1]))

        partial_derivative = weights_tiled - self.regularization_strength * target_tiled * features_batch
        return np.multiply(distance_mask, partial_derivative)

    @staticmethod
    def _add_bias_to_features(features):
        return np.append(features, np.ones((features.shape[0], 1)), axis=-1)
