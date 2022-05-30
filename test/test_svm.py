import numpy as np
import pytest

from svm import SVM


@pytest.fixture
def features():
    features = np.array([[10.],
                         [1.],
                         [8],
                         [-20]])
    return features


@pytest.fixture
def targets():
    targets = np.array([-1., 1., -1., 1.])
    return targets


@pytest.fixture
def svm():
    svm = SVM(learning_rate=1e-2, regularization_strength=1)
    svm.weights = np.array([-0.16, 0.43])
    return svm


def test_predict(svm, features, targets):
    predicted_labels = svm.predict(features)

    np.testing.assert_equal(targets, predicted_labels)


def test_compute_cost(svm, features, targets):
    expected_cost = 0.32525

    computed_cost = svm.compute_cost(features, targets, svm.weights)

    assert expected_cost == computed_cost


def test_compute_cost_gradient(svm, features, targets):
    expected_gradient = np.array([1.59, 0.43])

    computed_gradient = svm.compute_cost_gradient(features, targets, svm.weights)

    np.testing.assert_almost_equal(expected_gradient, computed_gradient)


def test__add_bias_to_features(svm):
    features = np.array([[0., 0.], [0., 0.]])
    expected_features = np.array([[0., 0., 1.], [0., 0., 1.]])

    calculated_features = svm._add_bias_to_features(features)

    np.testing.assert_equal(expected_features, calculated_features)


def test_fit(features, targets):
    model = SVM(learning_rate=1e-2, regularization_strength=1)
    expected_weights = np.array([-0.14952792, 0.20005])

    model.fit(features, targets, max_epochs=1000)
    calculated_weights = model.weights

    np.testing.assert_almost_equal(expected_weights, calculated_weights)
