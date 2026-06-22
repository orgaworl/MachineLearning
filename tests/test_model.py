import pytest
import torch
from activation_exp.model import CNN, ACTIVATIONS


def test_cifar10_output_shape():
    model = CNN(in_channels=3, num_classes=10, activation="relu")
    out = model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)


def test_mnist_output_shape():
    model = CNN(in_channels=1, num_classes=10, activation="relu")
    out = model(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 10)


def test_all_activations_produce_correct_shape():
    for act in ["relu", "leaky_relu", "swish", "gelu"]:
        model = CNN(in_channels=3, num_classes=10, activation=act)
        out = model(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, 10), f"Failed for activation={act}"


def test_invalid_activation_raises():
    with pytest.raises(KeyError):
        CNN(in_channels=3, num_classes=10, activation="tanh")


def test_activations_dict_contains_expected_keys():
    assert set(ACTIVATIONS.keys()) == {"relu", "leaky_relu", "swish", "gelu"}
