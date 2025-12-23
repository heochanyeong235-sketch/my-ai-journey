"""
Tests for the ML module.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ml.pytorch_basics import (
    tensor_creation,
    tensor_operations,
    tensor_indexing,
    tensor_reshaping,
    autograd_demo,
)
from ml.neural_networks import SimpleNN, DeepNN
from ml.cnn import SimpleCNN, MiniResNet


class TestPyTorchBasics:
    """Tests for PyTorch basics module."""

    def test_tensor_creation(self):
        tensors = tensor_creation()
        assert tensors["t1"].shape == (5,)
        assert tensors["t2d"].shape == (3, 3)
        assert tensors["zeros"].shape == (3, 3)
        assert tensors["ones"].shape == (2, 4)

    def test_tensor_operations(self):
        ops = tensor_operations()
        assert ops["sum"].item() == 15.0
        assert ops["mean"].item() == 3.0

    def test_tensor_indexing(self):
        idx = tensor_indexing()
        assert idx["element[1,2]"].item() == 7
        assert idx["row_0"].tolist() == [1, 2, 3, 4]

    def test_tensor_reshaping(self):
        rs = tensor_reshaping()
        assert rs["view_3x4"].shape == (3, 4)
        assert rs["view_3d"].shape == (2, 2, 3)

    def test_autograd(self):
        grad = autograd_demo()
        assert grad["dz/dx"] == 4.0  # 2x = 2*2 = 4
        assert grad["dz/dy"] == 27.0  # 3y^2 = 3*3*3 = 27


class TestNeuralNetworks:
    """Tests for neural networks module."""

    def test_simple_nn(self):
        model = SimpleNN(10, 20, 5)
        x = torch.randn(4, 10)
        output = model(x)
        assert output.shape == (4, 5)

    def test_deep_nn(self):
        model = DeepNN(10, [64, 32, 16], 5)
        x = torch.randn(4, 10)
        output = model(x)
        assert output.shape == (4, 5)


class TestCNN:
    """Tests for CNN module."""

    def test_simple_cnn(self):
        model = SimpleCNN(num_classes=10)
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        assert output.shape == (2, 10)

    def test_mini_resnet(self):
        model = MiniResNet(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)
