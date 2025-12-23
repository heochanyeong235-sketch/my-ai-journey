"""
Neural Networks Tutorial

This module covers neural network basics with PyTorch:
- Building neural networks with nn.Module
- Activation functions
- Loss functions
- Optimizers
- Training loop
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleNN(nn.Module):
    """Simple feedforward neural network."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


class DeepNN(nn.Module):
    """Deeper neural network with multiple layers."""

    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.2):
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SequentialNN(nn.Module):
    """Neural network using nn.Sequential."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x)


def demonstrate_activations():
    """Demonstrate different activation functions."""
    x = torch.linspace(-5, 5, 100)

    activations = {
        "ReLU": nn.ReLU()(x),
        "Sigmoid": nn.Sigmoid()(x),
        "Tanh": nn.Tanh()(x),
        "LeakyReLU": nn.LeakyReLU(0.1)(x),
        "ELU": nn.ELU()(x),
        "GELU": nn.GELU()(x),
        "Softplus": nn.Softplus()(x),
    }

    return activations


def demonstrate_loss_functions():
    """Demonstrate different loss functions."""
    # Predictions and targets
    predictions = torch.randn(10, 5)
    targets_regression = torch.randn(10, 5)
    targets_classification = torch.randint(0, 5, (10,))
    targets_binary = torch.rand(10, 5)

    losses = {
        "MSELoss": nn.MSELoss()(predictions, targets_regression).item(),
        "L1Loss": nn.L1Loss()(predictions, targets_regression).item(),
        "CrossEntropyLoss": nn.CrossEntropyLoss()(predictions, targets_classification).item(),
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss()(predictions, targets_binary).item(),
        "SmoothL1Loss": nn.SmoothL1Loss()(predictions, targets_regression).item(),
    }

    return losses


def demonstrate_optimizers():
    """Demonstrate different optimizers."""
    model = SimpleNN(10, 20, 5)

    optimizers = {
        "SGD": optim.SGD(model.parameters(), lr=0.01),
        "SGD_momentum": optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        "Adam": optim.Adam(model.parameters(), lr=0.001),
        "AdamW": optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01),
        "RMSprop": optim.RMSprop(model.parameters(), lr=0.001),
    }

    return list(optimizers.keys())


def training_loop_example():
    """Demonstrate a complete training loop."""
    # Create synthetic dataset
    torch.manual_seed(42)
    X = torch.randn(1000, 10)
    y = (X[:, 0] + X[:, 1] * 2 - X[:, 2] + 0.1 * torch.randn(1000)).unsqueeze(1)

    # Split into train and validation
    train_X, val_X = X[:800], X[800:]
    train_y, val_y = y[:800], y[800:]

    # Create data loaders
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize model, loss, and optimizer
    model = SimpleNN(10, 32, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 10
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)

    return {
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "history": history,
    }


def model_inspection(model):
    """Inspect model architecture and parameters."""
    info = {
        "architecture": str(model),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "layers": [],
    }

    for name, layer in model.named_modules():
        if name:
            info["layers"].append(f"{name}: {layer.__class__.__name__}")

    return info


def save_and_load_model():
    """Demonstrate saving and loading models."""
    # Create and train a simple model
    model = SimpleNN(10, 20, 5)

    # Method 1: Save state dict (recommended)
    # torch.save(model.state_dict(), 'model_state.pth')

    # Method 2: Save entire model
    # torch.save(model, 'model_full.pth')

    # Loading
    # model.load_state_dict(torch.load('model_state.pth'))
    # model = torch.load('model_full.pth')

    return "Model save/load methods demonstrated (not actually saved)"


def learning_rate_scheduling():
    """Demonstrate learning rate schedulers."""
    model = SimpleNN(10, 20, 5)
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    schedulers = {
        "StepLR": optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
        "ExponentialLR": optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
        "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50),
        "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5
        ),
    }

    return list(schedulers.keys())


if __name__ == "__main__":
    print("=== Neural Networks Tutorial ===")

    # Create models
    simple_model = SimpleNN(10, 20, 5)
    deep_model = DeepNN(10, [64, 32, 16], 5)
    sequential_model = SequentialNN(10, 32, 5)

    print("\nModel Architectures:")
    print("\nSimple NN:")
    print(simple_model)

    print("\nDeep NN:")
    print(deep_model)

    # Model inspection
    print("\nModel Inspection:")
    info = model_inspection(deep_model)
    print(f"  Total parameters: {info['parameters']}")
    print(f"  Trainable parameters: {info['trainable_parameters']}")

    # Activation functions
    print("\nActivation Functions:")
    for name in demonstrate_activations().keys():
        print(f"  - {name}")

    # Loss functions
    print("\nLoss Functions:")
    losses = demonstrate_loss_functions()
    for name, value in losses.items():
        print(f"  {name}: {value:.4f}")

    # Optimizers
    print("\nOptimizers available:", demonstrate_optimizers())

    # LR Schedulers
    print("\nLR Schedulers:", learning_rate_scheduling())

    # Training example
    print("\nTraining Example:")
    results = training_loop_example()
    print(f"  Final train loss: {results['final_train_loss']:.4f}")
    print(f"  Final val loss: {results['final_val_loss']:.4f}")
