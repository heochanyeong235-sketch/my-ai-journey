"""
Convolutional Neural Networks Tutorial

This module covers CNNs with PyTorch:
- Convolutional layers
- Pooling layers
- Building CNN architectures
- Image classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Simple CNN for image classification."""

    def __init__(self, num_classes=10):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Conv block 1: 28x28 -> 14x14
        x = self.pool(F.relu(self.conv1(x)))

        # Conv block 2: 14x14 -> 7x7
        x = self.pool(F.relu(self.conv2(x)))

        # Conv block 3: 7x7 -> 3x3
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten
        x = x.view(-1, 64 * 3 * 3)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class VGGBlock(nn.Module):
    """VGG-style convolutional block."""

    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MiniVGG(nn.Module):
    """Mini VGG-style network."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            VGGBlock(3, 64, 2),  # 32 -> 16
            VGGBlock(64, 128, 2),  # 16 -> 8
            VGGBlock(128, 256, 3),  # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MiniResNet(nn.Module):
    """Mini ResNet architecture."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def demonstrate_conv_operations():
    """Demonstrate convolution operations."""
    # Create sample image batch: (batch, channels, height, width)
    images = torch.randn(4, 3, 32, 32)

    # Different convolution configurations
    conv_3x3 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    conv_5x5 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
    conv_stride = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
    conv_dilated = nn.Conv2d(3, 16, kernel_size=3, padding=2, dilation=2)

    return {
        "input_shape": images.shape,
        "conv_3x3_output": conv_3x3(images).shape,
        "conv_5x5_output": conv_5x5(images).shape,
        "conv_stride_output": conv_stride(images).shape,
        "conv_dilated_output": conv_dilated(images).shape,
    }


def demonstrate_pooling():
    """Demonstrate pooling operations."""
    feature_map = torch.randn(1, 16, 32, 32)

    max_pool = nn.MaxPool2d(2, 2)
    avg_pool = nn.AvgPool2d(2, 2)
    adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
    global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

    return {
        "input_shape": feature_map.shape,
        "max_pool_output": max_pool(feature_map).shape,
        "avg_pool_output": avg_pool(feature_map).shape,
        "adaptive_pool_output": adaptive_pool(feature_map).shape,
        "global_max_pool_output": global_max_pool(feature_map).shape,
    }


def demonstrate_batch_norm():
    """Demonstrate batch normalization."""
    # For conv layers
    bn2d = nn.BatchNorm2d(64)

    # For fully connected layers
    bn1d = nn.BatchNorm1d(128)

    # Layer normalization (alternative)
    layer_norm = nn.LayerNorm([64, 32, 32])

    return {
        "BatchNorm2d": "Normalizes over (N, H, W) for each channel",
        "BatchNorm1d": "Normalizes over N for each feature",
        "LayerNorm": "Normalizes over (C, H, W) for each sample",
    }


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_size):
    """Get model summary with output shapes."""

    def hook_fn(module, input, output):
        class_name = module.__class__.__name__
        if hasattr(output, "shape"):
            summary.append(f"{class_name}: {output.shape}")

    summary = []
    hooks = []

    for layer in model.modules():
        if not isinstance(layer, nn.Sequential) and layer != model:
            hooks.append(layer.register_forward_hook(hook_fn))

    # Forward pass
    x = torch.randn(*input_size)
    with torch.no_grad():
        model(x)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return summary[:10]  # Return first 10 layers


if __name__ == "__main__":
    print("=== CNN Tutorial ===")

    # Create models
    simple_cnn = SimpleCNN(num_classes=10)
    mini_vgg = MiniVGG(num_classes=10)
    mini_resnet = MiniResNet(num_classes=10)

    print("\nModel Architectures:")

    print("\nSimple CNN:")
    print(f"  Parameters: {count_parameters(simple_cnn):,}")
    x = torch.randn(1, 1, 28, 28)
    out = simple_cnn(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")

    print("\nMini VGG:")
    print(f"  Parameters: {count_parameters(mini_vgg):,}")
    x = torch.randn(1, 3, 32, 32)
    out = mini_vgg(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")

    print("\nMini ResNet:")
    print(f"  Parameters: {count_parameters(mini_resnet):,}")
    x = torch.randn(1, 3, 32, 32)
    out = mini_resnet(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")

    print("\nConvolution Operations:")
    conv_ops = demonstrate_conv_operations()
    for name, shape in conv_ops.items():
        print(f"  {name}: {shape}")

    print("\nPooling Operations:")
    pool_ops = demonstrate_pooling()
    for name, shape in pool_ops.items():
        print(f"  {name}: {shape}")

    print("\nBatch Normalization:")
    bn_info = demonstrate_batch_norm()
    for name, desc in bn_info.items():
        print(f"  {name}: {desc}")

    print("\nModel Summary (Simple CNN):")
    summary = model_summary(simple_cnn, (1, 1, 28, 28))
    for line in summary:
        print(f"  {line}")
