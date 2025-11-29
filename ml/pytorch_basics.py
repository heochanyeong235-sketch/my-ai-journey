"""
PyTorch Basics Tutorial

This module covers PyTorch fundamentals:
- Tensors
- Operations
- Autograd
- GPU acceleration
"""

import torch


def tensor_creation():
    """Demonstrate tensor creation methods."""
    # From Python list
    t1 = torch.tensor([1, 2, 3, 4, 5])

    # 2D tensor
    t2d = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Special tensors
    zeros = torch.zeros(3, 3)
    ones = torch.ones(2, 4)
    random_tensor = torch.rand(3, 3)
    randn_tensor = torch.randn(3, 3)  # Normal distribution

    # Range tensors
    range_tensor = torch.arange(0, 10, 2)
    linspace_tensor = torch.linspace(0, 1, 5)

    # Identity matrix
    eye_tensor = torch.eye(3)

    return {
        "t1": t1,
        "t2d": t2d,
        "zeros": zeros,
        "ones": ones,
        "random": random_tensor,
        "randn": randn_tensor,
        "range": range_tensor,
        "linspace": linspace_tensor,
        "eye": eye_tensor,
    }


def tensor_properties():
    """Demonstrate tensor properties."""
    t = torch.randn(3, 4, 5)

    return {
        "shape": t.shape,
        "dtype": t.dtype,
        "device": t.device,
        "ndim": t.ndim,
        "numel": t.numel(),  # Total elements
        "requires_grad": t.requires_grad,
    }


def tensor_operations():
    """Demonstrate tensor operations."""
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    b = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])

    # Element-wise operations
    addition = a + b
    subtraction = b - a
    multiplication = a * b
    division = b / a

    # Mathematical functions
    sqrt = torch.sqrt(a)
    exp = torch.exp(a)
    log = torch.log(a)
    sin = torch.sin(a)

    # Aggregations
    total = torch.sum(a)
    mean = torch.mean(a)
    std = torch.std(a)
    min_val = torch.min(a)
    max_val = torch.max(a)

    return {
        "addition": addition,
        "multiplication": multiplication,
        "sqrt": sqrt,
        "sum": total,
        "mean": mean,
        "std": std,
    }


def matrix_operations():
    """Demonstrate matrix operations."""
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    # Matrix multiplication
    matmul = torch.matmul(A, B)
    matmul_operator = A @ B

    # Transpose
    transpose = A.T

    # Inverse
    inv = torch.inverse(A)

    # Determinant
    det = torch.det(A)

    # Eigenvalues
    eigenvalues = torch.linalg.eigvals(A)

    return {
        "A": A,
        "B": B,
        "matmul": matmul,
        "transpose": transpose,
        "inverse": inv,
        "determinant": det,
        "eigenvalues": eigenvalues,
    }


def tensor_indexing():
    """Demonstrate tensor indexing and slicing."""
    t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    # Basic indexing
    element = t[1, 2]

    # Slicing
    row = t[0]
    col = t[:, 1]
    submatrix = t[0:2, 1:3]

    # Boolean indexing
    mask = t > 5
    filtered = t[mask]

    return {
        "original": t,
        "element[1,2]": element,
        "row_0": row,
        "col_1": col,
        "submatrix": submatrix,
        "filtered": filtered,
    }


def tensor_reshaping():
    """Demonstrate tensor reshaping."""
    t = torch.arange(12)

    reshaped = t.view(3, 4)
    reshaped_3d = t.view(2, 2, 3)
    flattened = reshaped.flatten()

    # Transpose for 2D
    transposed = reshaped.T

    # Squeeze and unsqueeze
    unsqueezed = t.unsqueeze(0)  # Add dimension
    squeezed = unsqueezed.squeeze(0)  # Remove dimension

    # Stack and concat
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    stacked = torch.stack([a, b])
    concatenated = torch.cat([a, b])

    return {
        "original": t,
        "view_3x4": reshaped,
        "view_3d": reshaped_3d,
        "flattened": flattened,
        "transposed": transposed,
        "stacked": stacked,
        "concatenated": concatenated,
    }


def autograd_demo():
    """Demonstrate autograd for automatic differentiation."""
    # Create tensors with gradient tracking
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)

    # Forward pass
    z = x**2 + y**3

    # Backward pass
    z.backward()

    # Gradients
    return {
        "x": x.item(),
        "y": y.item(),
        "z": z.item(),
        "dz/dx": x.grad.item(),  # Should be 2x = 4
        "dz/dy": y.grad.item(),  # Should be 3y^2 = 27
    }


def neural_network_gradient():
    """Demonstrate gradient computation for a simple neural network."""
    # Input
    x = torch.randn(1, 10)

    # Weights with gradient tracking
    w1 = torch.randn(10, 5, requires_grad=True)
    w2 = torch.randn(5, 1, requires_grad=True)

    # Forward pass
    h = torch.relu(x @ w1)  # Hidden layer
    y = h @ w2  # Output

    # Loss
    target = torch.tensor([[1.0]])
    loss = (y - target) ** 2

    # Backward pass
    loss.backward()

    return {
        "input_shape": x.shape,
        "w1_shape": w1.shape,
        "w2_shape": w2.shape,
        "output": y.item(),
        "loss": loss.item(),
        "w1_grad_shape": w1.grad.shape,
        "w2_grad_shape": w2.grad.shape,
    }


def device_operations():
    """Demonstrate device operations (CPU/GPU)."""
    # Check device availability
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    # Create tensor on CPU
    cpu_tensor = torch.randn(3, 3)

    # Move to GPU if available
    if cuda_available:
        gpu_tensor = cpu_tensor.to("cuda")
        device = gpu_tensor.device
    elif mps_available:
        gpu_tensor = cpu_tensor.to("mps")
        device = gpu_tensor.device
    else:
        gpu_tensor = None
        device = "cpu"

    return {
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "cpu_tensor_device": str(cpu_tensor.device),
        "gpu_tensor_device": str(device),
    }


if __name__ == "__main__":
    print("=== PyTorch Basics Tutorial ===")

    print("\nTensor Creation:")
    tensors = tensor_creation()
    for name, t in list(tensors.items())[:3]:
        print(f"  {name}: {t}")

    print("\nTensor Properties:")
    props = tensor_properties()
    for name, val in props.items():
        print(f"  {name}: {val}")

    print("\nTensor Operations:")
    ops = tensor_operations()
    for name, val in list(ops.items())[:4]:
        print(f"  {name}: {val}")

    print("\nMatrix Operations:")
    mat_ops = matrix_operations()
    print(f"  A @ B:\n{mat_ops['matmul']}")

    print("\nTensor Indexing:")
    idx = tensor_indexing()
    print(f"  Filtered (>5): {idx['filtered']}")

    print("\nTensor Reshaping:")
    rs = tensor_reshaping()
    print(f"  Stacked:\n{rs['stacked']}")

    print("\nAutograd Demo:")
    grad = autograd_demo()
    for name, val in grad.items():
        print(f"  {name}: {val}")

    print("\nNeural Network Gradient:")
    nn_grad = neural_network_gradient()
    for name, val in nn_grad.items():
        print(f"  {name}: {val}")

    print("\nDevice Operations:")
    devices = device_operations()
    for name, val in devices.items():
        print(f"  {name}: {val}")
