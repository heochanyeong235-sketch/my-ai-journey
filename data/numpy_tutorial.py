"""
NumPy Tutorial

This module covers NumPy fundamentals:
- Array creation
- Array operations
- Indexing and slicing
- Broadcasting
- Linear algebra
"""

import numpy as np


def create_arrays():
    """Demonstrate array creation methods."""
    # From Python list
    arr1 = np.array([1, 2, 3, 4, 5])

    # 2D array
    arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Special arrays
    zeros = np.zeros((3, 3))
    ones = np.ones((2, 4))
    identity = np.eye(3)
    random_arr = np.random.rand(3, 3)

    # Ranges
    range_arr = np.arange(0, 10, 2)
    linspace_arr = np.linspace(0, 1, 5)

    return {
        "arr1": arr1,
        "arr2d": arr2d,
        "zeros": zeros,
        "ones": ones,
        "identity": identity,
        "random": random_arr,
        "range": range_arr,
        "linspace": linspace_arr,
    }


def array_operations():
    """Demonstrate array operations."""
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([10, 20, 30, 40, 50])

    # Element-wise operations
    addition = a + b
    subtraction = b - a
    multiplication = a * b
    division = b / a

    # Universal functions
    sqrt = np.sqrt(a)
    exp = np.exp(a)
    log = np.log(a)
    sin = np.sin(a)

    # Aggregations
    total = np.sum(a)
    mean = np.mean(a)
    std = np.std(a)
    min_val = np.min(a)
    max_val = np.max(a)

    return {
        "addition": addition,
        "subtraction": subtraction,
        "multiplication": multiplication,
        "division": division,
        "sqrt": sqrt,
        "exp": exp,
        "sum": total,
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
    }


def indexing_and_slicing():
    """Demonstrate array indexing and slicing."""
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    # Basic indexing
    element = arr[1, 2]  # Row 1, Col 2

    # Slicing
    row = arr[0]  # First row
    col = arr[:, 1]  # Second column
    subarray = arr[0:2, 1:3]

    # Boolean indexing
    mask = arr > 5
    filtered = arr[mask]

    # Fancy indexing
    indices = np.array([0, 2])
    selected_rows = arr[indices]

    return {
        "original": arr,
        "element[1,2]": element,
        "row_0": row,
        "col_1": col,
        "subarray": subarray,
        "mask": mask,
        "filtered": filtered,
        "selected_rows": selected_rows,
    }


def broadcasting():
    """Demonstrate NumPy broadcasting."""
    # Broadcasting scalar
    arr = np.array([1, 2, 3, 4])
    scalar_result = arr * 10

    # Broadcasting 1D with 2D
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    row_vector = np.array([1, 0, 1])
    broadcast_result = matrix + row_vector

    # Column broadcast
    col_vector = np.array([[10], [20], [30]])
    col_broadcast = matrix + col_vector

    return {
        "scalar_broadcast": scalar_result,
        "row_broadcast": broadcast_result,
        "col_broadcast": col_broadcast,
    }


def linear_algebra():
    """Demonstrate linear algebra operations."""
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    # Matrix multiplication
    dot_product = np.dot(A, B)
    matmul = A @ B

    # Transpose
    transpose = A.T

    # Determinant
    det = np.linalg.det(A)

    # Inverse
    inv = np.linalg.inv(A)

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Solve linear system Ax = b
    b = np.array([1, 2])
    x = np.linalg.solve(A, b)

    return {
        "A": A,
        "B": B,
        "dot_product": dot_product,
        "transpose": transpose,
        "determinant": det,
        "inverse": inv,
        "eigenvalues": eigenvalues,
        "solution_x": x,
    }


def array_reshaping():
    """Demonstrate array reshaping."""
    arr = np.arange(12)

    reshaped_2d = arr.reshape(3, 4)
    reshaped_3d = arr.reshape(2, 2, 3)
    flattened = reshaped_2d.flatten()
    transposed = reshaped_2d.T

    # Stacking
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    vstacked = np.vstack([a, b])
    hstacked = np.hstack([a, b])

    return {
        "original": arr,
        "reshaped_2d": reshaped_2d,
        "reshaped_3d": reshaped_3d,
        "flattened": flattened,
        "transposed": transposed,
        "vstacked": vstacked,
        "hstacked": hstacked,
    }


if __name__ == "__main__":
    print("=== NumPy Tutorial ===")

    print("\nArray Creation:")
    for name, arr in create_arrays().items():
        print(f"  {name}: shape={getattr(arr, 'shape', 'N/A')}")

    print("\nArray Operations:")
    ops = array_operations()
    for name, val in list(ops.items())[:5]:
        print(f"  {name}: {val}")

    print("\nIndexing and Slicing:")
    idx = indexing_and_slicing()
    print(f"  Original shape: {idx['original'].shape}")
    print(f"  Element [1,2]: {idx['element[1,2]']}")
    print(f"  Filtered (>5): {idx['filtered']}")

    print("\nBroadcasting:")
    bc = broadcasting()
    for name, val in bc.items():
        print(f"  {name}:\n{val}")

    print("\nLinear Algebra:")
    la = linear_algebra()
    print(f"  Determinant of A: {la['determinant']}")
    print(f"  Eigenvalues: {la['eigenvalues']}")

    print("\nReshaping:")
    rs = array_reshaping()
    print(f"  2D reshaped:\n{rs['reshaped_2d']}")
