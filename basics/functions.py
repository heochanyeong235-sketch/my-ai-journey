"""
Lesson 03: Functions

This module covers Python functions:
- Function definition and calling
- Parameters and arguments
- Return values
- Lambda functions
- Decorators
"""


def greet(name):
    """Simple function with one parameter."""
    return f"Hello, {name}!"


def add(a, b):
    """Function with multiple parameters."""
    return a + b


def power(base, exponent=2):
    """Function with default parameter."""
    return base ** exponent


def describe_person(name, age, **kwargs):
    """Function with keyword arguments."""
    description = f"{name} is {age} years old"
    for key, value in kwargs.items():
        description += f", {key}: {value}"
    return description


def sum_all(*args):
    """Function with variable arguments."""
    return sum(args)


def apply_operation(x, y, operation):
    """Higher-order function that takes a function as argument."""
    return operation(x, y)


def create_multiplier(factor):
    """Closure: function that returns a function."""

    def multiplier(x):
        return x * factor

    return multiplier


def memoize(func):
    """Decorator for caching function results."""
    cache = {}

    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return wrapper


@memoize
def fibonacci(n):
    """Recursive Fibonacci with memoization."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def demonstrate_lambda():
    """Demonstrate lambda functions."""
    # Lambda for simple operations
    square = lambda x: x ** 2
    add_lambda = lambda x, y: x + y

    # Lambda with map, filter, reduce
    numbers = [1, 2, 3, 4, 5]
    squared = list(map(lambda x: x ** 2, numbers))
    evens = list(filter(lambda x: x % 2 == 0, numbers))

    # Lambda for sorting
    pairs = [(1, "one"), (3, "three"), (2, "two")]
    sorted_pairs = sorted(pairs, key=lambda x: x[0])

    return {
        "square_4": square(4),
        "add_3_5": add_lambda(3, 5),
        "squared": squared,
        "evens": evens,
        "sorted_pairs": sorted_pairs,
    }


if __name__ == "__main__":
    print("=== Functions Demo ===")

    # Basic functions
    print("\nGreeting:", greet("Alice"))
    print("Add 3 + 5:", add(3, 5))
    print("Power 2^3:", power(2, 3))
    print("Power 2 (default):", power(2))

    # Variable arguments
    print("\nSum all 1,2,3,4,5:", sum_all(1, 2, 3, 4, 5))

    # Keyword arguments
    print("Describe:", describe_person("Bob", 25, city="NYC", job="Developer"))

    # Higher-order functions
    print("\nApply add:", apply_operation(3, 4, add))

    # Closures
    double = create_multiplier(2)
    triple = create_multiplier(3)
    print("Double 5:", double(5))
    print("Triple 5:", triple(5))

    # Memoization
    print("\nFibonacci 10:", fibonacci(10))
    print("Fibonacci 20:", fibonacci(20))

    # Lambda functions
    print("\nLambda functions:", demonstrate_lambda())
