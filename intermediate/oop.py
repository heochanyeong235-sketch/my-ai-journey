"""
Lesson 01: Object-Oriented Programming

This module covers OOP concepts in Python:
- Classes and Objects
- Inheritance
- Encapsulation
- Polymorphism
- Magic methods
"""

from abc import ABC, abstractmethod


class Animal(ABC):
    """Abstract base class for animals."""

    def __init__(self, name, age):
        self._name = name  # Protected attribute
        self._age = age

    @property
    def name(self):
        """Getter for name."""
        return self._name

    @name.setter
    def name(self, value):
        """Setter for name with validation."""
        if not value:
            raise ValueError("Name cannot be empty")
        self._name = value

    @property
    def age(self):
        """Getter for age."""
        return self._age

    @abstractmethod
    def speak(self):
        """Abstract method - must be implemented by subclasses."""
        pass

    def describe(self):
        """Describe the animal."""
        return f"{self._name} is {self._age} years old"


class Dog(Animal):
    """Dog class inheriting from Animal."""

    def __init__(self, name, age, breed):
        super().__init__(name, age)
        self.breed = breed

    def speak(self):
        """Dog's speak implementation."""
        return "Woof!"

    def fetch(self):
        """Dog-specific method."""
        return f"{self._name} is fetching the ball!"


class Cat(Animal):
    """Cat class inheriting from Animal."""

    def __init__(self, name, age, indoor=True):
        super().__init__(name, age)
        self.indoor = indoor

    def speak(self):
        """Cat's speak implementation."""
        return "Meow!"

    def purr(self):
        """Cat-specific method."""
        return f"{self._name} is purring..."


class Vector:
    """Vector class demonstrating magic methods."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        """Official string representation."""
        return f"Vector({self.x}, {self.y})"

    def __str__(self):
        """User-friendly string representation."""
        return f"({self.x}, {self.y})"

    def __add__(self, other):
        """Vector addition."""
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """Vector subtraction."""
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        """Scalar multiplication."""
        return Vector(self.x * scalar, self.y * scalar)

    def __eq__(self, other):
        """Equality comparison."""
        return self.x == other.x and self.y == other.y

    def __abs__(self):
        """Magnitude of vector."""
        return (self.x ** 2 + self.y ** 2) ** 0.5


class BankAccount:
    """Bank account demonstrating encapsulation."""

    def __init__(self, owner, balance=0):
        self.owner = owner
        self.__balance = balance  # Private attribute

    def deposit(self, amount):
        """Deposit money into account."""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.__balance += amount
        return self.__balance

    def withdraw(self, amount):
        """Withdraw money from account."""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.__balance:
            raise ValueError("Insufficient funds")
        self.__balance -= amount
        return self.__balance

    def get_balance(self):
        """Get current balance."""
        return self.__balance


class Counter:
    """Class with class variables and methods."""

    count = 0  # Class variable

    def __init__(self, name):
        self.name = name
        Counter.count += 1

    @classmethod
    def get_count(cls):
        """Class method to get total count."""
        return cls.count

    @staticmethod
    def validate_name(name):
        """Static method for validation."""
        return isinstance(name, str) and len(name) > 0


if __name__ == "__main__":
    print("=== OOP Demo ===")

    # Inheritance and polymorphism
    dog = Dog("Buddy", 3, "Golden Retriever")
    cat = Cat("Whiskers", 5)

    print("\nAnimals:")
    print(f"  {dog.describe()}, says: {dog.speak()}")
    print(f"  {cat.describe()}, says: {cat.speak()}")
    print(f"  {dog.fetch()}")
    print(f"  {cat.purr()}")

    # Magic methods
    v1 = Vector(3, 4)
    v2 = Vector(1, 2)

    print("\nVectors:")
    print(f"  v1 = {v1}")
    print(f"  v2 = {v2}")
    print(f"  v1 + v2 = {v1 + v2}")
    print(f"  v1 - v2 = {v1 - v2}")
    print(f"  v1 * 2 = {v1 * 2}")
    print(f"  |v1| = {abs(v1)}")

    # Encapsulation
    account = BankAccount("Alice", 100)
    print("\nBank Account:")
    print(f"  Initial balance: ${account.get_balance()}")
    account.deposit(50)
    print(f"  After deposit: ${account.get_balance()}")
    account.withdraw(30)
    print(f"  After withdrawal: ${account.get_balance()}")

    # Class methods and static methods
    c1 = Counter("First")
    c2 = Counter("Second")
    print("\nCounters:")
    print(f"  Total counters: {Counter.get_count()}")
    print(f"  Valid name 'test': {Counter.validate_name('test')}")
