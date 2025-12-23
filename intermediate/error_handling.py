"""
Lesson 03: Error Handling

This module covers exception handling in Python:
- Try/except/finally
- Raising exceptions
- Custom exceptions
- Context managers for cleanup
- Logging errors
"""

import logging


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, message, field=None):
        super().__init__(message)
        self.field = field
        self.message = message


class InsufficientFundsError(Exception):
    """Custom exception for insufficient funds."""

    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"Cannot withdraw ${amount}. Balance: ${balance}")


def divide(a, b):
    """Divide with error handling."""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        logger.error("Division by zero attempted")
        return None
    except TypeError as e:
        logger.error(f"Type error: {e}")
        raise


def safe_int_conversion(value):
    """Safely convert value to integer."""
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Could not convert '{value}' to int")
        return None
    except TypeError:
        logger.warning(f"Invalid type for conversion: {type(value)}")
        return None


def process_list(items, index):
    """Access list with error handling."""
    try:
        return items[index]
    except IndexError:
        logger.warning(f"Index {index} out of range for list of length {len(items)}")
        return None
    except TypeError:
        logger.error("Invalid index type")
        raise


def read_file_safe(filepath):
    """Read file with comprehensive error handling."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None
    except PermissionError:
        logger.error(f"Permission denied: {filepath}")
        return None
    except IOError as e:
        logger.error(f"I/O error reading file: {e}")
        return None


def validate_user_data(data):
    """Validate user data with custom exceptions."""
    if not isinstance(data, dict):
        raise ValidationError("Data must be a dictionary")

    if "name" not in data:
        raise ValidationError("Name is required", field="name")

    if not data.get("name"):
        raise ValidationError("Name cannot be empty", field="name")

    if "age" in data:
        if not isinstance(data["age"], int):
            raise ValidationError("Age must be an integer", field="age")
        if data["age"] < 0 or data["age"] > 150:
            raise ValidationError("Age must be between 0 and 150", field="age")

    if "email" in data:
        if "@" not in data["email"]:
            raise ValidationError("Invalid email format", field="email")

    return True


class BankAccountWithExceptions:
    """Bank account with proper exception handling."""

    def __init__(self, owner, balance=0):
        if balance < 0:
            raise ValueError("Initial balance cannot be negative")
        self.owner = owner
        self._balance = balance

    def deposit(self, amount):
        """Deposit money with validation."""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._balance += amount
        logger.info(f"Deposited ${amount}. New balance: ${self._balance}")
        return self._balance

    def withdraw(self, amount):
        """Withdraw money with validation."""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self._balance:
            raise InsufficientFundsError(self._balance, amount)
        self._balance -= amount
        logger.info(f"Withdrew ${amount}. New balance: ${self._balance}")
        return self._balance

    @property
    def balance(self):
        """Get current balance."""
        return self._balance


def demonstrate_try_finally():
    """Demonstrate try/finally for cleanup."""
    resources = []

    try:
        resources.append("resource_1")
        resources.append("resource_2")
        # Simulate some processing
        result = 10 / 2
        return result
    finally:
        # Cleanup always runs
        resources.clear()
        logger.debug("Resources cleaned up")


def demonstrate_multiple_exceptions():
    """Demonstrate handling multiple exception types."""
    test_cases = [
        ("10", None),  # Valid
        ("abc", ValueError),  # Invalid string
        (None, TypeError),  # None value
    ]

    results = []
    for value, expected_error in test_cases:
        try:
            result = int(value) if value is not None else int(value)
            results.append({"value": value, "result": result, "error": None})
        except (ValueError, TypeError) as e:
            results.append({"value": value, "result": None, "error": type(e).__name__})

    return results


def reraise_with_context():
    """Demonstrate exception chaining."""
    try:
        try:
            result = 1 / 0
            return result
        except ZeroDivisionError as e:
            raise ValueError("Calculation failed") from e
    except ValueError as e:
        logger.error(f"Error: {e}, caused by: {e.__cause__}")
        return None


if __name__ == "__main__":
    print("=== Error Handling Demo ===")

    # Basic exception handling
    print("\nDivision:")
    print(f"  10 / 2 = {divide(10, 2)}")
    print(f"  10 / 0 = {divide(10, 0)}")

    # Safe conversion
    print("\nSafe int conversion:")
    print(f"  '42' -> {safe_int_conversion('42')}")
    print(f"  'abc' -> {safe_int_conversion('abc')}")

    # List access
    numbers = [1, 2, 3]
    print("\nList access:")
    print(f"  Index 1: {process_list(numbers, 1)}")
    print(f"  Index 10: {process_list(numbers, 10)}")

    # Custom exceptions
    print("\nValidation:")
    valid_data = {"name": "Alice", "age": 30, "email": "alice@example.com"}
    try:
        validate_user_data(valid_data)
        print("  Valid data accepted")
    except ValidationError as e:
        print(f"  Validation error: {e.message}")

    invalid_data = {"name": "", "age": -5}
    try:
        validate_user_data(invalid_data)
    except ValidationError as e:
        print(f"  Validation error on '{e.field}': {e.message}")

    # Bank account with exceptions
    print("\nBank account:")
    account = BankAccountWithExceptions("Bob", 100)
    account.deposit(50)
    try:
        account.withdraw(200)
    except InsufficientFundsError as e:
        print(f"  Error: {e}")

    # Multiple exceptions
    print("\nMultiple exceptions:", demonstrate_multiple_exceptions())

    # Exception chaining
    print("\nException chaining:", reraise_with_context())
