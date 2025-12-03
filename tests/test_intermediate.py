"""
Tests for the intermediate module.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intermediate.oop import Dog, Cat, Vector, BankAccount, Counter
from intermediate.error_handling import (
    divide,
    safe_int_conversion,
    validate_user_data,
    ValidationError,
    InsufficientFundsError,
    BankAccountWithExceptions,
)


class TestOOP:
    """Tests for OOP module."""

    def test_dog(self):
        dog = Dog("Buddy", 3, "Golden Retriever")
        assert dog.name == "Buddy"
        assert dog.speak() == "Woof!"
        assert "Buddy" in dog.fetch()
        assert "3 years old" in dog.describe()

    def test_cat(self):
        cat = Cat("Whiskers", 5)
        assert cat.name == "Whiskers"
        assert cat.speak() == "Meow!"
        assert "Whiskers" in cat.purr()

    def test_vector(self):
        v1 = Vector(3, 4)
        v2 = Vector(1, 2)

        assert str(v1) == "(3, 4)"
        assert v1 + v2 == Vector(4, 6)
        assert v1 - v2 == Vector(2, 2)
        assert v1 * 2 == Vector(6, 8)
        assert abs(v1) == 5.0

    def test_bank_account(self):
        account = BankAccount("Alice", 100)
        assert account.get_balance() == 100
        account.deposit(50)
        assert account.get_balance() == 150
        account.withdraw(30)
        assert account.get_balance() == 120

    def test_counter(self):
        Counter.count = 0  # Reset
        c1 = Counter("First")
        c2 = Counter("Second")
        assert Counter.get_count() == 2
        assert Counter.validate_name("test") is True
        assert Counter.validate_name("") is False


class TestErrorHandling:
    """Tests for error handling module."""

    def test_divide(self):
        assert divide(10, 2) == 5.0
        assert divide(10, 0) is None

    def test_safe_int_conversion(self):
        assert safe_int_conversion("42") == 42
        assert safe_int_conversion("abc") is None
        assert safe_int_conversion(None) is None

    def test_validate_user_data_valid(self):
        data = {"name": "Alice", "age": 30, "email": "alice@example.com"}
        assert validate_user_data(data) is True

    def test_validate_user_data_missing_name(self):
        try:
            validate_user_data({"age": 30})
            assert False, "Should raise ValidationError"
        except ValidationError as e:
            assert e.field == "name"

    def test_validate_user_data_invalid_email(self):
        try:
            validate_user_data({"name": "Alice", "email": "invalid"})
            assert False, "Should raise ValidationError"
        except ValidationError as e:
            assert e.field == "email"

    def test_bank_account_with_exceptions(self):
        account = BankAccountWithExceptions("Bob", 100)
        account.deposit(50)
        assert account.balance == 150

        try:
            account.withdraw(200)
            assert False, "Should raise InsufficientFundsError"
        except InsufficientFundsError as e:
            assert e.balance == 150
            assert e.amount == 200
