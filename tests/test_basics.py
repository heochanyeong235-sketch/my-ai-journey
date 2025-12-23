"""
Tests for the basics module.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basics.variables_and_types import (
    demonstrate_integers,
    demonstrate_floats,
    demonstrate_strings,
    demonstrate_booleans,
    demonstrate_type_conversion,
)
from basics.data_structures import (
    demonstrate_lists,
    demonstrate_tuples,
    demonstrate_dictionaries,
    demonstrate_sets,
)
from basics.functions import (
    greet,
    add,
    power,
    sum_all,
    fibonacci,
    create_multiplier,
)
from basics.control_flow import (
    check_number,
    get_grade,
    fizzbuzz,
    find_first_even,
    get_odd_numbers,
)


class TestVariablesAndTypes:
    """Tests for variables and types module."""

    def test_demonstrate_integers(self):
        result = demonstrate_integers()
        assert result["addition"] == 13
        assert result["subtraction"] == 7
        assert result["multiplication"] == 30
        assert result["floor_division"] == 3
        assert result["modulo"] == 1
        assert result["exponentiation"] == 1000

    def test_demonstrate_floats(self):
        result = demonstrate_floats()
        assert result["rounded"] == 3.14

    def test_demonstrate_strings(self):
        result = demonstrate_strings()
        assert result["concatenation"] == "Hello World"
        assert result["repetition"] == "HelloHelloHello"
        assert result["length"] == 11

    def test_demonstrate_booleans(self):
        result = demonstrate_booleans()
        assert result["and"] is False
        assert result["or"] is True
        assert result["not"] is False

    def test_demonstrate_type_conversion(self):
        result = demonstrate_type_conversion()
        assert result["int_to_float"] == 42.0
        assert result["float_to_int"] == 3
        assert result["string_to_int"] == 100
        assert result["int_to_string"] == "42"


class TestDataStructures:
    """Tests for data structures module."""

    def test_demonstrate_lists(self):
        result = demonstrate_lists()
        assert "date" in result["fruits"]
        assert result["first"] == "apple"
        assert result["squares"] == [1, 4, 9, 16, 25]
        assert result["evens"] == [2, 4]

    def test_demonstrate_tuples(self):
        result = demonstrate_tuples()
        assert result["x"] == 10
        assert result["y"] == 20
        assert result["z"] == 30
        assert result["is_immutable"] is True

    def test_demonstrate_dictionaries(self):
        result = demonstrate_dictionaries()
        assert result["name"] == "Alice"
        assert result["squares_dict"] == {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

    def test_demonstrate_sets(self):
        result = demonstrate_sets()
        assert result["intersection"] == {4, 5}
        assert result["difference"] == {1, 2, 3}
        assert result["contains_3"] is True
        assert result["contains_10"] is False


class TestFunctions:
    """Tests for functions module."""

    def test_greet(self):
        assert greet("Alice") == "Hello, Alice!"
        assert greet("World") == "Hello, World!"

    def test_add(self):
        assert add(2, 3) == 5
        assert add(-1, 1) == 0
        assert add(0, 0) == 0

    def test_power(self):
        assert power(2, 3) == 8
        assert power(2) == 4  # Default exponent
        assert power(10, 2) == 100

    def test_sum_all(self):
        assert sum_all(1, 2, 3) == 6
        assert sum_all(10, 20, 30, 40) == 100
        assert sum_all() == 0

    def test_fibonacci(self):
        assert fibonacci(0) == 0
        assert fibonacci(1) == 1
        assert fibonacci(10) == 55
        assert fibonacci(20) == 6765

    def test_create_multiplier(self):
        double = create_multiplier(2)
        triple = create_multiplier(3)
        assert double(5) == 10
        assert triple(5) == 15


class TestControlFlow:
    """Tests for control flow module."""

    def test_check_number(self):
        assert check_number(5) == "positive"
        assert check_number(-5) == "negative"
        assert check_number(0) == "zero"

    def test_get_grade(self):
        assert get_grade(95) == "A"
        assert get_grade(85) == "B"
        assert get_grade(75) == "C"
        assert get_grade(65) == "D"
        assert get_grade(55) == "F"

    def test_fizzbuzz(self):
        result = fizzbuzz(15)
        assert result[2] == "Fizz"  # 3
        assert result[4] == "Buzz"  # 5
        assert result[14] == "FizzBuzz"  # 15

    def test_find_first_even(self):
        assert find_first_even([1, 3, 5, 6, 7]) == 6
        assert find_first_even([1, 3, 5]) is None
        assert find_first_even([2, 4, 6]) == 2

    def test_get_odd_numbers(self):
        assert get_odd_numbers([1, 2, 3, 4, 5]) == [1, 3, 5]
        assert get_odd_numbers([2, 4, 6]) == []
        assert get_odd_numbers([1, 3, 5]) == [1, 3, 5]
