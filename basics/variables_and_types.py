"""
Lesson 01: Variables and Data Types

This module covers basic Python variables and data types.
"""


def demonstrate_integers():
    """Demonstrate integer operations."""
    x = 10
    y = 3
    
    addition = x + y
    subtraction = x - y
    multiplication = x * y
    division = x / y
    floor_division = x // y
    modulo = x % y
    exponentiation = x ** y
    
    return {
        "addition": addition,
        "subtraction": subtraction,
        "multiplication": multiplication,
        "division": division,
        "floor_division": floor_division,
        "modulo": modulo,
        "exponentiation": exponentiation,
    }


def demonstrate_floats():
    """Demonstrate float operations."""
    pi = 3.14159
    e = 2.71828
    
    product = pi * e
    rounded = round(pi, 2)
    
    return {"product": product, "rounded": rounded}


def demonstrate_strings():
    """Demonstrate string operations."""
    greeting = "Hello"
    name = "World"
    
    concatenation = greeting + " " + name
    repetition = greeting * 3
    length = len(concatenation)
    uppercase = concatenation.upper()
    lowercase = concatenation.lower()
    
    return {
        "concatenation": concatenation,
        "repetition": repetition,
        "length": length,
        "uppercase": uppercase,
        "lowercase": lowercase,
    }


def demonstrate_booleans():
    """Demonstrate boolean operations."""
    is_true = True
    is_false = False
    
    and_result = is_true and is_false
    or_result = is_true or is_false
    not_result = not is_true
    
    return {"and": and_result, "or": or_result, "not": not_result}


def demonstrate_type_conversion():
    """Demonstrate type conversion."""
    integer = 42
    float_val = 3.14
    string = "100"
    
    int_to_float = float(integer)
    float_to_int = int(float_val)
    string_to_int = int(string)
    int_to_string = str(integer)
    
    return {
        "int_to_float": int_to_float,
        "float_to_int": float_to_int,
        "string_to_int": string_to_int,
        "int_to_string": int_to_string,
    }


if __name__ == "__main__":
    print("=== Variables and Types Demo ===")
    print("\nIntegers:", demonstrate_integers())
    print("\nFloats:", demonstrate_floats())
    print("\nStrings:", demonstrate_strings())
    print("\nBooleans:", demonstrate_booleans())
    print("\nType Conversion:", demonstrate_type_conversion())
