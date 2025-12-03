"""
Lesson 02: Data Structures

This module covers Python's built-in data structures:
- Lists
- Tuples
- Dictionaries
- Sets
"""


def demonstrate_lists():
    """Demonstrate list operations."""
    fruits = ["apple", "banana", "cherry"]
    
    # Basic operations
    fruits.append("date")
    first_fruit = fruits[0]
    last_fruit = fruits[-1]
    sliced = fruits[1:3]
    length = len(fruits)
    
    # List comprehension
    numbers = [1, 2, 3, 4, 5]
    squares = [n ** 2 for n in numbers]
    evens = [n for n in numbers if n % 2 == 0]
    
    return {
        "fruits": fruits,
        "first": first_fruit,
        "last": last_fruit,
        "sliced": sliced,
        "length": length,
        "squares": squares,
        "evens": evens,
    }


def demonstrate_tuples():
    """Demonstrate tuple operations."""
    coordinates = (10, 20, 30)
    
    x, y, z = coordinates  # Unpacking
    first = coordinates[0]
    length = len(coordinates)
    
    # Tuples are immutable
    is_immutable = True
    
    return {
        "coordinates": coordinates,
        "x": x,
        "y": y,
        "z": z,
        "first": first,
        "length": length,
        "is_immutable": is_immutable,
    }


def demonstrate_dictionaries():
    """Demonstrate dictionary operations."""
    person = {"name": "Alice", "age": 30, "city": "New York"}
    
    # Accessing values
    name = person["name"]
    age = person.get("age")
    
    # Adding/updating
    person["email"] = "alice@example.com"
    person["age"] = 31
    
    # Dictionary methods
    keys = list(person.keys())
    values = list(person.values())
    items = list(person.items())
    
    # Dictionary comprehension
    squares_dict = {x: x ** 2 for x in range(1, 6)}
    
    return {
        "person": person,
        "name": name,
        "age": age,
        "keys": keys,
        "values": values,
        "items": items,
        "squares_dict": squares_dict,
    }


def demonstrate_sets():
    """Demonstrate set operations."""
    set_a = {1, 2, 3, 4, 5}
    set_b = {4, 5, 6, 7, 8}
    
    # Set operations
    union = set_a | set_b
    intersection = set_a & set_b
    difference = set_a - set_b
    symmetric_diff = set_a ^ set_b
    
    # Membership testing
    contains_3 = 3 in set_a
    contains_10 = 10 in set_a
    
    return {
        "set_a": set_a,
        "set_b": set_b,
        "union": union,
        "intersection": intersection,
        "difference": difference,
        "symmetric_diff": symmetric_diff,
        "contains_3": contains_3,
        "contains_10": contains_10,
    }


if __name__ == "__main__":
    print("=== Data Structures Demo ===")
    print("\nLists:", demonstrate_lists())
    print("\nTuples:", demonstrate_tuples())
    print("\nDictionaries:", demonstrate_dictionaries())
    print("\nSets:", demonstrate_sets())
