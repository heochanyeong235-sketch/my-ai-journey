"""
Lesson 04: Control Flow

This module covers Python control flow:
- If/elif/else statements
- For loops
- While loops
- Break, continue, pass
- Match statements (Python 3.10+)
"""


def check_number(n):
    """Demonstrate if/elif/else."""
    if n > 0:
        return "positive"
    elif n < 0:
        return "negative"
    else:
        return "zero"


def get_grade(score):
    """Convert score to letter grade."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


def demonstrate_for_loops():
    """Demonstrate for loop variations."""
    results = {}

    # Basic for loop
    squares = []
    for i in range(1, 6):
        squares.append(i ** 2)
    results["squares"] = squares

    # Loop with enumerate
    fruits = ["apple", "banana", "cherry"]
    indexed = []
    for idx, fruit in enumerate(fruits):
        indexed.append((idx, fruit))
    results["indexed"] = indexed

    # Loop with zip
    names = ["Alice", "Bob"]
    ages = [25, 30]
    combined = []
    for name, age in zip(names, ages):
        combined.append(f"{name}: {age}")
    results["combined"] = combined

    # Nested loops
    matrix = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(i * 3 + j)
        matrix.append(row)
    results["matrix"] = matrix

    return results


def demonstrate_while_loop():
    """Demonstrate while loop."""
    # Basic while
    count = 0
    sequence = []
    while count < 5:
        sequence.append(count)
        count += 1

    # While with condition
    n = 1
    powers_of_2 = []
    while n <= 100:
        powers_of_2.append(n)
        n *= 2

    return {"sequence": sequence, "powers_of_2": powers_of_2}


def find_first_even(numbers):
    """Demonstrate break statement."""
    for n in numbers:
        if n % 2 == 0:
            return n  # break equivalent
    return None


def get_odd_numbers(numbers):
    """Demonstrate continue statement."""
    odds = []
    for n in numbers:
        if n % 2 == 0:
            continue
        odds.append(n)
    return odds


def fizzbuzz(n):
    """Classic FizzBuzz implementation."""
    results = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            results.append("FizzBuzz")
        elif i % 3 == 0:
            results.append("Fizz")
        elif i % 5 == 0:
            results.append("Buzz")
        else:
            results.append(str(i))
    return results


def get_day_type(day):
    """Demonstrate match statement (Python 3.10+)."""
    match day.lower():
        case "saturday" | "sunday":
            return "weekend"
        case "monday" | "tuesday" | "wednesday" | "thursday" | "friday":
            return "weekday"
        case _:
            return "unknown"


if __name__ == "__main__":
    print("=== Control Flow Demo ===")

    # If/elif/else
    print("\nNumber check:")
    for n in [-5, 0, 10]:
        print(f"  {n}: {check_number(n)}")

    # Grades
    print("\nGrades:")
    for score in [95, 85, 75, 65, 55]:
        print(f"  {score}: {get_grade(score)}")

    # For loops
    print("\nFor loops:", demonstrate_for_loops())

    # While loops
    print("\nWhile loops:", demonstrate_while_loop())

    # Break and continue
    numbers = [1, 3, 5, 6, 7, 8]
    print(f"\nFirst even in {numbers}: {find_first_even(numbers)}")
    print(f"Odd numbers from {numbers}: {get_odd_numbers(numbers)}")

    # FizzBuzz
    print("\nFizzBuzz (1-15):", fizzbuzz(15))

    # Match statement
    print("\nDay types:")
    for day in ["Monday", "Saturday", "Holiday"]:
        print(f"  {day}: {get_day_type(day)}")
