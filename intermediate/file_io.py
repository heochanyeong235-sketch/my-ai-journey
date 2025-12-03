"""
Lesson 02: File I/O

This module covers file operations in Python:
- Reading and writing text files
- Reading and writing binary files
- Working with CSV files
- Working with JSON files
- Context managers
"""

import csv
import io
import json
import os
import tempfile


def write_text_file(filepath, content):
    """Write content to a text file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath


def read_text_file(filepath):
    """Read content from a text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def append_to_file(filepath, content):
    """Append content to a file."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(content)
    return filepath


def read_lines(filepath):
    """Read file line by line."""
    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def write_csv(filepath, data, headers=None):
    """Write data to a CSV file."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if headers:
            writer.writerow(headers)
        writer.writerows(data)
    return filepath


def read_csv(filepath):
    """Read data from a CSV file."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data


def read_csv_dict(filepath):
    """Read CSV as list of dictionaries."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(dict(row))
    return data


def write_json(filepath, data, indent=2):
    """Write data to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    return filepath


def read_json(filepath):
    """Read data from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def write_binary_file(filepath, data):
    """Write binary data to a file."""
    with open(filepath, "wb") as f:
        f.write(data)
    return filepath


def read_binary_file(filepath):
    """Read binary data from a file."""
    with open(filepath, "rb") as f:
        return f.read()


class FileManager:
    """Context manager for file operations."""

    def __init__(self, filepath, mode="r"):
        self.filepath = filepath
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filepath, self.mode, encoding="utf-8")
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False  # Don't suppress exceptions


def demonstrate_file_operations():
    """Demonstrate various file operations."""
    results = {}

    # Create a temporary directory for demos
    with tempfile.TemporaryDirectory() as tmpdir:
        # Text file operations
        text_file = os.path.join(tmpdir, "sample.txt")
        write_text_file(text_file, "Hello, World!\nThis is line 2.\n")
        results["text_content"] = read_text_file(text_file)
        results["lines"] = read_lines(text_file)

        # CSV operations
        csv_file = os.path.join(tmpdir, "data.csv")
        headers = ["name", "age", "city"]
        data = [
            ["Alice", "30", "New York"],
            ["Bob", "25", "San Francisco"],
            ["Charlie", "35", "Chicago"],
        ]
        write_csv(csv_file, data, headers)
        results["csv_data"] = read_csv(csv_file)
        results["csv_dict"] = read_csv_dict(csv_file)

        # JSON operations
        json_file = os.path.join(tmpdir, "config.json")
        config = {
            "app_name": "MyApp",
            "version": "1.0.0",
            "settings": {"debug": True, "max_items": 100},
        }
        write_json(json_file, config)
        results["json_data"] = read_json(json_file)

        # Binary file operations
        binary_file = os.path.join(tmpdir, "data.bin")
        binary_data = bytes([0, 1, 2, 3, 4, 5])
        write_binary_file(binary_file, binary_data)
        results["binary_data"] = list(read_binary_file(binary_file))

    return results


if __name__ == "__main__":
    print("=== File I/O Demo ===")

    results = demonstrate_file_operations()

    print("\nText file content:")
    print(results["text_content"])

    print("Lines:", results["lines"])

    print("\nCSV data:", results["csv_data"])
    print("CSV as dicts:", results["csv_dict"])

    print("\nJSON data:", results["json_data"])

    print("\nBinary data:", results["binary_data"])
