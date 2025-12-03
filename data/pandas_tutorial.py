"""
Pandas Tutorial

This module covers Pandas fundamentals:
- DataFrames and Series
- Data loading and saving
- Data selection and filtering
- Data manipulation
- Aggregation and grouping
"""

import io

import pandas as pd


def create_dataframe():
    """Demonstrate DataFrame creation."""
    # From dictionary
    data = {
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 28, 32],
        "city": ["New York", "San Francisco", "Chicago", "Boston", "Seattle"],
        "salary": [70000, 85000, 90000, 75000, 88000],
    }
    df = pd.DataFrame(data)

    # From list of dictionaries
    list_data = [
        {"product": "A", "price": 100},
        {"product": "B", "price": 200},
        {"product": "C", "price": 150},
    ]
    df2 = pd.DataFrame(list_data)

    return {"employees": df, "products": df2}


def series_operations():
    """Demonstrate Series operations."""
    # Create Series
    s = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])

    # Basic operations
    operations = {
        "series": s,
        "sum": s.sum(),
        "mean": s.mean(),
        "std": s.std(),
        "min": s.min(),
        "max": s.max(),
        "cumsum": s.cumsum(),
        "pct_change": s.pct_change(),
    }

    return operations


def data_selection():
    """Demonstrate data selection methods."""
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"],
        },
        index=["row1", "row2", "row3", "row4", "row5"],
    )

    selections = {
        "original": df,
        "column_A": df["A"],
        "columns_A_B": df[["A", "B"]],
        "row_by_loc": df.loc["row2"],
        "row_by_iloc": df.iloc[1],
        "slice_loc": df.loc["row2":"row4"],
        "slice_iloc": df.iloc[1:4],
        "cell_value": df.loc["row3", "B"],
        "boolean_filter": df[df["A"] > 2],
        "multiple_conditions": df[(df["A"] > 1) & (df["B"] < 50)],
    }

    return selections


def data_manipulation():
    """Demonstrate data manipulation."""
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "salary": [70000, 85000, 90000],
        }
    )

    # Add column
    df["bonus"] = df["salary"] * 0.1

    # Apply function
    df["age_group"] = df["age"].apply(lambda x: "young" if x < 30 else "senior")

    # Rename columns
    df_renamed = df.rename(columns={"name": "employee_name"})

    # Sort values
    df_sorted = df.sort_values("salary", ascending=False)

    # Drop column
    df_dropped = df.drop("bonus", axis=1)

    return {
        "modified": df,
        "renamed": df_renamed,
        "sorted": df_sorted,
        "dropped": df_dropped,
    }


def handle_missing_data():
    """Demonstrate handling missing data."""
    df = pd.DataFrame(
        {
            "A": [1, 2, None, 4, 5],
            "B": [None, 2, 3, None, 5],
            "C": [1, 2, 3, 4, None],
        }
    )

    results = {
        "original": df,
        "isna": df.isna(),
        "dropna": df.dropna(),
        "fillna_zero": df.fillna(0),
        "fillna_mean": df.fillna(df.mean(numeric_only=True)),
        "fillna_forward": df.ffill(),
    }

    return results


def groupby_operations():
    """Demonstrate groupby operations."""
    df = pd.DataFrame(
        {
            "category": ["A", "B", "A", "B", "A", "B"],
            "subcategory": ["X", "X", "Y", "Y", "X", "Y"],
            "value": [10, 20, 30, 40, 50, 60],
            "quantity": [1, 2, 3, 4, 5, 6],
        }
    )

    # Basic groupby
    grouped_sum = df.groupby("category")["value"].sum()
    grouped_mean = df.groupby("category")["value"].mean()

    # Multiple aggregations
    agg_result = df.groupby("category").agg(
        {"value": ["sum", "mean", "max"], "quantity": ["sum", "count"]}
    )

    # Multiple groupby columns
    multi_group = df.groupby(["category", "subcategory"])["value"].sum()

    return {
        "original": df,
        "grouped_sum": grouped_sum,
        "grouped_mean": grouped_mean,
        "agg_result": agg_result,
        "multi_group": multi_group,
    }


def merge_and_join():
    """Demonstrate merging and joining DataFrames."""
    df1 = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})

    df2 = pd.DataFrame({"id": [1, 2, 4], "salary": [70000, 85000, 90000]})

    # Different merge types
    inner_merge = pd.merge(df1, df2, on="id", how="inner")
    left_merge = pd.merge(df1, df2, on="id", how="left")
    right_merge = pd.merge(df1, df2, on="id", how="right")
    outer_merge = pd.merge(df1, df2, on="id", how="outer")

    # Concatenation
    df3 = pd.DataFrame({"id": [5, 6], "name": ["David", "Eve"]})
    concat_result = pd.concat([df1, df3], ignore_index=True)

    return {
        "df1": df1,
        "df2": df2,
        "inner_merge": inner_merge,
        "left_merge": left_merge,
        "right_merge": right_merge,
        "outer_merge": outer_merge,
        "concat": concat_result,
    }


def pivot_and_reshape():
    """Demonstrate pivot and reshape operations."""
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "category": ["A", "B", "A", "B"],
            "value": [10, 20, 30, 40],
        }
    )

    # Pivot table
    pivot = df.pivot_table(values="value", index="date", columns="category", aggfunc="sum")

    # Melt (unpivot)
    wide_df = pd.DataFrame(
        {"name": ["Alice", "Bob"], "math": [90, 85], "science": [88, 92]}
    )
    melted = pd.melt(wide_df, id_vars=["name"], var_name="subject", value_name="score")

    return {"original": df, "pivot": pivot, "wide": wide_df, "melted": melted}


def datetime_operations():
    """Demonstrate datetime operations."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "value": [10, 20, 15, 30, 25, 35, 40, 30, 45, 50],
        }
    )

    # Extract datetime components
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek

    # Set datetime index
    df_indexed = df.set_index("date")

    # Resample (aggregate by time period)
    resampled = df_indexed["value"].resample("3D").sum()

    # Rolling statistics
    df["rolling_mean"] = df["value"].rolling(window=3).mean()

    return {"df": df, "resampled": resampled}


if __name__ == "__main__":
    print("=== Pandas Tutorial ===")

    print("\nDataFrame Creation:")
    dfs = create_dataframe()
    print(dfs["employees"].head())

    print("\nSeries Operations:")
    series_ops = series_operations()
    print(f"  Sum: {series_ops['sum']}")
    print(f"  Mean: {series_ops['mean']}")

    print("\nData Selection:")
    sel = data_selection()
    print("  Boolean filter (A > 2):")
    print(sel["boolean_filter"])

    print("\nData Manipulation:")
    manip = data_manipulation()
    print(manip["modified"])

    print("\nHandling Missing Data:")
    missing = handle_missing_data()
    print("  Original:\n", missing["original"])
    print("  After fillna(mean):\n", missing["fillna_mean"])

    print("\nGroupBy Operations:")
    grp = groupby_operations()
    print("  Sum by category:\n", grp["grouped_sum"])

    print("\nMerge and Join:")
    merge = merge_and_join()
    print("  Inner merge:\n", merge["inner_merge"])

    print("\nPivot and Reshape:")
    pivot = pivot_and_reshape()
    print("  Pivot table:\n", pivot["pivot"])

    print("\nDatetime Operations:")
    dt = datetime_operations()
    print(dt["df"].head())
