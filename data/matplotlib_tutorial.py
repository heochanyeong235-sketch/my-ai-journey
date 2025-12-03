"""
Matplotlib Tutorial

This module covers Matplotlib for data visualization:
- Basic plots
- Customization
- Multiple subplots
- Different chart types
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np


def line_plot_demo(save_path=None):
    """Demonstrate line plots."""
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, y1, label="sin(x)", color="blue", linestyle="-", linewidth=2)
    ax.plot(x, y2, label="cos(x)", color="red", linestyle="--", linewidth=2)

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_title("Sine and Cosine Functions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.close(fig)
    return fig


def scatter_plot_demo(save_path=None):
    """Demonstrate scatter plots."""
    np.random.seed(42)
    x = np.random.randn(100)
    y = 2 * x + np.random.randn(100) * 0.5
    colors = np.random.rand(100)
    sizes = np.abs(np.random.randn(100)) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap="viridis")

    ax.set_xlabel("X values")
    ax.set_ylabel("Y values")
    ax.set_title("Scatter Plot with Colors and Sizes")
    plt.colorbar(scatter, ax=ax, label="Color value")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.close(fig)
    return fig


def bar_chart_demo(save_path=None):
    """Demonstrate bar charts."""
    categories = ["A", "B", "C", "D", "E"]
    values1 = [25, 40, 30, 55, 45]
    values2 = [20, 35, 40, 45, 35]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width / 2, values1, width, label="Group 1", color="steelblue")
    bars2 = ax.bar(x + width / 2, values2, width, label="Group 2", color="coral")

    ax.set_xlabel("Categories")
    ax.set_ylabel("Values")
    ax.set_title("Grouped Bar Chart")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.close(fig)
    return fig


def histogram_demo(save_path=None):
    """Demonstrate histograms."""
    np.random.seed(42)
    data = np.random.randn(1000)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Basic histogram
    axes[0].hist(data, bins=30, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Basic Histogram")

    # Histogram with KDE-like overlay
    axes[1].hist(data, bins=30, density=True, edgecolor="black", alpha=0.7)
    x = np.linspace(-4, 4, 100)
    axes[1].plot(
        x,
        1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2),
        "r-",
        linewidth=2,
        label="Normal PDF",
    )
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Normalized Histogram with Normal Distribution")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.close(fig)
    return fig


def pie_chart_demo(save_path=None):
    """Demonstrate pie charts."""
    labels = ["Python", "JavaScript", "Java", "C++", "Other"]
    sizes = [35, 25, 20, 10, 10]
    explode = (0.1, 0, 0, 0, 0)  # Explode first slice
    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#ff99cc"]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )

    ax.set_title("Programming Language Popularity")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.close(fig)
    return fig


def subplots_demo(save_path=None):
    """Demonstrate multiple subplots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: Line plot
    x = np.linspace(0, 10, 100)
    axes[0, 0].plot(x, np.sin(x))
    axes[0, 0].set_title("Sine Wave")

    # Top-right: Scatter plot
    np.random.seed(42)
    axes[0, 1].scatter(np.random.randn(50), np.random.randn(50))
    axes[0, 1].set_title("Random Scatter")

    # Bottom-left: Bar chart
    categories = ["A", "B", "C", "D"]
    values = [15, 30, 25, 40]
    axes[1, 0].bar(categories, values, color="steelblue")
    axes[1, 0].set_title("Bar Chart")

    # Bottom-right: Histogram
    data = np.random.randn(500)
    axes[1, 1].hist(data, bins=20, edgecolor="black")
    axes[1, 1].set_title("Histogram")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.close(fig)
    return fig


def heatmap_demo(save_path=None):
    """Demonstrate heatmaps."""
    np.random.seed(42)
    data = np.random.rand(8, 8)

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(data, cmap="YlOrRd")

    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels([f"Col{i}" for i in range(8)])
    ax.set_yticklabels([f"Row{i}" for i in range(8)])

    plt.colorbar(im, ax=ax)
    ax.set_title("Heatmap")

    # Add text annotations
    for i in range(8):
        for j in range(8):
            text = ax.text(
                j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black"
            )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.close(fig)
    return fig


def box_plot_demo(save_path=None):
    """Demonstrate box plots."""
    np.random.seed(42)
    data = [np.random.randn(100) + i for i in range(4)]

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(data, patch_artist=True)

    colors = ["lightblue", "lightgreen", "lightyellow", "lightpink"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_xlabel("Group")
    ax.set_ylabel("Value")
    ax.set_title("Box Plot Comparison")
    ax.set_xticklabels(["Group A", "Group B", "Group C", "Group D"])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.close(fig)
    return fig


if __name__ == "__main__":
    print("=== Matplotlib Tutorial ===")

    # Save plots to files
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        print("\nGenerating plots...")

        line_path = os.path.join(tmpdir, "line_plot.png")
        line_plot_demo(line_path)
        print(f"  Line plot saved to {line_path}")

        scatter_path = os.path.join(tmpdir, "scatter_plot.png")
        scatter_plot_demo(scatter_path)
        print(f"  Scatter plot saved to {scatter_path}")

        bar_path = os.path.join(tmpdir, "bar_chart.png")
        bar_chart_demo(bar_path)
        print(f"  Bar chart saved to {bar_path}")

        hist_path = os.path.join(tmpdir, "histogram.png")
        histogram_demo(hist_path)
        print(f"  Histogram saved to {hist_path}")

        pie_path = os.path.join(tmpdir, "pie_chart.png")
        pie_chart_demo(pie_path)
        print(f"  Pie chart saved to {pie_path}")

        subplots_path = os.path.join(tmpdir, "subplots.png")
        subplots_demo(subplots_path)
        print(f"  Subplots saved to {subplots_path}")

        heatmap_path = os.path.join(tmpdir, "heatmap.png")
        heatmap_demo(heatmap_path)
        print(f"  Heatmap saved to {heatmap_path}")

        box_path = os.path.join(tmpdir, "box_plot.png")
        box_plot_demo(box_path)
        print(f"  Box plot saved to {box_path}")

        print("\nAll plots generated successfully!")
