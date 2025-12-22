#!/usr/bin/env python3
"""Benchmark script for cubic Bezier control point approximation performance."""

import timeit
from typing import List

import numpy as np

from ave.bezier import BezierCurve


def generate_test_data(num_points: int) -> np.ndarray:
    """Generate test points along a cubic Bezier curve."""
    start = np.array([0.0, 0.0], dtype=np.float64)
    ctrl1 = np.array([1.0, 3.0], dtype=np.float64)
    ctrl2 = np.array([2.0, 3.0], dtype=np.float64)
    end = np.array([3.0, 0.0], dtype=np.float64)

    control_points = np.array([start, ctrl1, ctrl2, end], dtype=np.float64)
    return BezierCurve.polygonize_cubic_curve(control_points, steps=num_points - 1)


def benchmark_current_implementation(sizes: List[int], repeats: int = 100) -> dict:
    """Benchmark current implementation across different input sizes."""
    benchmark_results = {}

    print(f"{'Size':>6} | {'Time (ms)':>10} | {'Throughput':>12}")
    print("-" * 38)

    for size in sizes:
        test_data = generate_test_data(size)

        timer = timeit.Timer(lambda d=test_data: BezierCurve.approximate_cubic_control_points(d))
        total_time = timer.timeit(number=repeats)
        avg_time_ms = (total_time / repeats) * 1000
        throughput = size / (total_time / repeats) if total_time > 0 else float("inf")

        benchmark_results[size] = {"avg_time_ms": avg_time_ms, "throughput_points_per_sec": throughput}

        print(f"{size:6d} | {avg_time_ms:10.4f} | {throughput:12.0f}")

    return benchmark_results


if __name__ == "__main__":
    test_sizes = [10, 20, 50, 80, 100, 150, 200, 300, 500]

    print("Benchmarking current implementation...")
    print("Each test repeated 100 times")
    print()

    results = benchmark_current_implementation(test_sizes)

    print()
    print("Summary:")
    print(f"Smallest (10 points):  {results[10]['avg_time_ms']:.4f} ms")
    print(f"Largest (500 points):  {results[500]['avg_time_ms']:.4f} ms")
    print(f"Scaling factor:        {results[500]['avg_time_ms'] / results[10]['avg_time_ms']:.2f}x")
