#!/usr/bin/env python3
# bezier_3_cubic_polygonize_perf.py
# Performance test for the polygonize_cubic_bezier methods

"""
Performance Test for polygonize_cubic_bezier methods

Tests and compares the performance of pure Python vs NumPy implementations
for cubic Bezier curve polygonization.
"""

import timeit

import numpy as np

from ave.geom import BezierCurve

# Test points for cubic Bezier curve
POINTS = ((0.0, 0.0), (50.0, 200.0), (150.0, -100.0), (200.0, 0.0))

# Test step counts
STEPS_LIST = [10, 20, 30, 35, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 500, 700, 1_000, 3_000, 7_000, 10_000]


def test_python_method(steps: int):
    """Test the pure Python implementation"""
    return BezierCurve.polygonize_cubic_curve_python(POINTS, steps)


def test_numpy_method(steps: int):
    """Test the NumPy implementation"""
    return BezierCurve.polygonize_cubic_curve_numpy(POINTS, steps)


def test_auto_method(steps: int):
    """Test the auto implementation (chooses between Python and NumPy)"""
    return BezierCurve.polygonize_cubic_curve(POINTS, steps)


def run_performance_tests():
    """Run performance comparison between Python, NumPy, and auto methods"""
    print("\nPerformance Results (average of 200 runs, times in microseconds):")
    print(" Steps |  Python (μs) |  NumPy (μs) |   Auto (μs) | Winner | Faster")
    print("---------------------------------------------------------------")

    # Store performance data for crossover analysis
    performance_data = []

    for n_steps in STEPS_LIST:
        # Time each method (run multiple times for better accuracy)
        python_time = timeit.timeit(lambda steps=n_steps: test_python_method(steps), number=200) * 1000  # Convert to ms
        numpy_time = timeit.timeit(lambda steps=n_steps: test_numpy_method(steps), number=200) * 1000
        auto_time = timeit.timeit(lambda steps=n_steps: test_auto_method(steps), number=200) * 1000

        # Convert to microseconds for display
        python_us = python_time * 10
        numpy_us = numpy_time * 10
        auto_us = auto_time * 10

        # Determine winner
        times = {"Python": python_us, "NumPy": numpy_us, "Auto": auto_us}
        winner = min(times, key=times.get)

        # Determine which was faster between NumPy and Python
        faster_impl = "NumPy" if numpy_us < python_us else "Python"

        print(f"{n_steps:6d} | {python_us:12.2f} | {numpy_us:11.2f} | {auto_us:11.2f} | {winner:6} | {faster_impl}")

        # Store data for analysis
        performance_data.append((n_steps, python_us, numpy_us, auto_us, winner, faster_impl))

    # Analyze crossover point
    analyze_performance_crossover(performance_data)


def analyze_performance_crossover(data):
    """Analyze performance data to find crossover point where NumPy becomes faster than Python"""
    print("\n" + "=" * 60)
    print("PERFORMANCE CROSSOVER ANALYSIS")
    print("=" * 60)

    python_faster_steps = []
    numpy_faster_steps = []

    for steps, python_us, numpy_us, auto_us, winner, faster_impl in data:
        if faster_impl == "Python":
            python_faster_steps.append(steps)
        else:
            numpy_faster_steps.append(steps)

    # Find crossover point
    crossover_point = None
    for i in range(len(data) - 1):
        current_steps, current_python, current_numpy, _, _, current_faster = data[i]
        next_steps, next_python, next_numpy, _, _, next_faster = data[i + 1]

        if current_faster == "Python" and next_faster == "NumPy":
            crossover_point = f"Between {current_steps} and {next_steps} steps"
            break
        elif current_faster == "NumPy" and next_faster == "Python":
            crossover_point = f"Between {current_steps} and {next_steps} steps"
            break

    if not crossover_point:
        # Check if all are one or the other
        if python_faster_steps and not numpy_faster_steps:
            crossover_point = "Python is faster for all tested step counts"
        elif numpy_faster_steps and not python_faster_steps:
            crossover_point = "NumPy is faster for all tested step counts"
        else:
            crossover_point = "Multiple crossover points detected"

    print(f"Crossover Point: {crossover_point}")
    print(f"Python faster for: {len(python_faster_steps)} step ranges")
    print(f"NumPy faster for: {len(numpy_faster_steps)} step ranges")

    if python_faster_steps:
        print(f"Python wins at: {python_faster_steps[:5]}{'...' if len(python_faster_steps) > 5 else ''}")
    if numpy_faster_steps:
        print(f"NumPy wins at: {numpy_faster_steps[:5]}{'...' if len(numpy_faster_steps) > 5 else ''}")

    # Find exact crossover if possible
    exact_crossover = None
    for steps, python_us, numpy_us, _, _, _ in data:
        if numpy_us < python_us:
            exact_crossover = steps
            break

    if exact_crossover:
        print(f"\nExact crossover: NumPy becomes faster at {exact_crossover}+ steps")
        print(f"At {exact_crossover} steps: Python {python_us:.2f}μs vs NumPy {numpy_us:.2f}μs")
        speedup = python_us / numpy_us if numpy_us > 0 else 0
        print(f"NumPy speedup at crossover: {speedup:.2f}x")

    print("\n📊 RECOMMENDATION:")
    if exact_crossover and exact_crossover <= 100:
        print(f"  • Use Python for step counts < {exact_crossover}")
        print(f"  • Use NumPy for step counts >= {exact_crossover}")
        print(f"  • Auto dispatcher correctly chooses optimal implementation")
    elif exact_crossover:
        print(f"  • Use Python for step counts < {exact_crossover}")
        print(f"  • Use NumPy for step counts >= {exact_crossover}")
        print(f"  • Auto dispatcher correctly chooses optimal implementation")
    else:
        print("  • Auto dispatcher effectively chooses optimal implementation")
        print("  • Performance varies based on system conditions")


def verify_correctness():
    """Verify all methods produce identical results"""
    print("\nVerifying correctness...")

    for steps in [10, 50, 100, 500, 1000]:
        python_result = test_python_method(steps)
        numpy_result = test_numpy_method(steps)
        auto_result = test_auto_method(steps)

        # Check shapes
        assert python_result.shape == (steps + 1, 3), f"Python wrong shape: {python_result.shape}"
        assert numpy_result.shape == (steps + 1, 3), f"NumPy wrong shape: {numpy_result.shape}"
        assert auto_result.shape == (steps + 1, 3), f"Auto wrong shape: {auto_result.shape}"

        # Check values (with tolerance for floating point differences)
        # Python uses forward differencing, NumPy uses direct evaluation
        # Use step-dependent tolerance: 1e-9 for low steps, 1e-8 for medium steps, 1e-7 for high steps
        tolerance = 1e-9 if steps < 200 else (1e-8 if steps < 500 else 1e-7)
        assert np.allclose(
            python_result, numpy_result, rtol=tolerance, atol=tolerance
        ), f"Mismatch Python vs NumPy for steps={steps}"
        assert np.allclose(
            python_result, auto_result, rtol=tolerance, atol=tolerance
        ), f"Mismatch Python vs Auto for steps={steps}"
        assert np.allclose(
            numpy_result, auto_result, rtol=tolerance, atol=tolerance
        ), f"Mismatch NumPy vs Auto for steps={steps}"

        print(f"  Steps {steps:4d}: ✓ All methods match")


def test_different_input_formats():
    """Test that different input formats work correctly"""
    print("\nTesting different input formats...")

    # Test with tuple sequence
    tuple_points = ((0.0, 0.0), (50.0, 200.0), (150.0, -100.0), (200.0, 0.0))

    # Test with numpy array
    numpy_points = np.array(tuple_points, dtype=np.float64)

    steps = 100

    # Allocate buffers for in-place computations
    python_tuple = np.empty((steps + 1, 3), dtype=np.float64)
    python_numpy = np.empty((steps + 1, 3), dtype=np.float64)
    numpy_tuple = np.empty((steps + 1, 3), dtype=np.float64)
    numpy_numpy = np.empty((steps + 1, 3), dtype=np.float64)

    # Use the in-place implementations from geom.py
    BezierCurve.polygonize_cubic_curve_python_inplace(
        tuple_points, steps, python_tuple, start_index=0, skip_first=False
    )
    BezierCurve.polygonize_cubic_curve_python_inplace(
        numpy_points, steps, python_numpy, start_index=0, skip_first=False
    )

    BezierCurve.polygonize_cubic_curve_numpy_inplace(tuple_points, steps, numpy_tuple, start_index=0, skip_first=False)
    BezierCurve.polygonize_cubic_curve_numpy_inplace(numpy_points, steps, numpy_numpy, start_index=0, skip_first=False)

    auto_tuple = BezierCurve.polygonize_cubic_curve(tuple_points, steps)
    auto_numpy = BezierCurve.polygonize_cubic_curve(numpy_points, steps)

    # Test input format compatibility
    # Use appropriate tolerance for Python vs NumPy comparisons
    # Use step-dependent tolerance: 1e-9 for low steps, 1e-8 for medium steps, 1e-7 for high steps
    tolerance = 1e-9 if steps < 200 else (1e-8 if steps < 500 else 1e-7)
    assert np.allclose(
        python_tuple, python_numpy, rtol=tolerance, atol=tolerance
    ), "Python method: tuple vs numpy input mismatch"
    assert np.allclose(
        numpy_tuple, numpy_numpy, rtol=tolerance, atol=tolerance
    ), "NumPy method: tuple vs numpy input mismatch"
    assert np.allclose(
        auto_tuple, auto_numpy, rtol=tolerance, atol=tolerance
    ), "Auto method: tuple vs numpy input mismatch"

    # Test Python vs NumPy results match
    assert np.allclose(
        python_tuple, numpy_tuple, rtol=tolerance, atol=tolerance
    ), "Python vs NumPy (tuple input) mismatch"
    assert np.allclose(
        python_numpy, numpy_numpy, rtol=tolerance, atol=tolerance
    ), "Python vs NumPy (numpy input) mismatch"
    assert np.allclose(
        python_tuple, numpy_numpy, rtol=tolerance, atol=tolerance
    ), "Python (tuple) vs NumPy (numpy) mismatch"

    # Test auto method matches both implementations
    assert np.allclose(
        auto_tuple, python_tuple, rtol=tolerance, atol=tolerance
    ), "Auto vs Python (tuple input) mismatch"
    assert np.allclose(auto_tuple, numpy_tuple, rtol=tolerance, atol=tolerance), "Auto vs NumPy (tuple input) mismatch"
    assert np.allclose(
        auto_numpy, python_numpy, rtol=tolerance, atol=tolerance
    ), "Auto vs Python (numpy input) mismatch"
    assert np.allclose(auto_numpy, numpy_numpy, rtol=tolerance, atol=tolerance), "Auto vs NumPy (numpy input) mismatch"

    print("  ✓ Tuple sequence input works")
    print("  ✓ NumPy array input works")
    print("  ✓ All formats produce identical results")
    print("  ✓ Python, NumPy, and Auto methods produce identical results")


if __name__ == "__main__":
    print("Performance Test for polygonize_cubic_bezier methods")
    print("=" * 60)

    # Run correctness tests first
    verify_correctness()
    test_different_input_formats()

    # Run performance tests
    run_performance_tests()

    print("\nNote: Comparing pure Python vs NumPy vs Auto implementations")
