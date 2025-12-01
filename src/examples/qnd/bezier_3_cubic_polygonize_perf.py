#!/usr/bin/env python3
# bezier_3_cubic_polygonize_perf.py
# Performance test for the polygonize_cubic_bezier methods

"""
Performance Test for polygonize_cubic_bezier methods

Tests and compares the performance of pure Python vs NumPy implementations
for cubic Bézier curve polygonization.
"""

import timeit

import numpy as np

from ave.geom import BezierCurve

# Test points for cubic Bézier curve
POINTS = ((0.0, 0.0), (50.0, 200.0), (150.0, -100.0), (200.0, 0.0))

# Test step counts
STEPS_LIST = [10, 20, 30, 35, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 500, 700, 1_000, 3_000, 7_000, 10_000]


def test_python_method(steps: int):
    """Test the pure Python implementation"""
    return BezierCurve.polygonize_cubic_bezier_python(POINTS, steps)


def test_numpy_method(steps: int):
    """Test the NumPy implementation"""
    return BezierCurve.polygonize_cubic_bezier_numpy(POINTS, steps)


def test_auto_method(steps: int):
    """Test the auto implementation (chooses between Python and NumPy)"""
    return BezierCurve.polygonize_cubic_bezier(POINTS, steps)


def run_performance_tests():
    """Run performance comparison between Python, NumPy, and auto methods"""
    print("\nPerformance Results (average of 200 runs, times in microseconds):")
    print(" Steps |  Python (μs) |  NumPy (μs) |   Auto (μs) | Winner")
    print("-------------------------------------------------------")

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

        print(f"{n_steps:6d} | {python_us:12.2f} | {numpy_us:11.2f} | {auto_us:11.2f} | {winner}")


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
        assert np.allclose(python_result, numpy_result, rtol=1e-12), f"Mismatch Python vs NumPy for steps={steps}"
        assert np.allclose(python_result, auto_result, rtol=1e-12), f"Mismatch Python vs Auto for steps={steps}"
        assert np.allclose(numpy_result, auto_result, rtol=1e-12), f"Mismatch NumPy vs Auto for steps={steps}"

        print(f"  Steps {steps:4d}: ✓ All methods match")


def test_different_input_formats():
    """Test that different input formats work correctly"""
    print("\nTesting different input formats...")

    # Test with tuple sequence
    tuple_points = ((0.0, 0.0), (50.0, 200.0), (150.0, -100.0), (200.0, 0.0))

    # Test with numpy array
    numpy_points = np.array(tuple_points, dtype=np.float64)

    steps = 100

    python_tuple = BezierCurve.polygonize_cubic_bezier_python(tuple_points, steps)
    python_numpy = BezierCurve.polygonize_cubic_bezier_python(numpy_points, steps)

    numpy_tuple = BezierCurve.polygonize_cubic_bezier_numpy(tuple_points, steps)
    numpy_numpy = BezierCurve.polygonize_cubic_bezier_numpy(numpy_points, steps)

    auto_tuple = BezierCurve.polygonize_cubic_bezier(tuple_points, steps)
    auto_numpy = BezierCurve.polygonize_cubic_bezier(numpy_points, steps)

    # Test input format compatibility
    assert np.allclose(python_tuple, python_numpy), "Python method: tuple vs numpy input mismatch"
    assert np.allclose(numpy_tuple, numpy_numpy), "NumPy method: tuple vs numpy input mismatch"
    assert np.allclose(auto_tuple, auto_numpy), "Auto method: tuple vs numpy input mismatch"

    # Test Python vs NumPy results match
    assert np.allclose(python_tuple, numpy_tuple), "Python vs NumPy (tuple input) mismatch"
    assert np.allclose(python_numpy, numpy_numpy), "Python vs NumPy (numpy input) mismatch"
    assert np.allclose(python_tuple, numpy_numpy), "Python (tuple) vs NumPy (numpy) mismatch"

    # Test auto method matches both implementations
    assert np.allclose(auto_tuple, python_tuple), "Auto vs Python (tuple input) mismatch"
    assert np.allclose(auto_tuple, numpy_tuple), "Auto vs NumPy (tuple input) mismatch"
    assert np.allclose(auto_numpy, python_numpy), "Auto vs Python (numpy input) mismatch"
    assert np.allclose(auto_numpy, numpy_numpy), "Auto vs NumPy (numpy input) mismatch"

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
