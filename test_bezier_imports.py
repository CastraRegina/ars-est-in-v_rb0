#!/usr/bin/env python3
"""Test script to verify BezierCurve imports work after renaming."""

import sys

sys.path.insert(0, ".")


def test_bezier_functionality():
    """Test that BezierCurve functionality works after module rename."""

    print("Testing BezierCurve after module rename...")

    # Test imports
    from ave.bezier import BezierCurve
    from ave.path import AvPath

    print("✓ All imports successful")

    # Test quadratic curve functionality
    points = [(0.0, 0.0), (50.0, 200.0), (200.0, 0.0)]
    result = BezierCurve.polygonize_quadratic_curve(points, 10)
    assert len(result) == 11
    print("✓ Quadratic curve polygonization works")

    # Test cubic curve functionality
    points_cubic = [(0.0, 0.0), (50.0, 200.0), (150.0, -100.0), (200.0, 0.0)]
    result_cubic = BezierCurve.polygonize_cubic_curve(points_cubic, 10)
    assert len(result_cubic) == 11
    print("✓ Cubic curve polygonization works")

    # Test in-place methods
    import numpy as np

    buffer = np.zeros((20, 3), dtype=np.float64)  # 3D buffer (x, y, z)
    count = BezierCurve.polygonize_quadratic_curve_python_inplace(points, 10, buffer, start_index=0, skip_first=False)
    assert count == 11
    print("✓ In-place quadratic polygonization works")

    print("\n✅ All BezierCurve functionality verified after module rename!")
    return True


if __name__ == "__main__":
    try:
        test_bezier_functionality()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
