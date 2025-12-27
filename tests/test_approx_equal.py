"""Test and demonstrate the approx_equal methods for AvGlyph and AvPath."""

import math

import numpy as np

from ave.glyph import AvGlyph
from ave.path import AvPath


def test_approx_equal():
    """Test the approx_equal methods with various scenarios."""

    # Test 1: Identical paths
    points1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [1.0, 1.0, 2.0], [0.0, 1.0, 0.0]])
    commands1 = ["M", "L", "L", "L", "Z"]
    path1 = AvPath(points1, commands1)
    path2 = AvPath(points1.copy(), commands1.copy())

    assert path1.approx_equal(path2), "Identical paths should be equal"
    print("✓ Test 1: Identical paths")

    # Test 2: Paths with small numerical differences
    points2 = points1 + np.array([1e-10, 1e-10, 0.0])  # Very small difference
    path3 = AvPath(points2, commands1)

    assert path1.approx_equal(path3), "Paths with tiny differences should be equal"
    print("✓ Test 2: Paths with small numerical differences")

    # Test 3: Paths with larger differences (should not be equal)
    points3 = points1 + np.array([1e-3, 1e-3, 0.0])  # Larger difference
    path4 = AvPath(points3, commands1)

    assert not path1.approx_equal(path4), "Paths with larger differences should not be equal"
    print("✓ Test 3: Paths with larger differences")

    # Test 4: Custom tolerance (using absolute tolerance for zero values)
    assert path1.approx_equal(path4, atol=1e-2), "Should be equal with relaxed absolute tolerance"
    print("✓ Test 4: Custom tolerance")

    # Test 5: Different commands
    commands2 = ["M", "L", "L", "L"]  # Missing Z
    path5 = AvPath(points1, commands2)

    assert not path1.approx_equal(path5), "Paths with different commands should not be equal"
    print("✓ Test 5: Different commands")

    # Test 6: Glyph comparison
    glyph1 = AvGlyph("A", 1000.0, path1)
    glyph2 = AvGlyph("A", 1000.0, path2)
    glyph3 = AvGlyph("A", 1000.000001, path2)  # Slightly different width
    glyph4 = AvGlyph("B", 1000.0, path2)  # Different character

    assert glyph1.approx_equal(glyph2), "Identical glyphs should be equal"
    print("✓ Test 6a: Identical glyphs")

    assert glyph1.approx_equal(glyph3), "Glyphs with small width differences should be equal"
    print("✓ Test 6b: Glyphs with small width differences")

    assert not glyph1.approx_equal(glyph4), "Glyphs with different characters should not be equal"
    print("✓ Test 6c: Glyphs with different characters")

    print("\nAll tests passed!")


def usage_examples():
    """Show usage examples of approx_equal methods."""

    print("\n=== Usage Examples ===\n")

    # Example 1: Basic usage
    path_a = AvPath([[0, 0], [1, 0], [1, 1], [0, 1]], ["M", "L", "L", "L", "Z"])
    path_b = AvPath([[0, 0], [1, 0], [1, 1], [0, 1]], ["M", "L", "L", "L", "Z"])

    print(f"path_a.approx_equal(path_b) = {path_a.approx_equal(path_b)}")

    # Example 2: With tolerance
    path_c = AvPath([[0, 0], [1.000001, 0], [1, 1], [0, 1]], ["M", "L", "L", "L", "Z"])
    print(f"path_a.approx_equal(path_c) = {path_a.approx_equal(path_c)}")
    print(f"path_a.approx_equal(path_c, rtol=1e-5) = {path_a.approx_equal(path_c, rtol=1e-5)}")

    # Example 3: Glyph comparison
    glyph_a = AvGlyph("A", 1000.0, path_a)
    glyph_b = AvGlyph("A", 1000.000001, path_c)

    print(f"\nglyph_a.approx_equal(glyph_b) = {glyph_a.approx_equal(glyph_b)}")

    # Example 4: In a test context
    def test_glyph_transformation(glyph, expected_glyph):
        """Example test function using approx_equal."""
        transformed = glyph  # Some transformation would happen here
        assert transformed.approx_equal(
            expected_glyph, rtol=1e-6
        ), f"Transformed glyph doesn't match expected (rtol=1e-6)"
        print("✓ Glyph transformation test passed")

    # This would normally involve actual transformations
    test_glyph_transformation(glyph_a, glyph_b)


if __name__ == "__main__":
    test_approx_equal()
    usage_examples()
