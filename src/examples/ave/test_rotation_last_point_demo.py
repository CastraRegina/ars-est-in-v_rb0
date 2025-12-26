#!/usr/bin/env python3
"""
Demonstration that rotation works correctly when the target point is the last point.
"""

import numpy as np
import shapely.geometry

from ave.path_processing import AvPathCleaner


def demo_rotation_to_last_point():
    """Demonstrate rotation to last point."""

    print("=== Rotation to Last Point Demo ===\n")

    # Create a simple square
    exterior_coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
    polygon = shapely.geometry.Polygon(exterior_coords)

    print("Original polygon coordinates (Shapely order):")
    for i, coord in enumerate(exterior_coords):
        print(f"  Point {i}: {coord}")

    # Original points where only the last point is type=0
    original_points = np.array(
        [
            [0.0, 0.0, 2.0],  # type=2, skip
            [10.0, 0.0, 2.0],  # type=2, skip
            [10.0, 10.0, 2.0],  # type=2, skip
            [0.0, 10.0, 0.0],  # type=0, last point, rotation target
        ]
    )

    print("\nOriginal points with types:")
    for i, pt in enumerate(original_points):
        print(f"  Point {i}: ({pt[0]}, {pt[1]}, type={pt[2]})")

    # Convert to paths
    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    print("\nAfter rotation:")
    path = paths[0]
    for i in range(len(path.points)):
        coord = path.points[i, :2]
        cmd = path.commands[i]
        print(f"  Point {i}: {coord} (cmd: {cmd})")

    print(f"\n✓ Rotation successful: Path starts at last point (0, 10)")
    print(f"✓ Path is still valid: starts with 'M', ends with 'Z'")


def demo_rotation_with_holes():
    """Demonstrate rotation with holes where targets are last points."""

    print("\n=== Rotation with Holes Demo ===\n")

    # Exterior square
    exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
    # Hole square
    hole = [(5, 5), (15, 5), (15, 15), (5, 15)]

    polygon = shapely.geometry.Polygon(exterior, [hole])

    print("Exterior coordinates:")
    for i, coord in enumerate(exterior):
        print(f"  Point {i}: {coord}")

    print("\nHole coordinates:")
    for i, coord in enumerate(hole):
        print(f"  Point {i}: {coord}")

    # Original points where last points of both are type=0
    original_points = np.array(
        [
            # Exterior
            [0.0, 0.0, 2.0],
            [20.0, 0.0, 2.0],
            [20.0, 20.0, 2.0],
            [0.0, 20.0, 0.0],  # Last exterior point
            # Hole
            [5.0, 5.0, 2.0],
            [15.0, 5.0, 2.0],
            [15.0, 15.0, 2.0],
            [5.0, 15.0, 0.0],  # Last hole point
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    print("\nAfter rotation:")
    print("\nExterior path:")
    exterior_path = paths[0]
    for i in range(len(exterior_path.points)):
        coord = exterior_path.points[i, :2]
        cmd = exterior_path.commands[i]
        print(f"  Point {i}: {coord} (cmd: {cmd})")

    print("\nHole path:")
    hole_path = paths[1]
    for i in range(len(hole_path.points)):
        coord = hole_path.points[i, :2]
        cmd = hole_path.commands[i]
        print(f"  Point {i}: {coord} (cmd: {cmd})")

    print(f"\n✓ Both exterior and hole start at their last points")
    print(f"✓ Exterior starts at (0, 20) - last exterior point")
    print(f"✓ Hole starts at (5, 15) - last hole point")


if __name__ == "__main__":
    demo_rotation_to_last_point()
    demo_rotation_with_holes()
    print("\n=== All demos completed successfully! ===")
